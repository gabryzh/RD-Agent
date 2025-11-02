"""
该工具模块旨在进行环境管理。

目标是为代理（agent）的运行创造一个统一的环境；
- 所有的代码和数据都应包含在一个文件夹内。
"""

# TODO: 将特定于场景的 docker 环境配置移至其他文件夹。

import contextlib
import json
import os
import pickle
import re
import select
import shutil
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generator, Generic, Mapping, Optional, TypeVar, cast

import docker  # type: ignore[import-untyped]
import docker.models  # type: ignore[import-untyped]
import docker.models.containers  # type: ignore[import-untyped]
import docker.types  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import filter_redundant_text
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry


def cleanup_container(container: docker.models.containers.Container | None, context: str = "") -> None:  # type: ignore[no-any-unimported]
    """
    用于清理 Docker 容器的共享辅助函数。
    在删除容器之前总是先停止它。

    参数
    ----------
    container : docker.models.containers.Container 或 None
        需要清理的容器对象，如果为 None 则不执行任何操作。
    context : str
        用于日志记录的额外上下文（例如，“health check”、“GPU test”）。
    """
    if container is not None:
        try:
            # 总是先停止容器 - 如果容器已经停止，stop() 不会引发错误
            container.stop()
            container.remove()
        except Exception as cleanup_error:
            # 记录清理错误，但不掩盖原始异常
            context_str = f" {context}" if context else ""
            logger.warning(f"清理{context_str}容器 {container.id} 失败: {cleanup_error}")


# 将所有卷绑定路径规范化为使用工作区（working_dir）的绝对路径。
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # 使用工作区（working_dir）将相对路径转换为绝对路径。
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # 支持两种格式:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


def pull_image_with_progress(image: str) -> None:
    """
    带进度条地拉取 Docker 镜像。
    """
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    pull_logs = client.pull(image, stream=True, decode=True)
    progress_bars = {}  # 存储每个图层的进度条

    for log in pull_logs:
        if "id" in log and log.get("progressDetail"):
            layer_id = log["id"]
            progress_detail = log["progressDetail"]
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)

            if total:
                if layer_id not in progress_bars:
                    progress_bars[layer_id] = tqdm(total=total, desc=f"图层 {layer_id}", unit="B", unit_scale=True)
                progress_bars[layer_id].n = current
                progress_bars[layer_id].refresh()

        elif "status" in log:
            print(log["status"])

    for pb in progress_bars.values():
        pb.close()


class EnvConf(ExtendedBaseSettings):
    """
    环境配置的基类。
    """
    default_entry: str  # 默认的入口命令
    extra_volumes: dict = {}  # 额外的卷挂载
    running_timeout_period: int | None = 3600  # 运行超时时间（秒），默认1小时
    # 用于支持透明缓存的辅助设置
    enable_cache: bool = True  # 是否启用缓存
    retry_count: int = 5  # Docker 运行的重试次数
    retry_wait_seconds: int = 10  # Docker 运行的重试等待时间（秒）

    model_config = SettingsConfigDict(
        # 允许从环境变量中解析 "None" 字符串为 None 值
        env_parse_none_str="None",
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    环境运行的结果。
    包含标准输出、退出代码和运行时间（秒）。
    """

    stdout: str
    exit_code: int
    running_time: float

    def get_truncated_stdout(self) -> str:
        """
        获取截断并过滤后的标准输出，以便于显示或传递给 LLM。
        """
        return shrink_text(
            filter_redundant_text(self.stdout),
            context_lines=RD_AGENT_SETTINGS.stdout_context_len,
            line_len=RD_AGENT_SETTINGS.stdout_line_len,
        )


class Env(Generic[ASpecificEnvConf]):
    """
    环境的泛型基类。
    我们使用 Pydantic 的 BaseModel 作为设置类，因为它提供了以下特性：
    - 基本的类型检查和验证。
    - 方便地加载和导出信息（例如，可以使用 `pydantic-yaml` 等包）。
    """

    conf: ASpecificEnvConf  # 不同的环境有不同的配置。

    def __init__(self, conf: ASpecificEnvConf):
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        将文件夹压缩成一个 zip 文件。
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

    def unzip_a_file_into_a_folder(self, zip_file_path: str, folder_path: str) -> None:
        """
        将 zip 文件解压到一个文件夹。
        """
        # 解压前清空目标文件夹
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        根据配置准备环境（例如，拉取镜像、创建 conda 环境）。
        这是一个抽象方法，需要子类实现。
        """

    def check_output(
        self, entry: str | None = None, local_path: str = ".", env: dict | None = None, **kwargs: dict
    ) -> str:
        """
        在环境中运行并返回标准输出。

        参数
        ----------
        entry : str | None
            入口命令。如果为 None，则使用默认入口。
        local_path : str | None
            要挂载到 Docker 中的本地路径（主要用于代码）。
        env : dict | None
            要设置的环境变量。

        返回
        -------
            标准输出字符串。
        """
        result = self.run(entry=entry, local_path=local_path, env=env, **kwargs)
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        """
        带重试逻辑的私有运行方法。
        """
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.info(f"运行时间: {end - start} 秒")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"运行时间超过 {self.conf.running_timeout_period} 秒，进程已被终止。"
                    )
                    log_output += f"\n\n运行时间超过 {self.conf.running_timeout_period} 秒，进程已被终止。"
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"运行容器时出错: {e}, 当前尝试次数: {retry_index + 1}, 剩余 {self.conf.retry_count - retry_index - 1} 次重试。"
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # 为了通过 CI 检查

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        **kwargs: dict,
    ) -> EnvResult:
        """
        在环境中运行，并返回包含标准输出、退出代码和运行时间的结果对象。

        参数
        ----------
        entry : str | None
            入口命令。
        local_path : str | None
            本地项目路径。
        env : dict | None
            环境变量。

        返回
        -------
            EnvResult: 包含运行结果的对象。
        """
        running_extra_volume = kwargs.get("running_extra_volume", {})
        if entry is None:
            entry = self.conf.default_entry

        if "|" in entry:
            logger.warning(
                "您正在使用带有管道符 ('|') 的命令。"
                "退出代码将反映管道中最后一个命令的结果。"
            )

        # FIXME: 输入路径和缓存路径在这里是硬编码的。
        def _get_chmod_cmd(workspace_path: str) -> str:
            """
            构造一个命令，用于更改工作区内除了缓存和输入目录之外的所有文件和目录的权限为 777。
            这是为了解决 Docker 容器内以 root 用户创建文件，导致宿主机用户无法访问的问题。
            """
            def _get_path_stem(path: str) -> str | None:
                # 如果输入路径是相对路径，只保留第一个组件
                p = Path(path)
                if not p.is_absolute() and p.parts:
                    return p.parts[0]
                return None

            find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"
            # 排除缓存和输入目录
            for name in [
                _get_path_stem(T("scenarios.data_science.share:scen.cache_path").r()),
                _get_path_stem(T("scenarios.data_science.share:scen.input_path").r()),
            ]:
                find_cmd += f" ! -name {name}"
            chmod_cmd = f"{find_cmd} -exec chmod -R 777 {{}} +"
            return chmod_cmd

        # 为入口命令添加超时和权限修改逻辑
        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"  # sh 命令开始
            + f"{timeout_cmd}; entry_exit_code=$?; "  # 执行带超时的命令并保存退出码
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "  # 修改文件权限
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"  # 以原始命令的退出码退出
            + "'"  # sh 命令结束
        )

        if self.conf.enable_cache:
            result = self.cached_run(entry_add_timeout, local_path, env, running_extra_volume)
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        """
        带缓存的运行方法。
        会缓存输出和文件夹差异，以便下次运行时使用。
        使用 Python 代码内容和运行参数（entry, running_extra_volume）作为哈希的键。
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        # 生成缓存键
        key = md5_hash(
            json.dumps(
                [
                    [str(path.relative_to(Path(local_path))), path.read_text()]
                    for path in sorted(list(Path(local_path).rglob("*.py")) + list(Path(local_path).rglob("*.csv")))
                ]
            )
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
        )

        # 检查缓存是否存在
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            # 加载缓存结果
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            # 恢复文件状态
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path)
        else:
            # 运行并保存结果到缓存
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        在给定环境和本地路径内执行指定的入口点。
        这是实际的执行逻辑，由子类实现。

        返回
        -------
        tuple[str, int]
            一个包含标准输出和退出代码的元组。
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        将代码转储到本地路径并运行，然后获取 pickle 文件的结果。
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results


# ----- 本地环境 -----


class LocalConf(EnvConf):
    """本地环境配置"""
    bin_path: str = ""
    """类似 <path1>:<path2>:<path3> 的路径，会前置到 PATH 环境变量中。"""

    retry_count: int = 0  # 本地环境通常不需要重试
    live_output: bool = True  # 是否实时输出日志


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    本地环境，用于直接在宿主机上运行命令。
    有时对于测试来说更方便。
    """

    def prepare(self) -> None: ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:

        # 处理卷链接，通过符号链接模拟 Docker 的卷挂载
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = T("scenarios.data_science.share:scen.cache_path").r()
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp

        assert local_path is not None, "本地环境需要指定 local_path"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            """一个上下文管理器，用于创建和清理符号链接。"""
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # 设置环境变量
            if env is None:
                env = {}
            path = [*self.conf.bin_path.split(":"), "/bin/", "/usr/bin/", *env.get("PATH", "").split(":")]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]本地环境日志开始[/bold green]", style="dark_orange"))
            # 打印运行信息
            table = Table(title="运行信息", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("入口", entry)
            table.add_row("本地路径", local_path or "")
            table.add_row("环境变量", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("卷", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            # 使用 subprocess.Popen 执行命令
            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            if process.stdout is None or process.stderr is None:
                raise RuntimeError("子进程未能正确创建 stdout/stderr 管道")

            if self.conf.live_output:
                # 使用 select.poll 实现实时输出
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # 捕获最后剩余的输出
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # 一次性读取所有输出
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]本地环境日志结束[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    """Conda 环境配置"""
    conda_env_name: str
    default_entry: str = "python main.py"

    @model_validator(mode="after")
    def change_bin_path(self) -> "CondaConf":
        """
        在模型验证后，自动获取 Conda 环境的 PATH 并设置到 bin_path 中。
        """
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""
        return self


class MLECondaConf(CondaConf):
    """机器学习基准测试的 Conda 配置"""
    enable_cache: bool = False  # 与 Docker 设置保持一致


# ----- Docker 环境 -----
class DockerConf(EnvConf):
    """Docker 环境配置"""
    build_from_dockerfile: bool = False  # 是否从 Dockerfile 构建镜像
    dockerfile_folder_path: Optional[Path] = None  # Dockerfile 所在文件夹的路径
    image: str  # 要使用的镜像名称
    mount_path: str  # 代码挂载到容器内的路径
    default_entry: str  # 默认入口命令

    extra_volumes: dict = {}
    """
    额外的卷挂载。接受一个字典，格式可以是：
    {<宿主机路径>: <容器路径>} 或
    {<宿主机路径>: {"bind": <容器路径>, "mode": <模式, ro/rw>}}
    """
    extra_volume_mode: str = "ro"  # 默认挂载为只读
    network: str | None = "bridge"  # Docker 网络模式
    shm_size: str | None = None  # 共享内存大小
    enable_gpu: bool = True  # 是否启用 GPU
    mem_limit: str | None = "48g"  # 内存限制
    cpu_count: int | None = None  # CPU 核心数限制

    running_timeout_period: int | None = 3600  # 1小时超时

    enable_cache: bool = True  # 启用缓存机制

    retry_count: int = 5
    retry_wait_seconds: int = 10


# 以下是针对特定场景预设的配置类

class QlibCondaConf(CondaConf):
    conda_env_name: str = "rdagent4qlib"
    enable_cache: bool = False
    default_entry: str = "qrun conf.yaml"

class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    def prepare(self) -> None:
        """如果 conda 环境不存在，则创建并准备环境。"""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]Conda 环境 '{self.conf.conda_env_name}' 未找到，正在创建...[/yellow]")
                subprocess.check_call(f"conda create -y -n {self.conf.conda_env_name} python=3.10", shell=True)
                subprocess.check_call(f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython", shell=True)
                subprocess.check_call(f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@3e72593b8c985f01979bebcf646658002ac43b00", shell=True)
                subprocess.check_call(f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost scipy==1.11.4 tables torch", shell=True)
        except Exception as e:
            print(f"[red]准备 conda 环境失败: {e}[/red]")


class QlibDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="QLIB_DOCKER_", env_parse_none_str="None")
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"
    image: str = "local_qlib:latest"
    mount_path: str = "/workspace/qlib_workspace/"
    default_entry: str = "qrun conf.yaml"
    extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): {"bind": "/root/.qlib/", "mode": "rw"}}
    shm_size: str | None = "16g"
    enable_gpu: bool = True
    enable_cache: bool = False


class KGDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="KG_DOCKER_")
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "kaggle_docker"
    image: str = "local_kg:latest"
    mount_path: str = "/workspace/kg_workspace/"
    default_entry: str = "python train.py"
    running_timeout_period: int | None = 600
    mem_limit: str | None = "48g"


class DSDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="DS_DOCKER_")
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "DS_docker"
    image: str = "local_ds:latest"
    mount_path: str = "/kaggle/workspace"
    default_entry: str = "python main.py"
    running_timeout_period: int | None = 600
    mem_limit: str | None = "48g"


class MLEBDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="MLEB_DOCKER_")
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "mle_bench_docker"
    image: str = "local_mle:latest"
    mount_path: str = "/workspace/data_folder/"
    default_entry: str = "mlebench prepare --all"
    mem_limit: str | None = "48g"
    enable_cache: bool = False


class DockerEnv(Env[DockerConf]):
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        如果镜像不存在，则下载或构建它。
        """
        client = docker.from_env()
        if (
            self.conf.build_from_dockerfile
            and self.conf.dockerfile_folder_path is not None
            and self.conf.dockerfile_folder_path.exists()
        ):
            logger.info(f"从 Dockerfile 构建镜像: {self.conf.dockerfile_folder_path}")
            resp_stream = client.api.build(
                path=str(self.conf.dockerfile_folder_path), tag=self.conf.image, network_mode=self.conf.network
            )
            # 显示构建进度
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as p:
                task = p.add_task("[cyan]正在构建镜像...")
                for part in resp_stream:
                    lines = part.decode("utf-8").split("\r\n")
                    for line in lines:
                        if line.strip():
                            status_dict = json.loads(line)
                            if "error" in status_dict:
                                p.update(task, description=f"[red]错误: {status_dict['error']}")
                                raise docker.errors.BuildError(status_dict["error"], "")
                            if "stream" in status_dict:
                                p.update(task, description=status_dict["stream"])
            logger.info(f"完成从 Dockerfile 构建镜像: {self.conf.dockerfile_folder_path}")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            # 如果本地不存在镜像，则从仓库拉取
            image_pull = client.api.pull(self.conf.image, stream=True, decode=True)
            # 显示拉取进度
            with Progress(TextColumn("{task.description}"), TextColumn("{task.fields[progress]}")) as sp:
                main_task = sp.add_task("[cyan]正在拉取镜像...", progress="")
                status_task = sp.add_task("[bright_magenta]图层状态", progress="")
                for line in image_pull:
                    # ... (此处省略了详细的进度条更新逻辑)
                    pass
        except docker.errors.APIError as e:
            raise RuntimeError(f"拉取镜像时出错: {e}")

    def _gpu_kwargs(self, client: docker.DockerClient) -> dict:  # type: ignore[no-any-unimported]
        """
        根据 GPU 的可用性获取 GPU 相关的参数。
        """
        if not self.conf.enable_gpu:
            return {}
        gpu_kwargs = {
            "device_requests": (
                [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if self.conf.enable_gpu else None
            ),
        }

        def get_image(image_name: str) -> None:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                pull_image_with_progress(image_name)

        @wait_retry(5, 10)
        def _f() -> dict:
            """
            运行一个临时容器执行 nvidia-smi 来检查 GPU 是否真的可用。
            """
            container = None
            try:
                get_image(self.conf.image)
                container = client.containers.run(self.conf.image, "nvidia-smi", detach=True, **gpu_kwargs)
                container.wait()
                logger.info("GPU 设备可用。")
            except docker.errors.APIError:
                # 如果 API 调用失败（例如，主机没有 nvidia-docker runtime），则认为 GPU 不可用
                return {}
            finally:
                cleanup_container(container, context="GPU test")
            return gpu_kwargs

        return _f()

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        if env is None:
            env = {}
        # 设置一些通用的环境变量
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        client = docker.from_env()

        # 准备卷挂载
        volumes = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumes[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}

        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = {"bind": T("scenarios.data_science.share:scen.cache_path").r(), "mode": "rw"}
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}

        volumes = normalize_volumes(cast(dict[str, str | dict[str, str]], volumes), self.conf.mount_path)

        log_output = ""
        container: docker.models.containers.Container | None = None  # type: ignore[no-any-unimported]

        try:
            # 运行容器
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumes,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,
                cpu_count=self.conf.cpu_count,
                **self._gpu_kwargs(client),
            )
            assert container is not None
            # 打印运行信息
            print(Rule("[bold green]Docker 日志开始[/bold green]", style="dark_orange"))
            table = Table(title="运行信息", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("镜像", self.conf.image)
            table.add_row("容器 ID", container.id)
            table.add_row("容器名称", container.name)
            table.add_row("入口", entry)
            table.add_row("环境变量", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("卷", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            # 流式传输日志
            logs = container.logs(stream=True)
            for log in logs:
                decoded_log = log.strip().decode()
                Console().print(decoded_log, markup=False)
                log_output += decoded_log + "\n"

            # 等待容器结束并获取退出码
            exit_status = container.wait()["StatusCode"]
            print(Rule("[bold green]Docker 日志结束[/bold green]", style="dark_orange"))
            return log_output, exit_status
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"运行容器时出错: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker 镜像未找到。")
        except docker.errors.APIError as e:
            raise RuntimeError(f"运行容器时出错: {e}")
        finally:
            # 清理容器
            cleanup_container(container)


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker 环境"""

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        super().__init__(conf)

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        下载镜像和 Qlib 数据（如果不存在）。
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            logger.info("正在下载 Qlib 数据！")
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.check_output(entry=cmd)
        else:
            logger.info("数据已存在，跳过下载。")


class KGDockerEnv(DockerEnv):
    """Kaggle 竞赛 Docker 环境"""

    def __init__(self, competition: str | None = None, conf: DockerConf = KGDockerConf()):
        super().__init__(conf)


class MLEBDockerEnv(DockerEnv):
    """MLEBench Docker 环境"""

    def __init__(self, conf: DockerConf = MLEBDockerConf()):
        super().__init__(conf)
