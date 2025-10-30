# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入标准库
import io
import os
import platform
import re
import shutil
import typing
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# 导入项目内部模块
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback

# 用于类型检查时导入，避免循环依赖
if TYPE_CHECKING:
    from rdagent.utils.env import EnvResult

if typing.TYPE_CHECKING:
    from rdagent.core.proposal import Hypothesis
    from rdagent.utils.env import Env

"""
该文件包含了在RD-Agent中组织任务相关的所有类。
"""


class AbsTask(ABC):
    """任务的抽象基类"""
    def __init__(self, name: str, version: int = 1) -> None:
        """
        初始化任务。
        :param name: 任务名称。
        :param version: 任务版本，默认为1。因为qlib任务和kaggle任务的执行方式不同，需要区分。
                        TODO: 未来可能会统一它们。
        """
        self.version = version
        self.name = name

    @abstractmethod
    def get_task_information(self) -> str:
        """获取任务信息字符串，用于构建唯一键。"""


class UserInstructions(list[str]):
    """用户指令类，继承自list"""
    def __str__(self) -> str:
        """将用户指令格式化为字符串"""
        if self:
            return ("\n用户指令 (最高优先级!):\n" + "\n".join(f"- {ui}" for ui in self)) if self else ""
        return ""


class Task(AbsTask):
    """具体的任务类"""
    def __init__(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        user_instructions: UserInstructions | None = None,
    ) -> None:
        """
        初始化任务。
        :param name: 任务名称。
        :param version: 任务版本。
        :param description: 任务描述。
        :param user_instructions: 用户指令。
        """
        super().__init__(name, version)
        self.description = description
        self.user_instructions = user_instructions

    def get_task_information(self) -> str:
        """获取格式化的任务信息"""
        return f"任务名称: {self.name}\n描述: {self.description}{self.user_instructions!s}"

    def __repr__(self) -> str:
        """返回任务对象的字符串表示"""
        return f"<{self.__class__.__name__} {self.name}>"


# 定义泛型类型变量
ASpecificTask = TypeVar("ASpecificTask", bound=Task)
ASpecificFeedback = TypeVar("ASpecificFeedback", bound=Feedback)


@dataclass
class RunningInfo:
    """运行信息的数据类"""
    result: object = None  # 实验结果，在不同场景下可以是不同类型
    running_time: float | None = None  # 运行时间


class Workspace(ABC, Generic[ASpecificTask, ASpecificFeedback]):
    """
    工作区是存储任务实现的地方。它随着开发人员实现任务而演进。
    要获取工作区的快照，请确保调用 `copy` 方法。
    """

    def __init__(self, target_task: ASpecificTask | None = None) -> None:
        self.target_task: ASpecificTask | None = target_task
        self.feedback: ASpecificFeedback | None = None
        self.running_info: RunningInfo = RunningInfo()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> object | None:
        """执行工作区中的任务"""
        error_message = "execute 方法未实现。"
        raise NotImplementedError(error_message)

    @abstractmethod
    def copy(self) -> Workspace:
        """复制工作区"""
        error_message = "copy 方法未实现。"
        raise NotImplementedError(error_message)

    @property
    @abstractmethod
    def all_codes(self) -> str:
        """将工作区中的所有代码文件作为单个字符串获取。"""

    # 当工作区是可变的时，提供创建检查点和恢复的支持
    @abstractmethod
    def create_ws_ckp(self) -> None:
        """
        创建工作区的内存检查点，以便稍后可以恢复。
        """

    @abstractmethod
    def recover_ws_ckp(self) -> None:
        """
        从 :py:meth:`create_ws_ckp` 创建的检查点恢复工作区。
        """


ASpecificWS = TypeVar("ASpecificWS", bound=Workspace)


class WsLoader(ABC, Generic[ASpecificTask, ASpecificWS]):
    """工作区加载器的抽象基类"""
    @abstractmethod
    def load(self, task: ASpecificTask) -> ASpecificWS:
        """加载工作区"""
        error_message = "load 方法未实现。"
        raise NotImplementedError(error_message)


class FBWorkspace(Workspace):
    """
    基于文件的任务工作区。

    已实现的任务将是一个包含相关元素的文件夹：
    - 数据
    - 代码工作区
    - 输出
        - 执行后，它将生成最终输出作为文件。

    运行FBWorkspace管道的典型方法如下：
    （我们没有将其添加为方法，因为我们可能需要根据需求向 `prepare` 或 `execute` 传递参数。）

    .. code-block:: python

        def run_pipeline(self, **files: str):
            self.prepare()
            self.inject_files(**files)
            self.execute()

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_dict: dict[str, Any] = {}  # 注入到文件夹中的代码，存储在此变量中以重现先前结果
        self.workspace_path: Path = RD_AGENT_SETTINGS.workspace_path / uuid.uuid4().hex  # 工作区路径
        self.ws_ckp: bytes | None = None  # 由 ``create_ws_ckp`` 创建的内存检查点数据
        self.change_summary: str | None = None  # 与先前版本工作区相比的变更摘要

    @staticmethod
    def _format_code_dict(code_dict: dict[str, str]) -> str:
        """
        将代码字典格式化为字符串的辅助函数。
        """
        code_string = ""
        for file_name in sorted(code_dict.keys()):
            code_string += f"\n文件路径: {file_name}\n```\n{code_dict[file_name]}\n```"
        return code_string

    @property
    def all_codes(self) -> str:
        """
        将工作区中的所有代码文件（不包括测试文件）作为单个字符串获取。
        """
        filtered_dict = {k: v for k, v in self.file_dict.items() if k.endswith(".py") and "test" not in k}
        return self._format_code_dict(filtered_dict)

    def get_codes(self, pattern: str) -> str:
        """
        获取与特定模式匹配的代码文件（不包括测试文件）作为单个字符串。
        """
        filtered_dict = {
            k: v for k, v in self.file_dict.items() if re.search(pattern, k) and k.endswith(".py") and "test" not in k
        }
        return self._format_code_dict(filtered_dict)

    def prepare(self) -> None:
        """
        准备工作区（不包括注入的代码）。
        - 数据
        - 文档
            `*args, **kwargs` 的典型用法：
                不同方法共享相同的数据。数据通过参数传递。
        """
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def link_all_files_in_folder_to_workspace(data_path: Path, workspace_path: Path) -> None:
        """将文件夹中的所有文件链接到工作区"""
        data_path = Path(data_path).absolute()  # 使用绝对路径，以防更改当前工作目录时失效
        workspace_path = Path(workspace_path)
        for data_file_path in data_path.iterdir():
            workspace_data_file_path = workspace_path / data_file_path.name
            if workspace_data_file_path.exists():
                workspace_data_file_path.unlink()
            if platform.system() in ("Linux", "Darwin"):
                workspace_data_file_path.symlink_to(data_file_path)
            if platform.system() == "Windows":
                os.link(data_file_path, workspace_data_file_path)

    DEL_KEY = "__DEL__"  # 用于表示删除文件的键

    def inject_files(self, **files: str) -> None:
        """
        将代码注入文件夹。
        {
            <文件名1>: <代码>,  // 表示将<code>写入<文件名>（创建新文件或替换现有文件）
            <文件名2>: "__DEL__"  // 表示删除文件名2。当我们想用新文件替换旧文件时，通常使用此方法
        }
        """
        self.prepare()
        for k, v in files.items():
            target_file_path = self.workspace_path / k
            if v == self.DEL_KEY:
                if target_file_path.exists():
                    target_file_path.unlink()
                self.file_dict.pop(k, None)
            else:
                self.file_dict[k] = v
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                target_file_path.write_text(v)

    def get_files(self) -> list[Path]:
        """
        获取环境描述。
        为保持通用性，我们只返回文件名列表。
        如何总结环境是开发人员的责任。
        """
        return list(self.workspace_path.iterdir())

    def inject_code_from_folder(self, folder_path: Path) -> None:
        """
        从文件夹加载工作区。
        """
        for file_path in folder_path.rglob("*"):
            if file_path.suffix in (".py", ".yaml", ".md"):
                relative_path = file_path.relative_to(folder_path)
                self.inject_files(**{str(relative_path): file_path.read_text()})

    def inject_code_from_file_dict(self, workspace: FBWorkspace) -> None:
        """
        从file_dict加载工作区。
        """
        for name, code in workspace.file_dict.items():
            self.inject_files(**{name: code})

    def copy(self) -> FBWorkspace:
        """
        从原始工作区复制一份。
        """
        return deepcopy(self)

    def clear(self) -> None:
        """
        清空工作区。
        """
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.file_dict = {}

    def before_execute(self) -> None:
        """
        在执行代码之前，我们需要准备工作区并注入代码。
        """
        self.prepare()
        self.inject_files(**self.file_dict)

    def execute(self, env: Env, entry: str) -> str:
        """
        在每次执行前，请确保准备和注入代码。
        """
        result = self.run(env, entry)
        return result.get_truncated_stdout()  # 注意：截断是为了与旧代码保持一致

    def run(self, env: Env, entry: str) -> EnvResult:
        """
        在环境中执行代码并返回一个EnvResult对象（stdout, exit_code, running_time）。
        在每次执行前，请确保准备和注入代码。
        """
        self.prepare()
        self.inject_files(**self.file_dict)
        return env.run(entry, str(self.workspace_path), env={"PYTHONPATH": "./"})

    def create_ws_ckp(self) -> None:
        """
        将 ``workspace_path`` 的内容压缩并将其存档保留在 ``self.ws_ckp`` 中，
        以便稍后通过 :py:meth:`recover_ws_ckp` 进行恢复。
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.workspace_path.rglob("*"):
                # 只包括小于特定大小的常规文件，以使检查点保持轻量。
                # 较大的文件（例如数据集）应单独重新创建或挂载。
                if file_path.is_symlink():
                    # 在存档中保留符号链接
                    zi = zipfile.ZipInfo(str(file_path.relative_to(self.workspace_path)))
                    zi.create_system = 3  # 表示Unix
                    zi.external_attr = 0o120777 << 16  # 符号链接文件类型 + 0777权限
                    zf.writestr(zi, str(file_path.readlink()))
                elif file_path.is_file():
                    size_limit = RD_AGENT_SETTINGS.workspace_ckp_size_limit
                    if (
                        RD_AGENT_SETTINGS.workspace_ckp_white_list_names is not None
                        and file_path.name in RD_AGENT_SETTINGS.workspace_ckp_white_list_names
                    ) or (size_limit <= 0 or file_path.stat().st_size <= size_limit):
                        zf.write(file_path, file_path.relative_to(self.workspace_path))
        self.ws_ckp = buf.getvalue()

    def recover_ws_ckp(self) -> None:
        """
        从由 :py:meth:`create_ws_ckp` 创建的内存检查点恢复工作区目录。
        """
        if self.ws_ckp is None:
            msg = "工作区检查点不存在。请先调用 `create_ws_ckp`。"
            raise RuntimeError(msg)
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(self.ws_ckp)
        with zipfile.ZipFile(buf, "r") as zf:
            for info in zf.infolist():
                dest_path = self.workspace_path / info.filename
                mode = (info.external_attr >> 16) & 0o170000
                symlink_mode = 0o120000
                if mode == symlink_mode:  # 符号链接
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    link_target = zf.read(info).decode()
                    dest_path.symlink_to(link_target)
                elif info.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with dest_path.open("wb") as f:
                        f.write(zf.read(info))
        # 注意：这对减小对象大小非常重要
        self.ws_ckp = None

    def __str__(self) -> str:
        """返回工作区对象的字符串表示"""
        return f"Workspace[{self.workspace_path=}" + (
            "]" if self.target_task is None else f",{self.target_task.name=}]"
        )


ASpecificWSForExperiment = TypeVar("ASpecificWSForExperiment", bound=Workspace)
ASpecificWSForSubTasks = TypeVar("ASpecificWSForSubTasks", bound=Workspace)


class ExperimentPlan(dict[str, Any]):
    """
    实验计划，是一个包含各阶段计划的字典。
    """


class Experiment(
    ABC,
    Generic[ASpecificTask, ASpecificWSForExperiment, ASpecificWSForSubTasks],
):
    """
    实验是任务的序列以及由开发人员生成后任务的实现。
    """

    def __init__(
        self,
        sub_tasks: Sequence[ASpecificTask],
        based_experiments: Sequence[ASpecificWSForExperiment] = [],
        hypothesis: Hypothesis | None = None,
    ) -> None:
        self.hypothesis: Hypothesis | None = hypothesis  # 实验可选择性地由假设生成
        self.sub_tasks: Sequence[ASpecificTask] = sub_tasks
        # None 表示
        # - 实现前的初始化占位符
        # - 开发人员主动跳过任务
        self.sub_workspace_list: list[ASpecificWSForSubTasks | None] = [None] * len(self.sub_tasks)
        # TODO:
        # 它将在历史记录中的运行器中使用
        # 如果我们实现了整个工作流，就不必使用它，然后将其删除。
        self.based_experiments: Sequence[ASpecificWSForExperiment] = based_experiments

        self.experiment_workspace: ASpecificWSForExperiment | None = None

        # 实验可能由不同的开发人员开发。
        # 上一个反馈用于向下一个开发人员传播信息。
        # 生命周期:
        # - 开发人员为下一个组件分配反馈；
        # - 工作流控制清除反馈。
        self.prop_dev_feedback: Feedback | None = None

        # 注意：假设
        # - 只有运行器会分配此变量
        # - 当我们进入下一个新循环时，我们将始终创建一个新实验，而不会复制以前的结果。
        self.running_info = RunningInfo()
        self.sub_results: dict[str, float] = {}  # TODO: 在Kaggle中，现在所有子结果都保存在self.result中，将来删除此项。

        # 支持并行多轨迹
        self.local_selection: tuple[int, ...] | None = None
        self.plan: ExperimentPlan | None = None  # 存储此实验的规划信息，应在exp_gen.gen内部生成
        self.user_instructions: UserInstructions | None = None  # 存储此实验的用户指令

    def set_user_instructions(self, user_instructions: UserInstructions | None) -> None:
        """设置用户指令并将其传播到子任务和工作区"""
        if user_instructions is None:
            return
        if not isinstance(user_instructions, UserInstructions) and isinstance(user_instructions, list):
            user_instructions = UserInstructions(user_instructions)
        self.user_instructions = user_instructions
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.target_task.user_instructions = user_instructions  # type: ignore[union-attr]
        for task in self.sub_tasks:
            task.user_instructions = user_instructions
        if self.experiment_workspace is not None and self.experiment_workspace.target_task is not None:
            self.experiment_workspace.target_task.user_instructions = user_instructions

    @property
    def result(self) -> object:
        """获取实验结果"""
        return self.running_info.result

    @result.setter
    def result(self, value: object) -> None:
        """设置实验结果"""
        self.running_info.result = value

    # 当工作区是可变的时，提供创建检查点和恢复的支持
    def create_ws_ckp(self) -> None:
        """为实验中的所有工作区创建检查点"""
        if self.experiment_workspace is not None:
            self.experiment_workspace.create_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.create_ws_ckp()

    def recover_ws_ckp(self) -> None:
        """从检查点恢复实验中的所有工作区"""
        if self.experiment_workspace is not None:
            self.experiment_workspace.recover_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                try:
                    ws.recover_ws_ckp()
                except RuntimeError:
                    # FBWorkspace在experiment_workspace和sub_workspace_list之间共享，
                    # 因此如果一个工作区被恢复两次，recover_ws_ckp会引发RuntimeError。
                    print("由于一个工作区被恢复两次，recover_ws_ckp失败。")


ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)
ASpecificPlan = TypeVar("ASpecificPlan", bound=ExperimentPlan)

TaskOrExperiment = TypeVar("TaskOrExperiment", Task, Experiment)


class Loader(ABC, Generic[TaskOrExperiment]):
    """加载器抽象基类，用于加载任务或实验"""
    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> TaskOrExperiment:
        """加载方法"""
        err_msg = "load 方法未实现。"
        raise NotImplementedError(err_msg)
