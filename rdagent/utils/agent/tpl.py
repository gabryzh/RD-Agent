"""
这里是一些构建 agent 的基础工具。

此模块设计了模板（Template）和 AgentOutput 的基础结构。
"""

import inspect
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FunctionLoader, StrictUndefined

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger

# 获取当前文件所在的目录
DIRNAME = Path(__file__).absolute().resolve().parent
# 获取项目根目录 (rdagent)
PROJ_PATH = DIRNAME.parent.parent


def get_caller_dir(upshift: int = 0) -> Path:
    """
    获取调用此函数的代码所在的目录。
    这通过检查调用栈来实现。

    参数:
    ----------
    upshift : int
        向上追溯调用栈的层数。0 表示直接调用者，1 表示调用者的调用者，以此类推。

    返回:
    -------
    Path
        调用者文件所在的目录路径。
    """
    stack = inspect.stack()
    # 1 + upshift: 1 是为了跳过 get_caller_dir 自身的栈帧
    caller_frame = stack[1 + upshift]
    caller_module = inspect.getmodule(caller_frame[0])
    if caller_module and caller_module.__file__:
        caller_dir = Path(caller_module.__file__).parent
    else:
        # 如果无法确定调用者模块（例如在交互式环境中），则返回当前文件目录
        caller_dir = DIRNAME
    return caller_dir


def load_content(uri: str, caller_dir: Path | None = None, ftype: str = "yaml") -> Any:
    """
    根据 URI 加载文件内容。
    支持从不同位置加载，并解析 YAML 路径。

    参数:
    ----------
    uri : str
        要加载内容的统一资源标识符。格式为 "path.part:yaml.trace"。
    caller_dir : Path | None
        调用者的目录，用于解析相对路径 URI。
    ftype : str
        文件类型，默认为 "yaml"。

    返回:
    -------
    Any
        加载并解析后的内容。
    """
    if caller_dir is None:
        # 如果未提供，则自动获取调用者的上层目录
        caller_dir = get_caller_dir(upshift=1)

    # 解析 URI，分离路径部分和 YAML 内部路径
    path_part, *yaml_trace = uri.split(":")
    assert len(yaml_trace) <= 1, f"无效的 URI '{uri}'，只允许一个 YAML 路径跟踪。"
    yaml_trace = [key for yt in yaml_trace for key in yt.split(".")]

    # 根据 URI 的格式确定文件搜索路径列表
    if path_part.startswith("."):
        # 相对路径 URI (以 '.' 开头)
        # 1. 在调用者目录下查找
        file_path_l = [caller_dir / f"{path_part[1:].replace('.', '/')}.{ftype}"]
        # 2. (如果设置了 app_tpl) 在应用模板目录下查找，具有更高优先级
        if RD_AGENT_SETTINGS.app_tpl is not None:
            file_path_l.insert(0, PROJ_PATH / RD_AGENT_SETTINGS.app_tpl / file_path_l[0].relative_to(PROJ_PATH))
    else:
        # 绝对路径 URI (相对于项目)
        file_path_l = [
            # 3. 相对于当前工作目录
            Path(path_part.replace(".", "/")).with_suffix(f".{ftype}"),
            # 4. 相对于项目根目录
            (PROJ_PATH / path_part.replace(".", "/")).with_suffix(f".{ftype}"),
        ]
        # 添加应用模板目录的搜索路径，使其具有更高优先级，以实现模板覆盖
        if RD_AGENT_SETTINGS.app_tpl is not None:
            # 2. 在应用模板目录下查找
            file_path_l.insert(
                0, (PROJ_PATH / RD_AGENT_SETTINGS.app_tpl / path_part.replace(".", "/")).with_suffix(f".{ftype}")
            )
            # 1. 在项目根目录的上一级查找 (用于模板间的递归扩展)
            file_path_l.insert(0, (PROJ_PATH.parent / path_part.replace(".", "/")).with_suffix(f".{ftype}"))

    # 按优先级顺序尝试加载文件
    for file_path in file_path_l:
        try:
            if ftype == "yaml":
                # 为了跨平台兼容性，使用 UTF-8 编码解析 YAML
                with file_path.open(encoding="utf-8") as file:
                    yaml_content = yaml.safe_load(file)
                # 遍历 YAML 内容以获取所需模板
                for key in yaml_trace:
                    yaml_content = yaml_content[key]
                return yaml_content

            # 如果是其他文件类型，直接读取文本内容
            return file_path.read_text()
        except FileNotFoundError:
            continue  # 文件不存在，继续尝试下一个路径
        except KeyError:
            continue  # 文件存在，但 YAML 键不存在，继续尝试下一个路径
    else:
        # 如果所有路径都尝试失败，则抛出异常
        raise FileNotFoundError(f"在以下路径中找不到 '{uri}': {file_path_l}")


class RDAT:
    """
    RD-Agent 的模板类 (RD-Agent's Template)。
    使用最简单的方式来 (C)reate 创建模板和 (r)ender 渲染它。
    """

    def __init__(self, uri: str, ftype: str = "yaml"):
        """
        URI 用法示例:
            case 1) "a.b.c:x.y.z"
                它会加载 <当前目录或RD-Agent包目录>/a/b/c.yaml，并返回 yaml[x][y][z] 的内容。
                例如，要加载 "rdagent/scenarios/kaggle/experiment/prompts.yaml"，
                `a.b.c` 应该是 "scenarios.kaggle.experiment.prompts"，"rdagent" 应被排除。

            case 2) ".c:x.y.z"
                它会加载调用者目录下的 c.yaml，并返回 yaml[x][y][z] 的内容。

            case 3) "a.b.c" 且 ftype="txt"
                它会加载 a/b/c.txt 并直接返回其内容。

        内容加载优先级 (从高到低):
        1. 应用模板目录 (app_tpl)
        2. 调用者目录 (用于相对路径) 或 当前工作目录 (用于绝对路径)
        3. RD-Agent 项目根目录 (默认模板)
        """
        self.uri = uri
        caller_dir = get_caller_dir(1)
        # 如果是相对路径，尝试将其转换为相对于项目根目录的“绝对”URI，便于调试和日志记录
        if uri.startswith("."):
            try:
                self.uri = f"{str(caller_dir.resolve().relative_to(PROJ_PATH)).replace('/', '.')}{uri}"
            except ValueError:
                # 如果调用者目录不在项目路径下，则忽略转换
                pass
        self.template = load_content(uri, caller_dir=caller_dir, ftype=ftype)

    def r(self, **context: Any) -> str:
        """
        使用给定的上下文渲染模板。
        """
        # loader=FunctionLoader(load_content) 支持在模板中使用 `include` 语法，
        # 例如 `{% include "scenarios.data_science.share:component_spec.DataLoadSpec" %}`
        env = Environment(undefined=StrictUndefined, loader=FunctionLoader(load_content))
        rendered = env.from_string(self.template).render(**context).strip("\n")

        # 压缩多余的换行符
        while "\n\n\n" in rendered:
            rendered = rendered.replace("\n\n\n", "\n\n")

        # 记录调试信息
        logger.log_object(
            obj={
                "uri": self.uri,
                "template": self.template,
                "context": context,
                "rendered": rendered,
            },
            tag="debug_tpl",
        )
        return rendered


T = RDAT  # 创建一个简写别名
