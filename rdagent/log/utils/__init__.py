import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict, cast


class LogColors:
    """
    用于控制台输出的 ANSI 颜色代码。
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls: type["LogColors"]) -> list:
        """获取所有定义的颜色代码。"""
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        根据输入的颜色和样式渲染文本。
        不建议输入已经渲染过的文本。
        """
        # 这个方法被调用得太频繁了，这样不好。
        colors = self.get_all_colors()
        # 这里或许应该区分颜色和字体样式。
        if color and color in colors:
            error_message = f"颜色应该在: {colors} 中，但现在是: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"样式应该在: {colors} 中，但现在是: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"

    @staticmethod
    def remove_ansi_codes(s: str) -> str:
        """
        用于移除字符串中的 ANSI 控制字符（例如彩色文本）。
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", s)


class CallerInfo(TypedDict):
    """调用者信息类型字典。"""
    function: str
    line: int
    name: Optional[str]


def get_caller_info(level: int = 2) -> CallerInfo:
    """获取调用者的信息。"""
    # 获取当前的堆栈信息
    stack = inspect.stack()
    # 第二个元素通常是调用者的信息
    caller_info = stack[level]
    frame = caller_info[0]
    info: CallerInfo = {
        "line": caller_info.lineno,
        "name": frame.f_globals["__name__"],  # 从帧的全局变量中获取模块名
        "function": frame.f_code.co_name,  # 获取调用者的函数名
    }
    return info


def is_valid_session(log_path: Path) -> bool:
    """检查日志路径是否为有效的会话。"""
    return log_path.is_dir() and log_path.joinpath("__session__").exists()


def extract_loopid_func_name(tag: str) -> tuple[str, str] | tuple[None, None]:
    """从消息的标签中提取循环 ID 和函数名。"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return cast(tuple[str, str], match.groups()) if match else (None, None)


def extract_evoid(tag: str) -> str | None:
    """从消息的标签中提取进化 ID。"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return cast(str, match.group(1)) if match else None


def extract_json(log_content: str) -> dict | None:
    """从日志内容中提取 JSON 对象。"""
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return cast(dict, json.loads(match.group(0)))
    return None


def gen_datetime(dt: datetime | None = None) -> datetime:
    """
    生成一个 UTC 时区的 datetime 对象。
    - 如果 `dt` 为 None，将返回当前的 UTC 时间。
    - 如果提供了 `dt`，将把它转换为 UTC 时区。
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc)


def dict_get_with_warning(d: dict, key: str, default: Any = None) -> Any:
    """
    动机:
    - 在处理来自 LLM 的响应时，我们可能会使用 dict.get 来获取值。
    - 该函数防止 **静默地** 回退到默认值。
    - 相反，它会记录一条警告消息。
    """
    from rdagent.log import rdagent_logger as logger

    if key not in d:
        logger.warning(f"在 {d} 中未找到键 {key}")
        return default
    return d[key]
