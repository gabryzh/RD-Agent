import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

from .base import Message, Storage
from .utils import gen_datetime

# 定义日志级别的类型别名
LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _remove_empty_dir(path: Path) -> None:
    """
    递归地删除空目录。
    此函数将在删除其子目录后，如果该目录为空，则删除该目录。
    """
    if path.is_dir():
        sub_dirs = [sub for sub in path.iterdir() if sub.is_dir()]
        for sub in sub_dirs:
            _remove_empty_dir(sub)

        if not any(path.iterdir()):
            path.rmdir()


class FileStorage(Storage):
    """
    信息被记录到文件系统中。

    TODO: 描述存储格式
    """

    def __init__(self, path: str | Path) -> None:
        """初始化文件存储，指定日志根目录。"""
        self.path = Path(path)

    def log(
        self,
        obj: object,
        tag: str = "",
        timestamp: datetime | None = None,
        save_type: Literal["json", "text", "pkl"] = "pkl",
        **kwargs: Any,
    ) -> str | Path:
        """
        记录一个对象到文件中。
        """
        # TODO: 我们可以在实现 PipeLog 后移除时间戳
        timestamp = gen_datetime(timestamp)

        # 根据标签创建目录结构
        cur_p = self.path / tag.replace(".", "/")
        cur_p.mkdir(parents=True, exist_ok=True)

        # 基于时间戳生成文件名
        path = cur_p / f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f')}.log"

        # 根据指定的保存类型写入文件
        if save_type == "json":
            path = path.with_suffix(".json")
            with path.open("w") as f:
                try:
                    json.dump(obj, f)
                except TypeError:
                    json.dump(json.loads(str(obj)), f)
            return path
        elif save_type == "pkl":
            path = path.with_suffix(".pkl")
            with path.open("wb") as f:
                pickle.dump(obj, f)
            return path
        elif save_type == "text":
            obj = str(obj)
            with path.open("w") as f:
                f.write(obj)
            return path

    # 定义用于解析标准日志格式的正则表达式
    log_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| "
        r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) *\| "
        r"(?P<caller>.+:.+:\d+) - "
    )

    def iter_msg(self, tag: str | None = None, pattern: str | None = None) -> Generator[Message, None, None]:
        """
        迭代存储中的消息。
        可以按标签或自定义模式进行筛选。
        """
        msg_l = []

        # 根据参数确定要搜索的文件模式
        if pattern:
            pkl_files = pattern
        elif tag:
            pkl_files = f"**/{tag.replace('.','/')}/**/*.pkl"
        else:
            pkl_files = "**/*.pkl"

        # 遍历所有匹配的 pkl 文件
        for file in self.path.glob(pkl_files):
            if file.name == "debug_llm.pkl":
                continue

            # 从文件路径中提取标签和进程ID
            pkl_log_tag = ".".join(file.relative_to(self.path).as_posix().replace("/", ".").split(".")[:-3])
            pid = file.parent.name

            # 读取 pkl 文件内容
            with file.open("rb") as f:
                content = pickle.load(f)

            # 从文件名中解析时间戳
            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)

            # 创建 Message 对象
            m = Message(tag=pkl_log_tag, level="INFO", timestamp=timestamp, caller="", pid_trace=pid, content=content)

            msg_l.append(m)

        # 按时间戳对消息进行排序
        msg_l.sort(key=lambda x: x.timestamp)
        # 逐个产生消息
        for m in msg_l:
            yield m

    def truncate(self, time: datetime) -> None:
        """
        删除指定时间之后的所有日志条目。
        """
        # 遍历所有 pkl 文件
        for file in self.path.glob("**/*.pkl"):
            timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
            # 如果文件的时间戳晚于指定时间，则删除文件
            if timestamp > time.replace(tzinfo=timezone.utc):
                file.unlink()

        # 删除可能变为空的目录
        _remove_empty_dir(self.path)

    def __str__(self) -> str:
        return f"FileStorage({self.path})"
