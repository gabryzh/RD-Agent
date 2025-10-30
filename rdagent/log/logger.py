import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Generator

from loguru import logger

from .conf import LOG_SETTINGS

# 如果在配置中指定了控制台格式，则移除默认的 loguru 处理器并添加自定义格式的处理器
if LOG_SETTINGS.format_console is not None:
    logger.remove()
    logger.add(sys.stdout, format=LOG_SETTINGS.format_console)

from psutil import Process

from rdagent.core.utils import SingletonBaseClass, import_class

from .base import Storage
from .storage import FileStorage
from .utils import get_caller_info


class RDAgentLog(SingletonBaseClass):
    """
    RDAgent 的日志记录器。
    文件根据标签（tag）和进程ID（PID）进行组织。
    这是一个标签示例的结构：

    .. code-block::

        a
        - b
        - c
            - 123
              - common_logs.log
            - 1322
              - common_logs.log
            - 1233
              - <timestamp>.pkl
            - d
                - 1233-673 ...
                - 1233-4563 ...
                - 1233-365 ...

    """

    # 线程/协程本地的标签；在 Linux fork 的子进程中，它将被复制到子进程。
    _tag_ctx: ContextVar[str] = ContextVar("_tag_ctx", default="")

    @property
    def _tag(self) -> str:  # 获取当前标签
        return self._tag_ctx.get()

    @_tag.setter  # 设置当前标签
    def _tag(self, value: str) -> None:
        self._tag_ctx.set(value)

    def __init__(self) -> None:
        """初始化日志记录器，设置主存储和其他存储。"""
        self.storage = FileStorage(LOG_SETTINGS.trace_path)
        self.other_storages: list[Storage] = []
        for storage, args in LOG_SETTINGS.storages.items():
            storage_cls = import_class(storage)
            self.other_storages.append(storage_cls(*args))

        self.main_pid = os.getpid()

    @contextmanager
    def tag(self, tag: str) -> Generator[None, None, None]:
        """
        一个上下文管理器，用于在特定代码块中设置日志标签。
        """
        if tag.strip() == "":
            raise ValueError("标签不能为空。")
        # 生成一个新的完整标签
        current_tag = self._tag_ctx.get()
        new_tag = tag if current_tag == "" else f"{current_tag}.{tag}"
        # 设置新标签并保存令牌以便稍后恢复
        token = self._tag_ctx.set(new_tag)
        try:
            yield
        finally:
            # 恢复之前的标签（线程/协程安全）
            self._tag_ctx.reset(token)

    def set_storages_path(self, path: str | Path) -> None:
        """设置所有存储的路径。"""
        for storage in [self.storage] + self.other_storages:
            if hasattr(storage, "path"):
                storage.path = path

    def truncate_storages(self, time: datetime) -> None:
        """截断所有存储中指定时间之后的日志。"""
        for storage in [self.storage] + self.other_storages:
            storage.truncate(time=time)

    def get_pids(self) -> str:
        """
        返回从当前进程到主进程的 PID 链字符串，以 '-' 分隔。
        """
        pid = os.getpid()
        process = Process(pid)
        pid_chain = f"{pid}"
        while process.pid != self.main_pid:
            parent_pid = process.ppid()
            parent_process = Process(parent_pid)
            pid_chain = f"{parent_pid}-{pid_chain}"
            process = parent_process
        return pid_chain

    def log_object(self, obj: object, *, tag: str = "") -> None:
        """
        记录一个 Python 对象。
        """
        # 构建完整的标签
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        # 在所有存储中记录该对象
        for storage in [self.storage] + self.other_storages:
            storage.log(obj, tag=tag)

    def _log(self, level: str, msg: str, *, tag: str = "", raw: bool = False) -> None:
        """
        内部日志记录方法。
        """
        caller_info = get_caller_info(level=3)
        # 构建完整的标签
        tag = f"{self._tag}.{tag}.{self.get_pids()}".strip(".")

        # 如果是原始模式，则移除格式
        if raw:
            logger.remove()
            logger.add(sys.stderr, format=lambda r: "{message}")

        # 获取 loguru 对应级别的日志函数并调用
        log_func = getattr(logger.patch(lambda r: r.update(caller_info)), level)
        log_func(msg)

        # 如果是原始模式，则恢复默认格式
        if raw:
            logger.remove()
            logger.add(sys.stderr)

    def info(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        """记录 INFO 级别的日志。"""
        self._log("info", msg, tag=tag, raw=raw)

    def warning(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        """记录 WARNING 级别的日志。"""
        self._log("warning", msg, tag=tag, raw=raw)

    def error(self, msg: str, *, tag: str = "", raw: bool = False) -> None:
        """记录 ERROR 级别的日志。"""
        self._log("error", msg, tag=tag, raw=raw)
