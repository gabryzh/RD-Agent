from __future__ import annotations

from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Message:
    """存储的信息单元"""

    tag: str  # 命名空间，如 a.b.c
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  # 日志级别
    timestamp: datetime  # 消息生成的时间
    caller: Optional[
        str
    ]  # 日志调用者，如 `rdagent.oai.llm_utils:_create_chat_completion_inner_function:55`(文件:函数:行号)
    pid_trace: Optional[str]  # 进程ID跟踪；A-B-C 表示 A 创建 B，B 创建 C
    content: object  # 内容


class Storage:
    """
    基础存储，支持保存对象；

    # 用法:

    存储主要有两种用户：
    - 日志记录端：您可以选择以下任一方法使用该对象
        - 我们可以直接使用原生日志存储
        - 我们可以将其与其他日志工具一起使用；例如，作为日志记录器的处理程序
    - 查看端：
        - 主要用于 `logging.base.View` 的子类
        - 它应提供两种提供内容的方式
            - 离线内容提供。
            - 在线内容提供。
    """

    @abstractmethod
    def log(
        self,
        obj: object,
        tag: str = "",
        timestamp: datetime | None = None,
    ) -> str | Path:
        """

        参数
        ----------
        obj : object
            要记录的对象。
        name : str
            对象的名称。例如 "a.b.c"
            我们可能会将许多对象记录到同一个名称下

        返回
        -------
        str | Path
            对象的存储标识符。
        """
        ...

    @abstractmethod
    def iter_msg(self) -> Generator[Message, None, None]:
        """
        迭代存储中的消息。
        """
        ...

    @abstractmethod
    def truncate(self, time: datetime) -> None:
        """
        删除指定时间之后的所有日志条目。
        """
        ...

    def __str__(self) -> str:
        return self.__class__.__name__


class View:
    """
    动机:

    显示存储中的内容
    """

    # TODO: 请修复我
    @abstractmethod
    def display(self, s: Storage, watch: bool = False) -> None:
        """

        参数
        ----------
        s : Storage
            存储对象
        watch : bool
            我们是否应该监视新内容并显示它们
        """
        ...
