from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class LogSettings(ExtendedBaseSettings):
    """日志设置类"""
    model_config = SettingsConfigDict(env_prefix="LOG_", protected_namespaces=())

    # 日志跟踪文件的路径，默认使用当前时间和UTC时区生成
    trace_path: str = str(Path.cwd() / "log" / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f"))

    # 控制台输出格式，如果为 None，则使用默认格式
    format_console: str | None = None

    # UI 服务器端口，如果为 None，则不启动UI服务器
    ui_server_port: int | None = None

    # 存储配置，键是存储类的完全限定名称，值是初始化参数
    storages: dict[str, list[int | str]] = {}

    def model_post_init(self, _context: Any, /) -> None:
        """
        Pydantic 模型初始化后的钩子函数。
        如果设置了 UI 服务器端口，则自动添加 WebStorage 的配置。
        """
        if self.ui_server_port is not None:
            self.storages["rdagent.log.ui.storage.WebStorage"] = [self.ui_server_port, self.trace_path]


# 创建一个全局的日志设置实例
LOG_SETTINGS = LogSettings()
