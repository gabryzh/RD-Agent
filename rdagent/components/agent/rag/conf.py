"""
RAG 代理的设置。

TODO: 如何运行 RAG mcp 服务器
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目特定设置。"""

    # RAG MCP 服务器的 URL
    url: str = "http://localhost:8124/mcp"
    # 请求超时时间（秒）
    timeout: int = 120

    model_config = SettingsConfigDict(
        # 环境变量前缀
        env_prefix="RAG_",
    )


# 创建一个全局的设置实例
SETTINGS = Settings()
