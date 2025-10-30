"""
context7 基于 context7 的修改版本。

您可以按照以下说明进行安装

    mkdir -p ~/tmp/
    cd ~/tmp/ && git clone https://github.com/Hoder-zyf/context7.git
    cd ~/tmp/context7
    npm install -g bun
    bun i && bun run build
    bun run dist/index.js --transport http --port 8124 # > bun.out 2>&1 &
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目特定设置。"""

    # context7 服务的 URL
    url: str = "http://localhost:8124/mcp"
    # 请求超时时间（秒）
    timeout: int = 120
    # 是否启用缓存
    enable_cache: bool = False
    # 在 .env 文件中设置 CONTEXT7_ENABLE_CACHE=true 以启用缓存

    model_config = SettingsConfigDict(
        # 环境变量前缀
        env_prefix="CONTEXT7_",
    )


# 创建一个全局的设置实例
SETTINGS = Settings()
