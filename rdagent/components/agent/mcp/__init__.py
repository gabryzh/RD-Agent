"""
这里是 MCP 服务器的列表。

MCP 服务器是一个独立的 RESTful API。因此，文件夹中仅包含以下内容：
- 设置。
  - 例如，mcp/<mcp_name>.py:class Settings(BaseSettings); 然后它被初始化为一个全局变量 SETTINGS。
  - 它只在 Python 类（即 Pydantic BaseSettings）中定义了设置的格式。
- 健康检查：
  - 例如，mcp/<mcp_name>.py:def health_check() -> bool;
"""
