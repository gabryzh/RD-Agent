# 导入必要的模块
from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.rag.conf import SETTINGS
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    一个专门用于 RAG（检索增强生成）的特定代理。
    """

    def __init__(self, system_prompt: str | None = None):
        """
        初始化 RAG 代理。

        参数:
            system_prompt (str | None): 可选的系统提示。如果未提供，将使用默认提示。
        """
        # 设置工具集，连接到 RAG 服务的 MCP 服务器
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]
        # 如果未提供系统提示，则使用默认的 RAG 代理提示
        if system_prompt is None:
            system_prompt = "你是一个检索增强生成（RAG）代理。请使用检索到的文档准确、简洁地回答用户的查询。"
        # 调用父类的构造函数进行初始化
        super().__init__(system_prompt=system_prompt, toolsets=toolsets)
