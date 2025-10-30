from abc import abstractmethod

import nest_asyncio
from prefect import task
from prefect.cache_policies import INPUTS
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.oai.backend.pydantic_ai import get_agent_model


class BaseAgent:
    """代理的基类，定义了代理应具备的基本接口。"""

    @abstractmethod
    def __init__(self, system_prompt: str, toolsets: list[str]):
        """初始化代理。"""
        ...

    @abstractmethod
    def query(self, query: str) -> str:
        """执行查询并返回结果。"""
        ...


class PAIAgent(BaseAgent):
    """
    Pydantic-AI 代理，支持可选的 Prefect 缓存功能。
    """

    agent: Agent
    enable_cache: bool

    def __init__(
        self,
        system_prompt: str,
        toolsets: list[str | MCPServerStreamableHTTP],
        enable_cache: bool = False,
    ):
        """
        初始化 Pydantic-AI 代理。

        参数
        ----------
        system_prompt : str
            代理的系统提示。
        toolsets : list[str | MCPServerStreamableHTTP]
            MCP 服务器 URL 或实例的列表。
        enable_cache : bool
            通过 Prefect 启用持久缓存。需要 Prefect 服务器：
            `prefect server start` 然后在环境中设置 PREFECT_API_URL。
        """
        toolsets = [(ts if isinstance(ts, MCPServerStreamableHTTP) else MCPServerStreamableHTTP(ts)) for ts in toolsets]
        self.agent = Agent(get_agent_model(), system_prompt=system_prompt, toolsets=toolsets)
        self.enable_cache = enable_cache

        # 如果启用缓存，则创建带缓存的查询函数
        if enable_cache:
            self._cached_query = task(cache_policy=INPUTS, persist_result=True)(self._run_query)

    def _run_query(self, query: str) -> str:
        """
        内部查询执行（无缓存）。
        """
        nest_asyncio.apply()  # 注意：非常重要。因为 pydantic-ai 使用了 asyncio！
        result = self.agent.run_sync(query)
        return result.output

    def query(self, query: str) -> str:
        """
        运行带可选缓存的代理查询。

        参数
        ----------
        query : str
            要执行的查询。

        返回
        -------
        str
            查询结果。
        """
        if self.enable_cache:
            return self._cached_query(query)
        else:
            return self._run_query(query)
