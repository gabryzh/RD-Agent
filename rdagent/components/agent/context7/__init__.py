# 导入类型提示和其他模块
from typing import Optional

from pydantic_ai.mcp import MCPServerStreamableHTTP

from rdagent.components.agent.base import PAIAgent
from rdagent.components.agent.context7.conf import SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T


class Agent(PAIAgent):
    """
    一个专门用于 context7 的特定代理。
    """

    def __init__(self):
        """
        初始化 context7 代理。
        设置工具集（连接到 MCP 服务器）和系统提示。
        """
        toolsets = [MCPServerStreamableHTTP(SETTINGS.url, timeout=SETTINGS.timeout)]

        super().__init__(
            system_prompt=T(".prompts:system_prompt").r(),
            toolsets=toolsets,
            enable_cache=SETTINGS.enable_cache,
        )

    def _build_enhanced_query(self, error_message: str, full_code: Optional[str] = None) -> str:
        """
        使用实验性的提示模板构建增强型查询。
        """
        # 使用模板构建上下文信息
        context_info = ""
        if full_code:
            context_info = T(".prompts:code_context_template").r(full_code=full_code)

        # 检查 timm 库的特殊情况（实验性优化）
        timm_trigger = error_message.lower().count("timm") >= 3
        timm_trigger_text = ""
        if timm_trigger:
            timm_trigger_text = T(".prompts:timm_special_case").r()
            logger.info("🎯 触发了 Timm 特殊处理", tag="context7")

        # 使用实验性模板构建增强型查询
        enhanced_query = T(".prompts:context7_enhanced_query_template").r(
            error_message=error_message, context_info=context_info, timm_trigger_text=timm_trigger_text
        )

        return enhanced_query

    def query(self, query: str) -> str:
        """
        执行查询。

        参数
        ----------
        query : str
            查询内容，应该类似于错误消息。

        返回
        -------
        str
            查询结果。
        """
        query = self._build_enhanced_query(error_message=query)
        return super().query(query)
