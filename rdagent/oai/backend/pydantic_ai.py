"""
pydantic-ai 的适配器工具
"""

import os

from litellm.utils import get_llm_provider
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.litellm import LiteLLMProvider

from rdagent.oai.backend.litellm import LiteLLMAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

# 注意：
# LiteLLM 的代码组织得不是很好。
# 我们无法重用任何组件来将提供商映射到环境名称
# 所以我们必须在这里硬编码。
PROVIDER_TO_ENV_MAP = {
    "openai": "OPENAI",
    "azure_ai": "AZURE_AI",
    "azure": "AZURE",
    "litellm_proxy": "LITELLM_PROXY",
}


def get_agent_model() -> OpenAIChatModel:
    """
    将 LiteLLM 转换为 pydantic-ai 模型。所以你可以这样使用

    .. code-block:: python

        from rdagent.oai.backend.pydantic_ai import get_agent_model
        model = get_agent_model()
        agent = Agent(model)

    """
    backend = APIBackend()
    assert isinstance(backend, LiteLLMAPIBackend), "仅支持 LiteLLMAPIBackend"

    compl_kwargs = backend.get_complete_kwargs()

    selected_model = compl_kwargs["model"]

    _, custom_llm_provider, _, _ = get_llm_provider(selected_model)
    assert (
        custom_llm_provider in PROVIDER_TO_ENV_MAP
    ), f"不支持提供商 {custom_llm_provider}。请将其添加到 `PROVIDER_TO_ENV_MAP`"
    prefix = PROVIDER_TO_ENV_MAP[custom_llm_provider]
    api_key = os.getenv(f"{prefix}_API_KEY", None)
    api_base = os.getenv(f"{prefix}_API_BASE", None)

    kwargs = {
        "openai_reasoning_effort": compl_kwargs.get("reasoning_effort"),
        "max_tokens": compl_kwargs.get("max_tokens"),
        "temperature": compl_kwargs.get("temperature"),
    }
    if compl_kwargs.get("max_tokens") is None:
        kwargs["max_tokens"] = LLM_SETTINGS.chat_max_tokens
    settings = OpenAIChatModelSettings(**kwargs)
    return OpenAIChatModel(
        selected_model, provider=LiteLLMProvider(api_base=api_base, api_key=api_key), settings=settings
    )
