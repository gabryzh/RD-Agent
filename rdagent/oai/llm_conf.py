from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from rdagent.core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    """
    LLM 设置类，用于配置语言模型的行为。
    """
    # 后端设置
    backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"

    chat_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-small"

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    enable_response_schema: bool = True
    # 是否在聊天模型中启用响应模式。对于不支持的模型可能不起作用。

    # 格式处理
    reasoning_think_rm: bool = False
    """
    一些 LLM 在其响应中包含 <think>...</think> 标签，这可能会干扰主要输出。
    将 reasoning_think_rm 设置为 True 以从响应中删除任何 <think>...</think> 内容。
    """

    # TODO: 大多数设置仅在 deprec.DeprecBackend 中使用。
    # 所以应该将这些设置移动到该文件夹。

    log_llm_chat_content: bool = True

    use_azure: bool = Field(default=False, deprecated=True)
    chat_use_azure: bool = False
    embedding_use_azure: bool = False

    chat_use_azure_token_provider: bool = False
    embedding_use_azure_token_provider: bool = False
    managed_identity_client_id: str | None = None
    max_retry: int = 10
    retry_wait_seconds: int = 1
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10
    timeout_fail_limit: int = 10
    violation_fail_limit: int = 1

    # 启用缓存时返回相同问题答案的行为
    use_auto_chat_cache_seed_gen: bool = False
    """
    `_create_chat_completion_inner_function` 提供了一个传入种子以影响缓存哈希键的功能。
    我们希望启用一个自动种子生成器，以便在未给出种子的情况下为 `_create_chat_completion_inner_function` 获取不同的默认种子。
    因此，只有在同一轮中问相同的问题时，缓存才不会丢失。
    """
    init_chat_cache_seed: int = 42

    # 聊天配置
    openai_api_key: str = ""  # TODO: 简化密钥设计。
    chat_openai_api_key: str | None = None
    chat_openai_base_url: str | None = None
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    chat_max_tokens: int | None = None
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = (
        100000  # 100000 是 gpt4 的最大限制，未来版本的 gpt 可能会增加
    )
    default_system_prompt: str = "你是一个AI助手，帮助回答用户的问题。"
    system_prompt_role: str = "system"
    """一些模型（如 o1）不支持“系统”角色。
    因此，我们将 system_prompt_role 设置为可自定义，以确保成功调用。"""

    # 嵌入配置
    embedding_openai_api_key: str = ""
    embedding_openai_base_url: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""
    embedding_max_str_num: int = 50
    embedding_max_length: int = 8192

    # 离线 llama2 相关配置
    use_llama2: bool = False
    llama2_ckpt_dir: str = "Llama-2-7b-chat"
    llama2_tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
    llams2_max_batch_size: int = 8

    # 服务器提供的端点
    use_gcr_endpoint: bool = False
    gcr_endpoint_type: str = "llama2_70b"  # 或 "llama3_70b", "phi2", "phi3_4k", "phi3_128k"

    llama2_70b_endpoint: str = ""
    llama2_70b_endpoint_key: str = ""
    llama2_70b_endpoint_deployment: str = ""

    llama3_70b_endpoint: str = ""
    llama3_70b_endpoint_key: str = ""
    llama3_70b_endpoint_deployment: str = ""

    phi2_endpoint: str = ""
    phi2_endpoint_key: str = ""
    phi2_endpoint_deployment: str = ""

    phi3_4k_endpoint: str = ""
    phi3_4k_endpoint_key: str = ""
    phi3_4k_endpoint_deployment: str = ""

    phi3_128k_endpoint: str = ""
    phi3_128k_endpoint_key: str = ""
    phi3_128k_endpoint_deployment: str = ""

    gcr_endpoint_temperature: float = 0.7
    gcr_endpoint_top_p: float = 0.9
    gcr_endpoint_do_sample: bool = False
    gcr_endpoint_max_token: int = 100

    chat_use_azure_deepseek: bool = False
    chat_azure_deepseek_endpoint: str = ""
    chat_azure_deepseek_key: str = ""

    chat_model_map: dict[str, dict[str, str]] = {}


LLM_SETTINGS = LLMSettings()
