"""
用于处理令牌限制和文本截断的嵌入实用程序。
"""

from typing import Optional

from litellm import decode, encode, get_max_tokens, token_counter

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS

# 常见的嵌入模型令牌限制
EMBEDDING_MODEL_LIMITS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "Qwen3-Embedding-8B": 32000,
    "Qwen3-Embedding-4B": 32000,
    "Qwen3-Embedding-0.6B": 32000,
    "bge-m3": 8191,
    "bce-embedding-base_v1": 511,
    "bge-large-zh-v1.5": 511,
    "bge-large-en-v1.5": 511,
}


def get_embedding_max_tokens(model: str) -> int:
    """
    获取嵌入模型的最大令牌限制。

    三级回退策略：
    1. 使用 litellm.get_max_tokens()
    2. 查询 EMBEDDING_MODEL_LIMITS 映射
    3. 使用默认值 8192

    参数：
        model：模型名称

    返回：
        最大令牌限制
    """
    # 移除前缀（例如，"provider/model" -> "model"）
    model_name = model.split("/")[-1] if "/" in model else model

    # 级别 1：尝试 litellm
    try:
        max_tokens = get_max_tokens(model_name)
        if max_tokens and max_tokens > 0:
            return max_tokens
    except Exception as e:
        logger.warning(f"无法获取 {model_name} 的最大令牌数：{e}")

    # 级别 2：查询映射表
    if model_name in EMBEDDING_MODEL_LIMITS:
        return EMBEDDING_MODEL_LIMITS[model_name]

    # 级别 3：回退到 LLM_SETTINGS.embedding_max_length
    default_max_tokens = LLM_SETTINGS.embedding_max_length
    logger.warning(f"未知的嵌入模型 {model}，使用默认 max_tokens={default_max_tokens}")
    return default_max_tokens


def trim_text_for_embedding(text: str, model: str, max_tokens: Optional[int] = None) -> str:
    """
    使用编码/解码方法截断嵌入模型的文本。

    参数：
        text：输入文本
        model：模型名称
        max_tokens：最大令牌限制，如果为 None 则自动检测。如果仍然超过限制，
                   则引发错误，指示用户设置 LLM_SETTINGS.embedding_max_length

    返回：
        截断后的文本
    """
    if not text:
        return ""

    # 获取模型的最大令牌限制
    if max_tokens is None:
        max_tokens = get_embedding_max_tokens(model)

    # 应用安全边际
    safe_max_tokens = int(max_tokens * 0.9)

    # 计算当前令牌数
    current_tokens = token_counter(model=model, text=text)

    if current_tokens <= safe_max_tokens:
        return text

    logger.warning(
        f"文本对于嵌入模型 {model} 过长： "
        f"{current_tokens} 令牌 > {safe_max_tokens} 限制（带安全边际）。 "
        f"正在使用编码/解码方法进行截断。"
    )

    try:
        # 使用编码/解码方法进行精确截断
        enc_ids = encode(model=model, text=text)
        enc_ids_trunc = enc_ids[:safe_max_tokens]
        text_trunc = decode(model=model, tokens=enc_ids_trunc)
        # 确保返回字符串类型（mypy 类型安全）
        text_trunc = str(text_trunc) if text_trunc is not None else ""

        final_tokens = token_counter(model=model, text=text_trunc)
        logger.warning(f"截断完成：{current_tokens} -> {final_tokens} 令牌")

        return text_trunc
    except Exception as e:
        raise RuntimeError(
            f"无法为嵌入模型 {model} 截断文本。 "
            f"请将 LLM_SETTINGS.embedding_max_length 设置为较小的值。 "
            f"原始错误：{e}"
        ) from e


def truncate_content_list(content_list: list[str], model: str) -> list[str]:
    """
    截断内容字符串列表。

    参数：
        content_list：要截断的内容字符串列表
        model：模型名称

    返回：
        截断后的内容字符串列表
    """
    truncated_list = []
    for content in content_list:
        truncated_content = trim_text_for_embedding(content, model)
        truncated_list.append(truncated_content)

    return truncated_list
