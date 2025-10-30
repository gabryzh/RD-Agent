import copyreg
from typing import Any, Literal, Optional, Type, TypedDict, Union, cast

import numpy as np
from litellm import (
    BadRequestError,
    completion,
    completion_cost,
    embedding,
    get_model_info,
    supports_function_calling,
    supports_response_schema,
    token_counter,
)
from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings


# 注意：修补！否则，异常将调用构造函数并出现以下错误：
# `BadRequestError.__init__() 缺少 2 个必需的位置参数：'model' 和 'llm_provider'`
def _reduce_no_init(exc: Exception) -> tuple:
    cls = exc.__class__
    return (cls.__new__, (cls,), exc.__dict__)


# 假设你想将此应用于 MyError
copyreg.pickle(BadRequestError, _reduce_no_init)


class LiteLLMSettings(LLMSettings):

    class Config:
        env_prefix = "LITELLM_"
        """使用 `LITELLM_` 作为环境变量的前缀"""

    # LiteLLM 特定设置的占位符，目前为空


LITELLM_SETTINGS = LiteLLMSettings()
ACC_COST = 0.0


class LiteLLMAPIBackend(APIBackend):
    """APIBackend 接口的 LiteLLM 实现"""

    _has_logged_settings: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self.__class__._has_logged_settings:
            logger.info(f"{LITELLM_SETTINGS}")
            logger.log_object(LITELLM_SETTINGS.model_dump(), tag="LITELLM_SETTINGS")
            self.__class__._has_logged_settings = True
        super().__init__(*args, **kwargs)

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        从消息中计算令牌数
        """
        num_tokens = token_counter(
            model=LITELLM_SETTINGS.chat_model,
            messages=messages,
        )
        logger.info(f"{LogColors.CYAN}令牌计数: {LogColors.END} {num_tokens}", tag="debug_litellm_token")
        return num_tokens

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        调用嵌入函数
        """
        model_name = LITELLM_SETTINGS.embedding_model
        logger.info(f"{LogColors.GREEN}使用 emb 模型{LogColors.END} {model_name}", tag="debug_litellm_emb")
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(
                f"{LogColors.MAGENTA}正在为以下内容创建嵌入{LogColors.END}: {input_content_list}",
                tag="debug_litellm_emb",
            )
        response = embedding(
            model=model_name,
            input=input_content_list,
        )
        response_list = [data["embedding"] for data in response.data]
        return response_list

    class CompleteKwargs(TypedDict):
        model: str
        temperature: float
        max_tokens: int | None
        reasoning_effort: Literal["low", "medium", "high"] | None

    def get_complete_kwargs(self) -> CompleteKwargs:
        """
        返回完成的几个关键设置
        从设置中获取这些值使得在代理系统中适应后端调用更加容易。
        """
        # 调用 LiteLLM 完成
        model = LITELLM_SETTINGS.chat_model
        temperature = LITELLM_SETTINGS.chat_temperature
        max_tokens = LITELLM_SETTINGS.chat_max_tokens
        reasoning_effort = LITELLM_SETTINGS.reasoning_effort

        if LITELLM_SETTINGS.chat_model_map:
            for t, mc in LITELLM_SETTINGS.chat_model_map.items():
                if t in logger._tag:
                    model = mc["model"]
                    if "temperature" in mc:
                        temperature = float(mc["temperature"])
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    if "reasoning_effort" in mc:
                        if mc["reasoning_effort"] in ["low", "medium", "high"]:
                            reasoning_effort = cast(Literal["low", "medium", "high"], mc["reasoning_effort"])
                        else:
                            reasoning_effort = None
                    break
        return self.CompleteKwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        调用聊天完成函数
        """

        if response_format and not supports_response_schema(model=LITELLM_SETTINGS.chat_model):
            # Deepseek 将进入此分支
            logger.warning(
                f"{LogColors.YELLOW}模型 {LITELLM_SETTINGS.chat_model} 不支持响应模式，将忽略 response_format 参数。{LogColors.END}",
                tag="llm_messages",
            )
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        complete_kwargs = self.get_complete_kwargs()
        model = complete_kwargs["model"]

        response = completion(
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            max_retries=0,
            **complete_kwargs,
            **kwargs,
        )
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(f"{LogColors.GREEN}使用聊天模型{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.BLUE}助手:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                if "content" in message["choices"][0]["delta"]:
                    chunk = (
                        message["choices"][0]["delta"]["content"] or ""
                    )  # 当 finish_reason 为 "stop" 时，内容为 None
                    content += chunk
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}完成原因: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}助手:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

        global ACC_COST
        try:
            cost = completion_cost(model=model, messages=messages, completion=content)
        except Exception as e:
            logger.warning(f"模型 {model} 的成本计算失败: {e}。跳过成本统计。")
            cost = np.nan
        else:
            ACC_COST += cost
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"当前成本: ${float(cost):.10f}; 累计成本: ${float(ACC_COST):.10f}; {finish_reason=}",
                )

        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=content)
        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )
        return content, finish_reason

    def supports_response_schema(self) -> bool:
        """
        检查后端是否支持函数调用
        """
        return supports_response_schema(model=LITELLM_SETTINGS.chat_model) and LITELLM_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        """建议输入令牌限制，确保上下文窗口中有足够的空间用于最大输出令牌。"""
        try:
            model_info = get_model_info(LITELLM_SETTINGS.chat_model)
            if model_info is None:
                return super().chat_token_limit

            max_input = model_info.get("max_input_tokens")
            max_output = model_info.get("max_output_tokens")

            if max_input is None or max_output is None:
                return super().chat_token_limit

            max_input_tokens = max_input - max_output
            return max_input_tokens
        except Exception as e:
            return super().chat_token_limit
