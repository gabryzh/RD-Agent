# type: ignore
from __future__ import annotations

import inspect
import json
import os
import random
import re
import sqlite3
import ssl
import time
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type, Union, cast

import numpy as np
import openai
import tiktoken
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass, import_class
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils import md5_hash

DEFAULT_QLIB_DOT_PATH = Path("./")

from rdagent.oai.backend.base import APIBackend

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    logger.warning("azure.identity 未安装。")

try:
    import openai
except ImportError:
    logger.warning("openai 未安装。")

try:
    from llama import Llama
except ImportError:
    if LLM_SETTINGS.use_llama2:
        logger.warning("llama 未安装。")

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        AssistantMessage,
        ChatRequestMessage,
        SystemMessage,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    if LLM_SETTINGS.chat_use_azure_deepseek:
        logger.warning("azure.ai.inference 或 azure.core.credentials 未安装。")


class ConvManager:
    """
    这是 LLM 的对话管理器
    它便于导出对话以进行调试。
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_QLIB_DOT_PATH / "llm_conv",
        recent_n: int = 10,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.recent_n = recent_n

    def _rotate_files(self) -> None:
        """轮换文件"""
        pairs = []
        for f in self.path.glob("*.json"):
            m = re.match(r"(\d+).json", f.name)
            if m is not None:
                n = int(m.group(1))
                pairs.append((n, f))
        pairs.sort(key=lambda x: x[0])
        for n, f in pairs[: self.recent_n][::-1]:
            if (self.path / f"{n+1}.json").exists():
                (self.path / f"{n+1}.json").unlink()
            f.rename(self.path / f"{n+1}.json")

    def append(self, conv: tuple[list, str]) -> None:
        """追加对话"""
        self._rotate_files()
        with (self.path / "0.json").open("w") as file:
            json.dump(conv, file)
        # TODO: 保留换行符，使其更便于直接编辑文件。


class DeprecBackend(APIBackend):
    """
    这是不同后端的统一接口。

    (xiao) 认为将各种 API 集成到一个类中不是一个好的设计。
    所以我们将来应该将它们拆分到 `oai/backends/` 中的不同类中。
    """

    # FIXME: (xiao) 我们应该避免使用 self.xxxx。
    # 相反，我们可以直接使用 LLM_SETTINGS。如果支持不同的后端设置很困难，我们可以将它们拆分为多个 BaseSettings。
    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if LLM_SETTINGS.use_llama2:
            self.generator = Llama.build(
                ckpt_dir=LLM_SETTINGS.llama2_ckpt_dir,
                tokenizer_path=LLM_SETTINGS.llama2_tokenizer_path,
                max_seq_len=LLM_SETTINGS.chat_max_tokens,
                max_batch_size=LLM_SETTINGS.llams2_max_batch_size,
            )
            self.encoder = None
        elif LLM_SETTINGS.use_gcr_endpoint:
            gcr_endpoint_type = LLM_SETTINGS.gcr_endpoint_type
            if gcr_endpoint_type == "llama2_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama2_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama2_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama2_70b_endpoint
            elif gcr_endpoint_type == "llama3_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama3_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama3_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama3_70b_endpoint
            elif gcr_endpoint_type == "phi2":
                self.gcr_endpoint_key = LLM_SETTINGS.phi2_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi2_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi2_endpoint
            elif gcr_endpoint_type == "phi3_4k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_4k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_4k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_4k_endpoint
            elif gcr_endpoint_type == "phi3_128k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_128k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_128k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_128k_endpoint
            else:
                error_message = f"无效的 gcr_endpoint_type: {gcr_endpoint_type}"
                raise ValueError(error_message)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.gcr_endpoint_key),
            }
            self.gcr_endpoint_temperature = LLM_SETTINGS.gcr_endpoint_temperature
            self.gcr_endpoint_top_p = LLM_SETTINGS.gcr_endpoint_top_p
            self.gcr_endpoint_do_sample = LLM_SETTINGS.gcr_endpoint_do_sample
            self.gcr_endpoint_max_token = LLM_SETTINGS.gcr_endpoint_max_token
            if not os.environ.get("PYTHONHTTPSVERIFY", "") and hasattr(ssl, "_create_unverified_context"):
                ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.chat_model = LLM_SETTINGS.chat_model
            self.encoder = None
        elif LLM_SETTINGS.chat_use_azure_deepseek:
            self.client = ChatCompletionsClient(
                endpoint=LLM_SETTINGS.chat_azure_deepseek_endpoint,
                credential=AzureKeyCredential(LLM_SETTINGS.chat_azure_deepseek_key),
            )
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.encoder = None
            self.chat_model = "deepseek-R1"
            self.chat_stream = LLM_SETTINGS.chat_stream
        else:
            self.chat_use_azure = LLM_SETTINGS.chat_use_azure or LLM_SETTINGS.use_azure
            self.embedding_use_azure = LLM_SETTINGS.embedding_use_azure or LLM_SETTINGS.use_azure
            self.chat_use_azure_token_provider = LLM_SETTINGS.chat_use_azure_token_provider
            self.embedding_use_azure_token_provider = LLM_SETTINGS.embedding_use_azure_token_provider
            self.managed_identity_client_id = LLM_SETTINGS.managed_identity_client_id

            # 优先级: chat_api_key/embedding_api_key > openai_api_key > os.environ.get("OPENAI_API_KEY")
            # TODO: 简化密钥设计。考虑 Pandatic 的字段别名和优先级。
            self.chat_api_key = (
                LLM_SETTINGS.chat_openai_api_key or LLM_SETTINGS.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
            self.embedding_api_key = (
                LLM_SETTINGS.embedding_openai_api_key or LLM_SETTINGS.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )

            self.chat_model = LLM_SETTINGS.chat_model
            self.chat_model_map = LLM_SETTINGS.chat_model_map
            self.encoder = self._get_encoder()
            self.chat_openai_base_url = LLM_SETTINGS.chat_openai_base_url
            self.embedding_openai_base_url = LLM_SETTINGS.embedding_openai_base_url
            self.chat_api_base = LLM_SETTINGS.chat_azure_api_base
            self.chat_api_version = LLM_SETTINGS.chat_azure_api_version
            self.chat_stream = LLM_SETTINGS.chat_stream
            self.chat_seed = LLM_SETTINGS.chat_seed

            self.embedding_model = LLM_SETTINGS.embedding_model
            self.embedding_api_base = LLM_SETTINGS.embedding_azure_api_base
            self.embedding_api_version = LLM_SETTINGS.embedding_azure_api_version

            if (self.chat_use_azure or self.embedding_use_azure) and (
                self.chat_use_azure_token_provider or self.embedding_use_azure_token_provider
            ):
                dac_kwargs = {}
                if self.managed_identity_client_id is not None:
                    dac_kwargs["managed_identity_client_id"] = self.managed_identity_client_id
                credential = DefaultAzureCredential(**dac_kwargs)
                token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default",
                )
            self.chat_client: openai.OpenAI = (
                openai.AzureOpenAI(
                    azure_ad_token_provider=token_provider if self.chat_use_azure_token_provider else None,
                    api_key=self.chat_api_key if not self.chat_use_azure_token_provider else None,
                    api_version=self.chat_api_version,
                    azure_endpoint=self.chat_api_base,
                )
                if self.chat_use_azure
                else openai.OpenAI(api_key=self.chat_api_key, base_url=self.chat_openai_base_url)
            )

            self.embedding_client: openai.OpenAI = (
                openai.AzureOpenAI(
                    azure_ad_token_provider=token_provider if self.embedding_use_azure_token_provider else None,
                    api_key=self.embedding_api_key if not self.embedding_use_azure_token_provider else None,
                    api_version=self.embedding_api_version,
                    azure_endpoint=self.embedding_api_base,
                )
                if self.embedding_use_azure
                else openai.OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_openai_base_url)
            )

        # 如果配置在运行时不应更改，则将其传输到类
        self.use_llama2 = LLM_SETTINGS.use_llama2
        self.use_gcr_endpoint = LLM_SETTINGS.use_gcr_endpoint
        self.chat_use_azure_deepseek = LLM_SETTINGS.chat_use_azure_deepseek

    def _get_encoder(self) -> tiktoken.Encoding:
        """
        tiktoken.encoding_for_model(self.chat_model) 并未涵盖所有应考虑的情况。

        此函数尝试处理几个边缘情况。
        """

        # 1) 案例
        def _azure_patch(model: str) -> str:
            """
            使用 Azure API 时，self.chat_model 是可以是任何字符串的部署名称。
            例如，它可能是 `gpt-4o_2024-08-06`。但是 tiktoken.encoding_for_model 无法处理此问题。
            """
            return model.replace("_", "-")

        model = self.chat_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"获取编码器失败。正在尝试修补模型名称")
            for patch_func in [_azure_patch]:
                try:
                    encoding = tiktoken.encoding_for_model(patch_func(model))
                except KeyError:
                    logger.error(f"即使在使用 {patch_func.__name__} 修补后也无法获取编码器")
                    raise
        return encoding

    def supports_response_schema(self) -> bool:
        """
        检查后端是否支持函数调用。
        目前，弃用的后端不支持函数调用，因此返回 False。 #FIXME: 可能需要一个到后端类的映射。
        """
        return False

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """创建嵌入内部函数"""
        content_to_embedding_dict = {}
        for sliced_filtered_input_content_list in [
            input_content_list[i : i + LLM_SETTINGS.embedding_max_str_num]
            for i in range(0, len(input_content_list), LLM_SETTINGS.embedding_max_str_num)
        ]:
            if self.embedding_use_azure:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=sliced_filtered_input_content_list,
                )
            else:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=sliced_filtered_input_content_list,
                )
            for index, data in enumerate(response.data):
                content_to_embedding_dict[sliced_filtered_input_content_list[index]] = data.embedding

        return [content_to_embedding_dict[content] for content in input_content_list]

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        add_json_in_prompt: bool = False,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        seed : Optional[int]
            在启用缓存的情况下重试时，它将继续返回相同的结果。
            为了使重试有用，我们需要启用一个种子。
            此种子不同于 GPT 的 `self.chat_seed`。它用于 RD-Agent 本地启用的本地缓存机制。
        """

        # TODO: 我们可以将此函数加回来，以避免如此多的 `self.cfg.log_llm_chat_content`
        if LLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")
        # TODO: 由于流响应，无法使用 loguru 适配器

        model = LLM_SETTINGS.chat_model
        temperature = LLM_SETTINGS.chat_temperature
        max_tokens = LLM_SETTINGS.chat_max_tokens
        frequency_penalty = LLM_SETTINGS.chat_frequency_penalty
        presence_penalty = LLM_SETTINGS.chat_presence_penalty

        if self.chat_model_map:
            for t, mc in self.chat_model_map.items():
                if t in logger._tag:
                    model = mc.get("model", model)
                    temperature = float(mc.get("temperature", temperature))
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    break

        finish_reason = None
        if self.use_llama2:
            response = self.generator.chat_completion(
                messages,
                max_gen_len=max_tokens,
                temperature=temperature,
            )
            resp = response[0]["generation"]["content"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}响应:{resp}{LogColors.END}", tag="llm_messages")
        elif self.use_gcr_endpoint:
            body = str.encode(
                json.dumps(
                    {
                        "input_data": {
                            "input_string": messages,
                            "parameters": {
                                "temperature": self.gcr_endpoint_temperature,
                                "top_p": self.gcr_endpoint_top_p,
                                "max_new_tokens": self.gcr_endpoint_max_token,
                            },
                        },
                    },
                ),
            )

            req = urllib.request.Request(self.gcr_endpoint, body, self.headers)  # noqa: S310
            response = urllib.request.urlopen(req)  # noqa: S310
            resp = json.loads(response.read().decode())["output"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}响应:{resp}{LogColors.END}", tag="llm_messages")
        elif self.chat_use_azure_deepseek:
            azure_style_message: list[ChatRequestMessage] = []
            for message in messages:
                if message["role"] == "system":
                    azure_style_message.append(SystemMessage(content=message["content"]))
                elif message["role"] == "user":
                    azure_style_message.append(UserMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    azure_style_message.append(AssistantMessage(content=message["content"]))

            response = self.client.complete(
                messages=azure_style_message,
                stream=self.chat_stream,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if self.chat_stream:
                resp = ""
                # TODO: 使用 logger.config(stream=self.chat_stream): 并添加一个 `stream_start` 标志为第一条消息添加时间戳。
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}响应:{LogColors.END}", tag="llm_messages")

                for chunk in response:
                    content = (
                        chunk.choices[0].delta.content
                        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None
                        else ""
                    )
                    if LLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + content + LogColors.END, raw=True, tag="llm_messages")
                    resp += content
                    if len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
            else:
                response = cast(ChatCompletion, response)
                resp = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}响应:{resp}{LogColors.END}", tag="llm_messages")
            match = re.search(r"<think>(.*?)</think>(.*)", resp, re.DOTALL)
            think_part, resp = match.groups() if match else ("", resp)
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}思考:{think_part}{LogColors.END}", tag="llm_messages")
                logger.info(f"{LogColors.CYAN}响应:{resp}{LogColors.END}", tag="llm_messages")
        else:
            call_kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=self.chat_stream,
                seed=self.chat_seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # 修复模型不支持 response_schema 的情况
            if response_format == {"type": "json_object"} and add_json_in_prompt:
                for message in messages[::-1]:
                    message["content"] = message["content"] + "\n请以 json 格式响应。"
                    if message["role"] == LLM_SETTINGS.system_prompt_role:
                        # 注意：假设 systemprompt 始终是第一条消息
                        break
                call_kwargs["response_format"] = {"type": "json_object"}
            response = self.chat_client.chat.completions.create(**call_kwargs)

            if self.chat_stream:
                resp = ""
                # TODO: 使用 logger.config(stream=self.chat_stream): 并添加一个 `stream_start` 标志为第一条消息添加时间戳。
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}响应:{LogColors.END}", tag="llm_messages")

                for chunk in response:
                    content = (
                        chunk.choices[0].delta.content
                        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None
                        else ""
                    )
                    if LLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + content + LogColors.END, raw=True, tag="llm_messages")
                    resp += content
                    if len(chunk.choices) > 0 and chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason

                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info("\n", raw=True, tag="llm_messages")

            else:
                resp = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(f"{LogColors.CYAN}响应:{resp}{LogColors.END}", tag="llm_messages")
                    logger.info(
                        json.dumps(
                            {
                                "total_tokens": response.usage.total_tokens,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "model": model,
                            }
                        ),
                        tag="llm_messages",
                    )
        return resp, finish_reason

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """从消息中计算令牌数"""
        if self.chat_use_azure_deepseek:
            return 0
        if self.encoder is None:
            raise ValueError("编码器未初始化。")
        if self.use_llama2 or self.use_gcr_endpoint:
            logger.warning("num_tokens_from_messages() 未为模型 llama2 实现。")
            return 0  # TODO 为 llama2 实现此函数

        if "gpt4" in self.chat_model or "gpt-4" in self.chat_model:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 4  # 每条消息都遵循 <start>{role/name}\n{content}<end>\n
            tokens_per_name = -1  # 如果有名称，则省略角色
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # 每个回复都以 <start>assistant<message> 开头
        return num_tokens
