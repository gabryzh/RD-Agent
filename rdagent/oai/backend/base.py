from __future__ import annotations

import io
import json
import re
import sqlite3
import time
import tokenize
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast

import pytz
from pydantic import BaseModel, TypeAdapter

from rdagent.core.exception import PolicyError
from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.utils.embedding import truncate_content_list
from rdagent.utils import md5_hash

try:
    import litellm
    import openai

    openai_imported = True
except ImportError:
    openai_imported = False


class JSONParser:
    """支持多种策略的 JSON 解析器"""

    def __init__(self, add_json_in_prompt: bool = False) -> None:
        self.strategies: List[Callable[[str], str]] = [
            self._direct_parse,
            self._extract_from_code_block,
            self._fix_python_syntax,
            self._extract_with_fix_combined,
        ]
        self.add_json_in_prompt = add_json_in_prompt

    def parse(self, content: str) -> str:
        """解析 JSON 内容，自动尝试多种策略"""
        original_content = content

        for strategy in self.strategies:
            try:
                return strategy(original_content)
            except json.JSONDecodeError:
                continue

        # 所有策略都失败了
        if not self.add_json_in_prompt:
            error = json.JSONDecodeError(
                "所有尝试都失败后无法解析 JSON，可能是因为 'messages' 必须以某种形式包含单词 'json'",
                original_content,
                0,
            )
            error.message = "所有尝试都失败后无法解析 JSON，可能是因为 'messages' 必须以某种形式包含单词 'json'"  # type: ignore[attr-defined]
            raise error
        else:
            raise json.JSONDecodeError("所有尝试都失败后无法解析 JSON", original_content, 0)

    def _direct_parse(self, content: str) -> str:
        """策略 1：直接解析（包括处理额外数据）"""
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                return self._extract_first_json(content)
            raise

    def _extract_from_code_block(self, content: str) -> str:
        """策略 2：从代码块中提取 JSON"""
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("未找到 JSON 代码块", content, 0)

        json_content = match.group(1).strip()
        return self._direct_parse(json_content)

    def _fix_python_syntax(self, content: str) -> str:
        """策略 3：解析前修复 Python 语法"""
        fixed = self._fix_python_booleans(content)
        return self._direct_parse(fixed)

    def _extract_with_fix_combined(self, content: str) -> str:
        """策略 4：组合策略 - 先修复 Python 语法，然后提取第一个 JSON 对象"""
        fixed = self._fix_python_booleans(content)

        # 尝试从修复后的内容中提取代码块
        match = re.search(r"```json\s*(.*?)\s*```", fixed, re.DOTALL)
        if match:
            fixed = match.group(1).strip()

        return self._direct_parse(fixed)

    @staticmethod
    def _fix_python_booleans(json_str: str) -> str:
        """使用 tokenize 安全地将 Python 风格的布尔值修复为 JSON 标准格式"""
        replacements = {"True": "true", "False": "false", "None": "null"}

        try:
            out = []
            io_string = io.StringIO(json_str)
            tokens = tokenize.generate_tokens(io_string.readline)

            for toknum, tokval, _, _, _ in tokens:
                if toknum == tokenize.NAME and tokval in replacements:
                    out.append(replacements[tokval])
                else:
                    out.append(tokval)

            result = "".join(out)
            return result

        except (tokenize.TokenError, json.JSONDecodeError):
            # 如果 tokenize 失败，则回退到正则表达式方法
            for python_val, json_val in replacements.items():
                json_str = re.sub(rf"\b{python_val}\b", json_val, json_str)
            return json_str

    @staticmethod
    def _extract_first_json(response: str) -> str:
        """提取第一个完整的 JSON 对象，忽略额外内容"""
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(response)
        return json.dumps(obj)


class SQliteLazyCache(SingletonBaseClass):
    """SQLite 延迟缓存"""
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        # TODO: sqlite3 不支持多进程。
        self.conn = sqlite3.connect(cache_location, timeout=20)
        self.c = self.conn.cursor()
        if not db_file_exist:
            self.c.execute(
                """
                CREATE TABLE chat_cache (
                    md5_key TEXT PRIMARY KEY,
                    chat TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE embedding_cache (
                    md5_key TEXT PRIMARY KEY,
                    embedding TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE message_cache (
                    conversation_id TEXT PRIMARY KEY,
                    message TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        """获取聊天缓存"""
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        """获取嵌入缓存"""
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        """设置聊天缓存"""
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()
        return None

    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        """设置嵌入缓存"""
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        """获取消息缓存"""
        self.c.execute("SELECT message FROM message_cache WHERE conversation_id=?", (conversation_id,))
        result = self.c.fetchone()
        return [] if result is None else cast(list[dict[str, Any]], json.loads(result[0]))

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        """设置消息缓存"""
        self.c.execute(
            "INSERT OR REPLACE INTO message_cache (conversation_id, message) VALUES (?, ?)",
            (conversation_id, json.dumps(message_value)),
        )
        self.conn.commit()
        return None


class SessionChatHistoryCache(SingletonBaseClass):
    """会话聊天历史缓存"""
    def __init__(self) -> None:
        """从 self.session_cache_location 加载所有历史对话 json 文件"""
        self.cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        """获取消息"""
        return self.cache.message_get(conversation_id)

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        """设置消息"""
        self.cache.message_set(conversation_id, message_value)


class ChatSession:
    """聊天会话"""
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SETTINGS.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        """构建聊天完成消息"""
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": LLM_SETTINGS.system_prompt_role, "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        """构建聊天完成消息并计算令牌"""
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend._calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """
        这个函数用于构建会话消息
        用户提示应该总是被提供
        """
        messages = self.build_chat_completion_message(user_prompt)

        with logger.tag(f"session_{self.conversation_id}"):
            start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
            response: str = self.api_backend._try_create_chat_completion_or_embedding(  # noqa: SLF001
                *args,
                messages=messages,
                chat_completion=True,
                **kwargs,
            )
            end_time = datetime.now(pytz.timezone("Asia/Shanghai"))
            logger.log_object(
                {"user": user_prompt, "resp": response, "start": start_time, "end": end_time}, tag="debug_llm"
            )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        SessionChatHistoryCache().message_set(self.conversation_id, messages)
        return response

    def get_conversation_id(self) -> str:
        """获取会话 ID"""
        return self.conversation_id

    def display_history(self) -> None:
        """显示历史记录"""
        # TODO: 实现一个漂亮的历史消息显示格式
        pass


class APIBackend(ABC):
    """
    LLM API 后端的抽象基类
    支持自动重试、缓存和自动继续
    内部 api 调用应在子类中实现
    """

    def __init__(
        self,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ):
        self.dump_chat_cache = LLM_SETTINGS.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = LLM_SETTINGS.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            LLM_SETTINGS.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = (
            LLM_SETTINGS.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        )
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = LLM_SETTINGS.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        self.retry_wait_seconds = LLM_SETTINGS.retry_wait_seconds

    def build_chat_session(
        self,
        conversation_id: str | None = None,
        session_system_prompt: str | None = None,
    ) -> ChatSession:
        """
        conversation_id 是由 uuid.uuid4() 创建的 256 位字符串，也是
        每个对话在 session_cache_folder/ 下的文件名
        """
        return ChatSession(self, conversation_id, session_system_prompt)

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict[str, Any]]:
        """
        构建消息以避免实现几行冗余代码

        """
        if former_messages is None:
            former_messages = []
        # shrink multiple break 将递归删除多个换行符（超过 2 个）
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": LLM_SETTINGS.system_prompt_role,
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * LLM_SETTINGS.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def _build_log_messages(self, messages: list[dict[str, Any]]) -> str:
        """构建日志消息"""
        log_messages = ""
        for m in messages:
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}角色:{LogColors.END}"
                f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}内容:{LogColors.END} "
                f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
            )
        return log_messages

    def build_messages_and_create_chat_completion(  # type: ignore[no-untyped-def]
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        shrink_multiple_break: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        负责构建消息和记录消息

        TODO: 奇怪的是，这个函数在我们将嵌入和聊天完成分开之前被调用。

        参数
        ----------
        user_prompt : str
        system_prompt : str | None
        former_messages : list | None
        response_format : BaseModel | dict
            一个基于 pydantic 的 BaseModel 或一个字典
        **kwargs
        返回
        -------
        str
        """
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt,
            system_prompt,
            former_messages,
            shrink_multiple_break=shrink_multiple_break,
        )

        start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            *args,
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )
        end_time = datetime.now(pytz.timezone("Asia/Shanghai"))
        if isinstance(resp, list):
            raise ValueError("_try_create_chat_completion_or_embedding 的响应应该是一个字符串。")
        logger.log_object(
            {"system": system_prompt, "user": user_prompt, "resp": resp, "start": start_time, "end": end_time},
            tag="debug_llm",
        )
        return resp

    def create_embedding(self, input_content: str | list[str], *args, **kwargs) -> list[float] | list[list[float]]:  # type: ignore[no-untyped-def]
        """创建嵌入"""
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            input_content_list=input_content_list,
            embedding=True,
            *args,
            **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]  # type: ignore[return-value]
        return resp  # type: ignore[return-value]

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        """构建消息并计算令牌"""
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt, system_prompt, former_messages, shrink_multiple_break=shrink_multiple_break
        )
        return self._calculate_token_from_messages(messages)

    def _try_create_chat_completion_or_embedding(  # type: ignore[no-untyped-def]
        self,
        max_retry: int = 10,
        chat_completion: bool = False,
        embedding: bool = False,
        *args,
        **kwargs,
    ) -> str | list[list[float]]:
        """此函数用于在嵌入和聊天完成之间共享操作"""
        assert not (chat_completion and embedding), "chat_completion 和 embedding 不能同时为 True"
        max_retry = LLM_SETTINGS.max_retry if LLM_SETTINGS.max_retry is not None else max_retry
        timeout_count = 0
        violation_count = 0
        embedding_truncated = False  # 跟踪我们是否已经尝试过截断
        for i in range(max_retry):
            API_start_time = datetime.now()
            try:
                if embedding:
                    return self._create_embedding_with_cache(*args, **kwargs)
                if chat_completion:
                    return self._create_chat_completion_auto_continue(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if hasattr(e, "message") and (
                    "'messages' must contain the word 'json' in some form" in e.message
                    or "\\'messages\\' must contain the word \\'json\\' in some form" in e.message
                ):
                    kwargs["add_json_in_prompt"] = True

                too_long_error_message = hasattr(e, "message") and (
                    "maximum context length" in e.message or "input must have less than" in e.message
                )

                if embedding and too_long_error_message:
                    if not embedding_truncated:
                        # 处理嵌入文本过长错误 - 截断一次并重试
                        model_name = LLM_SETTINGS.embedding_model
                        logger.warning(f"模型 {model_name} 的嵌入文本过长，正在截断内容")

                        # 对内容列表应用截断并继续重试
                        original_content_list = kwargs.get("input_content_list", [])
                        kwargs["input_content_list"] = truncate_content_list(original_content_list, model_name)
                        embedding_truncated = True  # 标记我们已经尝试过截断
                        # 继续下一次迭代以使用截断的内容重试嵌入
                    else:
                        # 已经尝试过截断，引发错误并提供指导
                        raise RuntimeError(
                            f"即使在截断后嵌入也失败了。 "
                            f"请将 LLM_SETTINGS.embedding_max_length 设置为更小的值。"
                        ) from e
                else:
                    RD_Agent_TIMER_wrapper.api_fail_count += 1
                    RD_Agent_TIMER_wrapper.latest_api_fail_time = datetime.now(pytz.timezone("Asia/Shanghai"))

                    if (
                        openai_imported
                        and isinstance(e, litellm.BadRequestError)
                        and (
                            isinstance(e.__cause__, litellm.ContentPolicyViolationError)
                            or "由于提示触发了 Azure OpenAI 的内容管理策略，响应被过滤"
                            in str(e)
                        )
                    ):
                        violation_count += 1
                        if violation_count >= LLM_SETTINGS.violation_fail_limit:
                            logger.warning("检测到内容策略违规。")
                            raise PolicyError(e)

                    if (
                        openai_imported
                        and isinstance(e, openai.APITimeoutError)
                        or (
                            isinstance(e, openai.APIError)
                            and hasattr(e, "message")
                            and "您的资源已被临时阻止，因为我们检测到可能违反我们内容策略的行为。"
                            in e.message
                        )
                    ):
                        timeout_count += 1
                        if timeout_count >= LLM_SETTINGS.timeout_fail_limit:
                            logger.warning("超时错误，请检查您的网络连接。")
                            raise e

                    recommended_wait_seconds = self.retry_wait_seconds
                    if openai_imported and isinstance(e, openai.RateLimitError) and hasattr(e, "message"):
                        match = re.search(r"请在 (\d+) 秒后重试\.", e.message)
                        if match:
                            recommended_wait_seconds = int(match.group(1))
                    time.sleep(recommended_wait_seconds)
                    if RD_Agent_TIMER_wrapper.timer.started and not isinstance(e, json.decoder.JSONDecodeError):
                        RD_Agent_TIMER_wrapper.timer.add_duration(datetime.now() - API_start_time)
                logger.warning(str(e))
                logger.warning(f"第 {i+1} 次重试...")
        error_message = f"在 {max_retry} 次重试后创建聊天完成失败。"
        raise RuntimeError(error_message)

    def _add_json_in_prompt(self, messages: list[dict[str, Any]]) -> None:
        """
        如果 add_json_in_prompt 为 True，则在提示中添加与 json 相关的内容
        """
        for message in messages[::-1]:
            message["content"] = message["content"] + "\n请以 json 格式响应。"
            if message["role"] == LLM_SETTINGS.system_prompt_role:
                # 注意：假设 system_prompt 始终是第一条消息
                break

    def _create_chat_completion_auto_continue(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        chat_cache_prefix: str = "",
        seed: Optional[int] = None,
        json_target_type: Optional[str] = None,
        add_json_in_prompt: bool = False,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        调用聊天完成函数，如果 finish_reason 是 length，则自动继续对话。
        """

        if response_format is None and json_mode:
            response_format = {"type": "json_object"}

        # 0) 如果缓存命中则直接返回
        if seed is None and LLM_SETTINGS.use_auto_chat_cache_seed_gen:
            seed = LLM_CACHE_SEED_GEN.get_next_seed()
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json + f"<seed={seed}/>"
        )  # FIXME 这是一个 hack，以确保缓存表示轮次索引
        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(self._build_log_messages(messages), tag="llm_messages")
                    logger.info(f"{LogColors.CYAN}响应:{cache_result}{LogColors.END}", tag="llm_messages")
                return cache_result

        # 1) 获取完整响应
        all_response = ""
        new_messages = deepcopy(messages)
        # 循环以获取完整响应
        try_n = 6
        # 在重试循环之前，初始化标志
        json_added = False
        for _ in range(try_n):  # 对于一些长代码，3 次可能不足以进行推理模型
            if response_format == {"type": "json_object"} and add_json_in_prompt and not json_added:
                self._add_json_in_prompt(new_messages)
                json_added = True
            response, finish_reason = self._create_chat_completion_inner_function(
                messages=new_messages,
                response_format=response_format,
                **kwargs,
            )
            all_response += response
            if finish_reason is None or finish_reason != "length":
                break  # 我们现在得到了一个完整的响应。
            new_messages.append({"role": "assistant", "content": response})
        else:
            raise RuntimeError(f"在 {try_n} 次重试后未能继续对话。")

        # 2) 优化响应并返回
        if LLM_SETTINGS.reasoning_think_rm:
            # 策略 1：尝试匹配完整的 <think>...</think> 模式
            match = re.search(r"<think>(.*?)</think>(.*)", all_response, re.DOTALL)
            if match:
                _, all_response = match.groups()
            else:
                # 策略 2：如果没有完整匹配，尝试只匹配 </think>
                match = re.search(r"</think>(.*)", all_response, re.DOTALL)
                if match:
                    all_response = match.group(1)
                # 如果根本没有匹配，则保留原始内容

        # 3) 格式检查
        if response_format == {"type": "json_object"} or json_target_type:
            parser = JSONParser(add_json_in_prompt=add_json_in_prompt)
            all_response = parser.parse(all_response)
            if json_target_type:
                # deepseek 将进入此分支
                TypeAdapter(json_target_type).validate_json(all_response)

        if response_format is not None:
            if not isinstance(response_format, dict) and issubclass(response_format, BaseModel):
                # 如果初始化失败，可能会引发 TypeError
                response_format(**json.loads(all_response))
            elif response_format == {"type": "json_object"}:
                logger.info(f"使用 OpenAI 响应格式: {response_format}")
            else:
                logger.warning(f"未知的 response_format: {response_format}，跳过验证。")
        if self.dump_chat_cache:
            self.cache.chat_set(input_content_json, all_response)
        return all_response

    def _create_embedding_with_cache(
        self, input_content_list: list[str], *args: Any, **kwargs: Any
    ) -> list[list[float]]:
        """使用缓存创建嵌入"""
        content_to_embedding_dict = {}
        filtered_input_content_list = []
        if self.use_embedding_cache:
            for content in input_content_list:
                cache_result = self.cache.embedding_get(content)
                if cache_result is not None:
                    content_to_embedding_dict[content] = cache_result
                else:
                    filtered_input_content_list.append(content)
        else:
            filtered_input_content_list = input_content_list

        if len(filtered_input_content_list) > 0:
            resp = self._create_embedding_inner_function(input_content_list=filtered_input_content_list)
            for index, data in enumerate(resp):
                content_to_embedding_dict[filtered_input_content_list[index]] = data
            if self.dump_embedding_cache:
                self.cache.embedding_set(content_to_embedding_dict)
        return [content_to_embedding_dict[content] for content in input_content_list]  # type: ignore[misc]

    @abstractmethod
    def supports_response_schema(self) -> bool:
        """
        检查后端是否支持函数调用
        """
        raise NotImplementedError("子类必须实现此方法")

    @abstractmethod
    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        从消息中计算令牌数
        """
        raise NotImplementedError("子类必须实现此方法")

    @abstractmethod
    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        调用嵌入函数
        """
        raise NotImplementedError("子类必须实现此方法")

    @abstractmethod
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
        raise NotImplementedError("子类必须实现此方法")

    @property
    def chat_token_limit(self) -> int:
        """聊天令牌限制"""
        return LLM_SETTINGS.chat_token_limit
