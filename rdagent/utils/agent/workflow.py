import json
from typing import Any, Callable, Type, TypeVar, Union, cast

from rdagent.core.exception import FormatError
from rdagent.log import rdagent_logger as logger

# 定义一个类型变量 T，用于泛型编程
T = TypeVar("T")


def build_cls_from_json_with_retry(
    cls: Type[T],
    system_prompt: str,
    user_prompt: str,
    retry_n: int = 5,
    init_kwargs_update_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    **kwargs: dict,
) -> T:
    """
    通过调用 LLM API 获取 JSON 响应，并使用该 JSON 数据实例化一个类。
    该函数包含重试逻辑，以处理 LLM 可能返回的不规范或错误的 JSON。

    Parameters
    ----------
    cls : Type[T]
        需要实例化的目标类的类型。
    system_prompt : str
        提供给 LLM 的系统提示，用于设定上下文。
    user_prompt : str
        用户提供的提示，用于指导 LLM 生成响应。
    retry_n : int
        在失败情况下进行重试的最大次数。
    init_kwargs_update_func : Union[Callable[[dict], dict], None], optional
        一个可选的回调函数。它接收从 JSON 响应解析出的字典作为输入，
        并返回更新后的字典。这可以用于在实例化类之前对数据进行预处理或修正。
        默认为 None。
    **kwargs
        传递给 API 调用的额外关键字参数。

    Returns
    -------
    T
        一个根据 LLM 响应数据创建的指定类的实例。

    Raises
    ------
    FormatError
        如果在达到最大重试次数后，仍然无法生成满足要求的 JSON 响应。
    """
    from rdagent.oai.llm_utils import APIBackend  # 局部导入以避免循环导入问题

    for i in range(retry_n):
        # 目前，它主要处理由类初始化引起的异常
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,  # 强制要求 LLM 返回 JSON 格式
            **kwargs  # type: ignore[arg-type]
        )
        try:
            # 尝试解析 JSON 响应
            resp_dict = json.loads(resp)
            # 如果提供了更新函数，则使用它来处理解析后的字典
            if init_kwargs_update_func:
                resp_dict = init_kwargs_update_func(resp_dict)
            # 使用处理后的字典作为关键字参数来实例化类
            return cls(**resp_dict)
        except Exception as e:
            # 如果解析或实例化失败，记录警告并准备重试
            logger.warning(f"尝试 {i + 1}: 上一次尝试因以下错误而失败: {e}")
            # 将上一次的错误信息附加到用户提示中，以便 LLM 在下一次尝试时进行修正
            user_prompt = user_prompt + f"\n\n尝试 {i + 1}: 上一次尝试因以下错误而失败: {e}"

    # 如果所有重试都失败，则抛出格式错误异常
    raise FormatError("无法生成满足指定要求的 JSON 响应。")
