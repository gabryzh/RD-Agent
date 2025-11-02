import time
from collections.abc import Callable
from typing import Any, TypeVar

# 定义一个类型变量 ASpecificRet，用于表示被装饰函数的不确定返回类型
ASpecificRet = TypeVar("ASpecificRet")


def wait_retry(
    retry_n: int = 3, sleep_time: int = 1, transform_args_fn: Callable[[tuple, dict], tuple[tuple, dict]] | None = None
) -> Callable[[Callable[..., ASpecificRet]], Callable[..., ASpecificRet]]:
    """
    一个装饰器，用于在函数执行失败时等待并重试 `retry_n` 次。

    Example:
    >>> import time
    >>> @wait_retry(retry_n=2, sleep_time=1)
    ... def test_func():
    ...     global counter
    ...     counter += 1
    ...     if counter < 2:  # 修改以匹配示例输出
    ...         raise ValueError("Counter is less than 2")
    ...     return counter
    >>> counter = 0
    >>> test_func()
    Error: Counter is less than 2
    2
    >>> counter
    2

    >>> @wait_retry(retry_n=1, sleep_time=1)
    ... def failing_func():
    ...     raise ValueError("Always fails")
    >>> try:
    ...     failing_func()
    ... except ValueError as e:
    ...     print(f"Caught an exception: {e}")
    Error: Always fails
    Error: Always fails
    Caught an exception: Always fails

    Args:
        retry_n (int): 重试的次数。函数总共会尝试执行 `retry_n + 1` 次。
        sleep_time (int): 每次重试前等待的秒数。
        transform_args_fn (Callable | None): 一个可选函数，用于在每次重试前转换函数的参数。
                                            它接收一个包含 (args, kwargs) 的元组，并应返回一个新的 (args, kwargs) 元组。

    Returns:
        Callable: 返回一个包装了原函数的装饰器。
    """
    assert retry_n >= 0, "retry_n 应该是 0 或正整数"

    def decorator(f: Callable[..., ASpecificRet]) -> Callable[..., ASpecificRet]:
        def wrapper(*args: Any, **kwargs: Any) -> ASpecificRet:
            # 循环尝试执行函数，总共 retry_n + 1 次
            for i in range(retry_n + 1):
                try:
                    # 尝试执行原函数
                    return f(*args, **kwargs)
                except Exception as e:
                    # 如果发生异常，打印错误信息
                    print(f"错误: {e}")
                    # 如果已经是最后一次尝试，则直接重新抛出异常
                    if i == retry_n:
                        raise

                    # 等待指定的秒数
                    time.sleep(sleep_time)

                    # 如果提供了参数转换函数，则在下一次重试前调用它来更新参数
                    if transform_args_fn is not None:
                        args, kwargs = transform_args_fn(args, kwargs)

            # 这段代码理论上不会被执行，因为循环的最后一次要么成功返回，要么抛出异常。
            # 仅为通过 mypy 类型检查而保留。
            raise RuntimeError("重试逻辑出现意外错误")

        return wrapper

    return decorator
