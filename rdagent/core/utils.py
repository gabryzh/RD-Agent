# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

import functools
import importlib
import json
import multiprocessing as mp
import pickle
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, NoReturn, cast

from filelock import FileLock
from fuzzywuzzy import fuzz  # type: ignore[import-untyped]

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.oai.llm_conf import LLM_SETTINGS


class RDAgentException(Exception):  # noqa: N818
    """自定义异常类"""
    pass


class SingletonBaseClass:
    """
    单例基类。
    我们希望通过 `class A(SingletonBaseClass)` 的方式定义单例，
    而不是 `A(metaclass=SingletonMeta)`，因此需要这个类。
    """

    _instance_dict: ClassVar[dict] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # 由于很难统一使用args和kwargs的不同调用方式，我们严格要求在单例中使用kwargs
        if args:
            # TODO: 这个限制可以解决
            exception_message = "请仅在单例中使用kwargs以避免误解。"
            raise RDAgentException(exception_message)
        class_name = [(-1, f"{cls.__module__}.{cls.__name__}")]
        args_l = [(i, args[i]) for i in args]
        kwargs_l = sorted(kwargs.items())
        all_args = class_name + args_l + kwargs_l
        kwargs_hash = hash(tuple(all_args))
        if kwargs_hash not in cls._instance_dict:
            cls._instance_dict[kwargs_hash] = super().__new__(cls)
        return cls._instance_dict[kwargs_hash]

    def __reduce__(self) -> NoReturn:
        """
        注意：
        从pickle加载对象时，__new__方法不会收到初始化时使用的`kwargs`。
        这使得检索正确的单例对象变得困难。
        因此，我们使其不可pickle。
        """
        msg = f"{self.__class__.__name__} 的实例不能被pickle"
        raise pickle.PicklingError(msg)


def parse_json(response: str) -> Any:
    """解析JSON字符串"""
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass
    error_message = f"解析响应失败: {response}, 请报告此问题或帮助我们修复。"
    raise ValueError(error_message)


def similarity(text1: str, text2: str) -> int:
    """计算两个文本的相似度"""
    text1 = text1 if isinstance(text1, str) else ""
    text2 = text2 if isinstance(text2, str) else ""

    # 也许我们可以使用其他相似度算法，例如tfidf
    return cast("int", fuzz.ratio(text1, text2))


def import_class(class_path: str) -> Any:
    """
    动态导入一个类。
    :param class_path: 类的路径，例如 "scripts.factor_implementation.baselines.naive.one_shot.OneshotFactorGen"
    :return: 导入的类
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class CacheSeedGen:
    """
    一个全局种子生成器，用于生成一系列种子。
    这将支持 `use_auto_chat_cache_seed_gen` 功能。

    注意：
    - 这个种子专门用于缓存，与常规种子不同。
    - 如果缓存被移除，设置相同的种子将不会产生相同的QA轨迹。
    """

    def __init__(self) -> None:
        self.set_seed(LLM_SETTINGS.init_chat_cache_seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def get_next_seed(self) -> int:
        """生成下一个随机整数"""
        return random.randint(0, 10000)  # noqa: S311


LLM_CACHE_SEED_GEN = CacheSeedGen()


def _subprocess_wrapper(f: Callable, seed: int, args: list) -> Any:
    """
    一个函数包装器，以确保子进程具有固定的起始种子。
    """
    LLM_CACHE_SEED_GEN.set_seed(seed)
    return f(*args)


def multiprocessing_wrapper(func_calls: list[tuple[Callable, tuple]], n: int) -> list:
    """
    使用多进程调用func_calls中的函数。
    如果n=1，则不使用多进程。

    注意：
    我们与chat_cache_seed功能协作，
    确保即使有多个种子也能获得相同的种子轨迹。

    :param func_calls: 函数及其参数的列表
    :param n: 子进程的数量
    :return: 函数调用的结果列表
    """
    if n == 1 or max(1, min(n, len(func_calls))) == 1:
        return [f(*args) for f, args in func_calls]

    with mp.Pool(processes=max(1, min(n, len(func_calls)))) as pool:
        results = [
            pool.apply_async(_subprocess_wrapper, args=(f, LLM_CACHE_SEED_GEN.get_next_seed(), args))
            for f, args in func_calls
        ]
        return [result.get() for result in results]


def cache_with_pickle(hash_func: Callable, post_process_func: Callable | None = None, force: bool = False) -> Callable:
    """
    一个装饰器，用pickle缓存函数的返回值。
    缓存键由hash_func生成。如果hash_func返回None，则不使用缓存。
    缓存将存储在 `RD_AGENT_SETTINGS.pickle_cache_folder_path_str` 指定的文件夹中。
    post_process_func将使用原始参数和缓存结果调用，以处理缓存结果。

    :param hash_func: 生成缓存键的函数。
    :param post_process_func: 处理缓存结果的函数。
    :param force: 如果为True，则即使 `RD_AGENT_SETTINGS.cache_with_pickle` 为False，也使用缓存。
    """

    def cache_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def cache_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not RD_AGENT_SETTINGS.cache_with_pickle and not force:
                return func(*args, **kwargs)

            target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"{func.__module__}.{func.__name__}"
            target_folder.mkdir(parents=True, exist_ok=True)
            hash_key = hash_func(*args, **kwargs)

            if hash_key is None:
                return func(*args, **kwargs)

            cache_file = target_folder / f"{hash_key}.pkl"
            lock_file = target_folder / f"{hash_key}.lock"

            if cache_file.exists():
                with cache_file.open("rb") as f:
                    cached_res = pickle.load(f)
                return post_process_func(*args, cached_res=cached_res, **kwargs) if post_process_func else cached_res

            if RD_AGENT_SETTINGS.use_file_lock:
                with FileLock(lock_file):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            with cache_file.open("wb") as f:
                pickle.dump(result, f)

            return result

        return cache_wrapper

    return cache_decorator
