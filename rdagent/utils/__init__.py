"""
这是一些通用的工具函数。
它不与特定的场景或框架绑定（因此不放在 rdagent.core.utils 中）。
"""

# TODO: 将 `rdagent.core.utils` 中的通用工具合并到此文件夹中
# TODO: 将来将此模块中的工具函数拆分到不同的模块中。

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Union

import regex  # type: ignore[import-untyped]  # 忽略 regex 模块的类型提示问题

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.utils.agent.tpl import T

# 所有正则表达式操作的默认超时时间（秒）
REGEX_TIMEOUT = 120.0


def get_module_by_module_path(module_path: Union[str, ModuleType]) -> ModuleType:
    """
    从类似 a/b/c/d.py 或 a.b.c.d 的路径加载模块。

    :param module_path: 模块路径字符串或模块对象。
    :return: 加载的模块对象。
    :raises: ModuleNotFoundError: 如果找不到或无法加载模块。
    """
    if module_path is None:
        raise ModuleNotFoundError("传入的 module_path 参数为 None")

    if isinstance(module_path, ModuleType):
        # 如果传入的已经是模块对象，则直接返回
        module = module_path
    else:
        if module_path.endswith(".py"):
            # 如果路径以 .py 结尾，则从文件加载
            # 将文件路径转换为合法的模块名
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            # 从文件位置创建模块规范
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            if module_spec is None:
                raise ModuleNotFoundError(f"在 {module_path} 找不到模块")
            module = importlib.util.module_from_spec(module_spec)
            # 将模块添加到 sys.modules 中，以便后续导入可以找到它
            sys.modules[module_name] = module
            if module_spec.loader is not None:
                # 执行模块加载
                module_spec.loader.exec_module(module)
            else:
                raise ModuleNotFoundError(f"无法加载模块 {module_path}")
        else:
            # 否则，作为普通的模块路径导入
            module = importlib.import_module(module_path)
    return module


def convert2bool(value: Union[str, bool]) -> bool:
    """
    动机：LLM（大语言模型）的返回值不稳定。此函数尝试将输入值转换为布尔值。
    """
    # TODO: 如果有更多类似的函数，我们可以构建一个库，用于将不稳定的 LLM 响应转换为稳定的结果。
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ["true", "yes", "ok"]:
            return True
        if v in ["false", "no"]:
            return False
        raise ValueError(f"无法将 '{value}' 转换为布尔值")
    elif isinstance(value, bool):
        return value
    else:
        raise ValueError(f"未知的值类型 {value}，无法转换为布尔值")


def try_regex_sub(pattern: str, text: str, replace_with: str = "", flags: int = 0) -> str:
    """
    尝试对文本字符串执行正则表达式替换操作。
    包含超时和异常处理。
    """
    try:
        # 使用 regex 库进行替换，并设置超时
        text = regex.sub(pattern, replace_with, text, timeout=REGEX_TIMEOUT, flags=flags)
    except TimeoutError:
        logger.warning(f"正则表达式 '{pattern}' 在 {REGEX_TIMEOUT} 秒后超时；已跳过。")
    except Exception as e:
        logger.warning(f"正则表达式 '{pattern}' 引发错误：{e}；已跳过。")
    return text


def filter_with_time_limit(regex_patterns: Union[str, list[str]], text: str) -> str:
    """
    使用一个或多个正则表达式模式过滤 `text`，并为每次替换设置超时。
    如果 `regex_patterns` 是一个列表，则按顺序应用；如果是一个字符串，则只应用该模式。
    """
    if not isinstance(regex_patterns, list):
        regex_patterns = [regex_patterns]
    for pattern in regex_patterns:
        text = try_regex_sub(pattern, text)
    return text


def filter_redundant_text(stdout: str) -> str:
    """
    使用基于正则表达式的修剪，从标准输出中过滤掉进度条和其他冗余模式。
    """
    from rdagent.oai.llm_utils import APIBackend  # 避免循环导入

    # 编译一个匹配常见进度条模式的正则表达式
    progress_bar_pattern = r"""(
        \d+/\d+\s+[━]+\s+\d+s?\s+\d+ms/step.*?\u0008+ |  # 例如: "10/100 ━━━━━━ 3s 50ms/step"
        \d+/\d+\s+[━]+\s+\d+s?\s+\d+ms/step |            # 例如: "10/100 ━━━━━━ 3s 50ms/step" (无退格)
        \d+/\d+\s+[━]+\s+\d+s?\s+\d+ms/step.* |           # 例如: 部分行
        \d+/\d+\s+[━]+.*?\u0008+ |                       # 例如: 带退格的
        \d+/\d+\s+[━]+.* |                                # 例如: 部分进度条
        [ ]*\u0008+ |                                     # 零散的退格符
        \d+%\|[█▏▎▍▌▋▊▉]+\s+\|\s+\d+/\d+\s+\[\d{2}:\d{2}<\d{2}:\d{2},\s+\d+\.\d+it/s\] |  # tqdm 风格
        \d+%\|[█]+\|\s+\d+/\d+\s+\[\d{2}:\d{2}<\d{2}:\d{2},\s*\d+\.\d+it/s\]
    )"""

    # 过滤掉 ANSI 转义序列（通常用于颜色和格式化）
    filtered_stdout = try_regex_sub(r"\x1B\[[0-?]*[ -/]*[@-~]", stdout)
    # 使用上面定义的模式过滤掉进度条
    filtered_stdout = try_regex_sub(progress_bar_pattern, filtered_stdout, flags=regex.VERBOSE)

    # 合并多余的空行和空格
    filtered_stdout = try_regex_sub(r"\s*\n\s*", filtered_stdout, replace_with="\n")

    # 删除重复的行
    lines_to_count: dict[str, int] = {}
    filtered_stdout_lines = filtered_stdout.splitlines()
    for line in filtered_stdout_lines:
        lines_to_count[line] = lines_to_count.get(line, 0) + 1
    # 只保留出现次数不多的行，以过滤掉重复的日志或状态更新
    filtered_stdout = "\n".join(
        [line for line in filtered_stdout_lines if lines_to_count[line] <= max(len(filtered_stdout_lines) // 10, 10)]
    )

    def _shrink_stdout_once(stdout: str) -> str:
        """单次缩减 stdout 字符串，取其头部和尾部。"""
        head = stdout[: int(APIBackend().chat_token_limit * 0.3)]
        tail = stdout[-int(APIBackend().chat_token_limit * 0.3) :]
        return head + tail

    # 迭代式地向 LLM 请求额外的过滤模式（最多3轮）
    for _ in range(3):
        truncated_stdout = filtered_stdout
        system_prompt = T(".prompts:filter_redundant_text.system").r()

        # 尝试缩减 stdout，使其令牌数可管理
        for __ in range(10):
            try:
                user_prompt = T(".prompts:filter_redundant_text.user").r(stdout=truncated_stdout)
                stdout_token_size = APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                if stdout_token_size < APIBackend().chat_token_limit * 0.1:
                    # 如果令牌数已经很小，直接返回截断后的输出
                    return truncated_stdout
                elif stdout_token_size > APIBackend().chat_token_limit * 0.6:
                    # 如果令牌数过大，则进行缩减
                    truncated_stdout = _shrink_stdout_once(truncated_stdout)
                else:
                    # 令牌数在可接受范围内，跳出循环
                    break
            except ValueError as e:
                # 处理 `build_messages_and_calculate_token` 可能引发的正则表达式错误
                logger.warning(f"因错误缩减: {e}")
                truncated_stdout = _shrink_stdout_once(truncated_stdout)

        try:
            # 请求 LLM 提供过滤建议
            response = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=dict,
                )
            )
        except Exception as e:
            logger.error(f"LLM 过滤请求失败: {e}")
            break

        needs_sub = response.get("needs_sub", True)
        regex_patterns = response.get("regex_patterns", [])

        try:
            # 应用 LLM 建议的正则表达式模式
            new_filtered = filter_with_time_limit(regex_patterns, truncated_stdout)
        except Exception as e:
            logger.error(f"应用 LLM 建议的模式时出错: {e}")
            break

        if not needs_sub:
            # 如果 LLM 认为不需要进一步替换，则返回结果
            return new_filtered

        # 合并多余的空行和空格，准备下一轮迭代
        filtered_stdout = try_regex_sub(r"\s*\n\s*", new_filtered, replace_with="\n")

    return filtered_stdout


def remove_path_info_from_str(base_path: Path, target_string: str) -> str:
    """
    从目标字符串中删除绝对路径信息。
    """
    target_string = re.sub(str(base_path), "...", target_string)
    target_string = re.sub(str(base_path.absolute()), "...", target_string)
    return target_string


def md5_hash(input_string: str) -> str:
    """
    计算输入字符串的 MD5 哈希值。
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)  # usedforsecurity=False 用于非安全场景
    input_bytes = input_string.encode("utf-8")
    hash_md5.update(input_bytes)
    return hash_md5.hexdigest()
