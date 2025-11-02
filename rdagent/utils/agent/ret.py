"""
Agent（代理）的输出非常重要。

我们认为这部分的功能可以被共享。
这个模块定义了不同类型的 Agent 输出格式及其解析逻辑。
"""

import json
import re
from abc import abstractclassmethod
from pathlib import Path
from typing import Any

from rdagent.utils.agent.apply_patch import apply_patch_from_text
from rdagent.utils.agent.tpl import T


class AgentOut:
    """
    Agent 输出的基类。
    """
    json_mode: bool = False  # 获取输出时，是否需要 LLM 开启 JSON 模式。

    @abstractclassmethod
    def get_spec(cls, **context: Any) -> str:
        """
        获取一个描述此输出格式规范的字符串（通常用于提示 LLM）。
        这是一个抽象类方法，需要子类实现。
        """
        raise NotImplementedError("请实现 `get_spec` 方法")

    @classmethod
    def extract_output(cls, resp: str) -> Any:
        """
        从 LLM 的响应字符串中提取结构化的输出。
        这是一个类方法，默认行为是直接抛出响应，子类应重写此方法。
        """
        raise NotImplementedError("请实现 `extract_output` 方法")


class PythonAgentOut(AgentOut):
    """
    用于处理以 Python 代码块形式返回的输出。
    """
    @classmethod
    def get_spec(cls):
        """
        获取 Python 输出格式的规范说明。
        """
        return T(".tpl:PythonAgentOut").r()

    @classmethod
    def extract_output(cls, resp: str):
        """
        从响应中提取第一个 Python 代码块。
        """
        # 使用非贪婪模式 (.*?) 来只匹配第一个代码块
        match = re.search(r".*```[Pp]ython\n(.*?)\n```.*", resp, re.DOTALL)
        if match:
            code = match.group(1)
            # 移除可能存在的多余的 <code></code> 标签
            code = re.sub(r"</?code>", "", code, flags=re.IGNORECASE)
            return code
        # 如果没有找到匹配的代码块，则返回原始响应
        return resp


class MarkdownAgentOut(AgentOut):
    """
    用于处理以 Markdown 代码块形式返回的输出。
    """
    @classmethod
    def get_spec(cls):
        """
        获取 Markdown 输出格式的规范说明。
        """
        return T(".tpl:MarkdownOut").r()

    @classmethod
    def extract_output(cls, resp: str):
        """
        从响应中提取第一个 Markdown 代码块。
        """
        match = re.search(r".*````markdown\n(.*)\n````.*", resp, re.DOTALL)
        if match:
            content = match.group(1)
            return content
        return resp


class BatchEditOut(AgentOut):
    """
    用于处理以 JSON 格式返回的批量编辑操作。
    """
    json_mode: bool = True  # 要求 LLM 使用 JSON 模式

    @classmethod
    def get_spec(cls, with_del=True):
        """
        获取批量编辑（JSON 格式）的规范说明。
        `with_del` 参数可以控制规范中是否包含删除操作的说明。
        """
        return T(".tpl:BatchEditOut").r(with_del=with_del)

    @classmethod
    def extract_output(cls, resp: str):
        """
        将 JSON 字符串响应解析为 Python 对象。
        """
        return json.loads(resp)


class PythonBatchEditOut(AgentOut):
    """
    用于处理以多个命名代码块形式返回的批量 Python 文件编辑。
    """
    @classmethod
    def get_spec(cls, with_del=True):
        """
        获取 Python 批量编辑（代码块格式）的规范说明。
        """
        return T(".tpl:PythonBatchEditOut").r(with_del=with_del)

    @classmethod
    def extract_output(cls, resp: str):
        """
        从响应中提取所有命名的代码块，并返回一个文件名到代码内容的字典。
        例如：
        ```file1.py
        print("hello")
        ```
        会被解析为 {"file1.py": 'print("hello")'}
        """
        code_blocks = {}
        pattern = re.compile(r"```(.*?)\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(resp)

        for match in matches:
            file_name, code = match
            code_blocks[file_name.strip()] = code.strip()

        return code_blocks


class PythonBatchPatchOut(AgentOut):
    """
    用于处理以自定义“伪 diff”补丁格式返回的批量 Python 文件编辑。
    """
    @classmethod
    def get_spec(cls):
        """
        获取 Python 批量补丁格式的规范说明。
        """
        return T(".tpl:PythonBatchPatchOut").r()

    @classmethod
    def extract_output(cls, resp: str, prefix: Path | None = None) -> str:
        """
        从响应中提取并应用补丁。
        注意：这个方法会直接修改文件系统中的文件。
        """
        code_blocks = {}
        # 步骤 1: 使用正则表达式提取补丁文本
        patch_pattern = re.compile(r"(\*\*\* Begin Patch\s*(.*?)\s*\*\*\* End Patch)", re.DOTALL)
        matches = patch_pattern.findall(resp)
        for match in matches:
            # 步骤 2: 调用 apply_patch_from_text 来解析和应用补丁
            # inplace=False 表示该函数返回一个包含变更的字典，而不是直接写入文件
            code_blocks.update(apply_patch_from_text(match[0], inplace=False, prefix=prefix))

        return code_blocks
