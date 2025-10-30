from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.components.document_reader.document_reader import (
    load_and_process_pdfs_by_langchain,
)
from rdagent.components.loader.task_loader import ModelTaskLoader
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


def extract_model_from_doc(doc_content: str) -> dict:
    """
    从文档内容中提取模型信息。

    Args:
        doc_content (str): 文档的文本内容。

    Returns:
        dict: 提取的模型信息字典，格式为 {model_name: {description, formulation, variables}}。
    """
    # 初始化聊天会话，并设置 system prompt
    session = APIBackend().build_chat_session(
        session_system_prompt=T(".prompts:extract_model_formulation_system").r(),
    )
    current_user_prompt = doc_content

    model_dict = {}

    # 最多重试10次以从文档中提取模型信息
    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=False,
        )
        # 从响应中用正则表达式提取 json 字符串
        re_search_res = re.search(r"```json(.*)```", extract_result_resp, re.S)
        ret_json_str = re_search_res.group(1) if re_search_res is not None else ""

        # 尝试解析json
        try:
            ret_dict = json.loads(ret_json_str)
            parse_success = isinstance(ret_dict, dict)
        except json.JSONDecodeError:
            parse_success = False

        # 如果解析失败，则要求 LLM 重试
        if not parse_success:
            current_user_prompt = "您的响应没有遵循指令，可能是json格式错误。请重试。"
        else:
            # 将成功提取的模型信息添加到结果字典中
            for name, formulation_and_description in ret_dict.items():
                if name not in model_dict:
                    model_dict[name] = formulation_and_description

            # 如果没有提取到任何模型，也要求重试
            if len(model_dict) == 0:
                current_user_prompt = "没有提取到模型。请重试。"
            else:
                # 成功提取，跳出循环
                break

    logger.info(f"已经完成 {len(model_dict)} 个模型的提取")
    return model_dict


def merge_file_to_model_dict_to_model_dict(
    file_to_model_dict: dict[str, dict],
) -> dict:
    """
    合并从多个文件中提取的模型字典。
    处理可能从不同文件中提取到相同模型的情况。

    Args:
        file_to_model_dict (dict[str, dict]): 格式为 {文件名: {模型名字: 模型信息}} 的字典。

    Returns:
        dict: 合并和去重后的模型字典。
    """
    model_dict = {}
    # 将所有模型按名称分组
    for file_name in file_to_model_dict:
        for model_name, model_info in file_to_model_dict[file_name].items():
            model_dict.setdefault(model_name, []).append(model_info)

    # 简单去重：对于同一个模型，选择公式（formulation）最长的那个版本
    model_dict_simple_deduplication = {}
    for model_name, model_infos in model_dict.items():
        if len(model_infos) > 1:
            model_dict_simple_deduplication[model_name] = max(
                model_infos,
                key=lambda x: len(x.get("formulation", "")),
            )
        else:
            model_dict_simple_deduplication[model_name] = model_infos[0]

    return model_dict_simple_deduplication


def extract_model_from_docs(docs_dict: dict[str, str]) -> dict:
    """
    从多个文档中提取模型信息。

    Args:
        docs_dict (dict[str, str]): 包含文档名称和内容的字典。

    Returns:
        dict: 格式为 {文件名: {模型名字: 模型信息}} 的字典。
    """
    model_dict = {}
    for doc_name, doc_content in docs_dict.items():
        model_dict[doc_name] = extract_model_from_doc(doc_content)
    return model_dict


class ModelExperimentLoaderFromDict(ModelTaskLoader):
    """从字典加载模型实验。"""
    def load(self, model_dict: dict) -> QlibModelExperiment:
        """从字典加载数据。"""
        task_l = []
        for model_name, model_data in model_dict.items():
            task = ModelTask(
                name=model_name,
                description=model_data.get("description", ""),
                formulation=model_data.get("formulation", ""),
                architecture=model_data.get("architecture", ""),
                variables=model_data.get("variables", {}),
                hyperparameters=model_data.get("hyperparameters", {}),
                training_hyperparameters=model_data.get("training_hyperparameters", {}),
                model_type=model_data.get("model_type", ""),
            )
            task_l.append(task)
        return QlibModelExperiment(sub_tasks=task_l)


class ModelExperimentLoaderFromPDFfiles(ModelTaskLoader):
    """从PDF文件加载模型实验。"""
    @wait_retry(retry_n=5)  # 加载失败时最多重试5次
    def load(self, file_or_folder_path: str) -> QlibModelExperiment:
        """
        从单个PDF文件或包含PDF的文件夹中加载模型实验。
        """
        # 1. 加载PDF内容
        docs_dict = load_and_process_pdfs_by_langchain(file_or_folder_path)
        # 2. 从文档内容中提取模型信息
        model_dict = extract_model_from_docs(docs_dict)
        # 3. 合并和去重
        model_dict = merge_file_to_model_dict_to_model_dict(model_dict)
        # 4. 从最终的字典中创建实验对象
        return ModelExperimentLoaderFromDict().load(model_dict)
