# 从 __future__ 导入 annotations，以支持更灵活的类型提示
from __future__ import annotations

# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 Mapping 类型提示
from typing import Mapping

# 导入 numpy 用于数值计算
import numpy as np
# 导入 pandas 用于数据处理
import pandas as pd
# 导入 KMeans 用于聚类
from sklearn.cluster import KMeans
# 导入 cosine_similarity 用于计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
# 导入 normalize 用于数据归一化
from sklearn.preprocessing import normalize
# 导入 tqdm 用于显示进度条
from tqdm.auto import tqdm

# 导入 PDF 文档读取器
from rdagent.components.document_reader.document_reader import (
    load_and_process_pdfs_by_langchain,
)
# 导入因子实验加载器基类
from rdagent.components.loader.experiment_loader import FactorExperimentLoader
# 导入 RD_AGENT 的配置
from rdagent.core.conf import RD_AGENT_SETTINGS
# 导入多进程包装器
from rdagent.core.utils import multiprocessing_wrapper
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入大语言模型相关配置和工具
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
# 导入 Qlib 因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
# 导入基于 JSON 的因子实验加载器
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorExperimentLoaderFromDict,
)
# 导入模板工具
from rdagent.utils.agent.tpl import T


def classify_report_from_dict(
    report_dict: Mapping[str, str],
    vote_time: int = 1,
    substrings: tuple[str] = (),
) -> dict[str, dict[str, str]]:
    """
    使用大语言模型对研究报告进行分类，判断其是否包含量化因子内容。

    Parameters:
    - report_dict (Dict[str, str]): 报告字典，键为报告路径，值为报告内容字符串。
    - vote_time (int): 对每个报告进行多次投票以提高准确性。
    - substrings (tuple[str]): 用于预过滤的关键词（当前未使用）。

    Returns:
    - Dict[str, Dict[str, str]]: 分类结果字典，键为报告路径，值为 {"class": 0 或 1}。
    """
    res_dict = {}
    classify_prompt = T(".prompts:classify_system").r()

    for key, value in tqdm(report_dict.items()):
        if not key.endswith(".pdf"):
            continue
        file_name = key

        if isinstance(value, str):
            content = value
        else:
            logger.warning(f"输入格式不符合要求: {file_name}")
            res_dict[file_name] = {"class": 0}
            continue

        # 当前跳过了基于关键词的预过滤

        # 如果内容超过 token 限制，则进行截断
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=content,
                system_prompt=classify_prompt,
            )
            > APIBackend().chat_token_limit
        ):
            content = content[: -(APIBackend().chat_token_limit // 100)]

        # 多次调用 LLM 进行投票
        vote_list = []
        for _ in range(vote_time):
            user_prompt = content
            system_prompt = classify_prompt
            res = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            try:
                res = json.loads(res)
                vote_list.append(int(res["class"]))
            except json.JSONDecodeError:
                logger.warning(f"返回值无法解析: {file_name}")
                res_dict[file_name] = {"class": 0}

            # 如果已有多数票，则提前结束投票
            count_0 = vote_list.count(0)
            count_1 = vote_list.count(1)
            if max(count_0, count_1) > int(vote_time / 2):
                break

        # 少数服从多数
        result = 1 if count_1 > count_0 else 0
        res_dict[file_name] = {"class": result}

    return res_dict


def __extract_factors_name_and_desc_from_content(
    content: str,
) -> dict[str, dict[str, str]]:
    """
    从报告内容中提取因子的名称和描述。
    """
    session = APIBackend().build_chat_session(
        session_system_prompt=T(".prompts:extract_factors_system").r(),
    )

    extracted_factor_dict = {}
    current_user_prompt = content

    # 多轮对话，直到提取完所有因子
    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=True,
        )
        ret_dict = json.loads(extract_result_resp)
        factors = ret_dict["factors"]
        if len(factors) == 0:
            break
        for factor_name, factor_description in factors.items():
            extracted_factor_dict[factor_name] = factor_description
        current_user_prompt = T(".prompts:extract_factors_follow_user").r()

    return extracted_factor_dict


def __extract_factors_formulation_from_content(
    content: str,
    factor_dict: dict[str, str],
) -> dict[str, dict[str, str]]:
    """
    根据已提取的因子名称和描述，从报告内容中进一步提取因子的公式和变量。
    """
    factor_dict_df = pd.DataFrame(
        factor_dict.items(),
        columns=["factor_name", "factor_description"],
    )

    system_prompt = T(".prompts:extract_factor_formulation_system").r()
    current_user_prompt = T(".prompts:extract_factor_formulation_user").r(
        report_content=content,
        factor_dict=factor_dict_df.to_string(),
    )

    session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
    factor_to_formulation = {}

    # 多轮对话，直到为所有因子都提取到公式
    for _ in range(10):
        extract_result_resp = session.build_chat_completion(
            user_prompt=current_user_prompt,
            json_mode=True,
        )
        ret_dict = json.loads(extract_result_resp)
        for name, formulation_and_description in ret_dict.items():
            if name in factor_dict:
                factor_to_formulation[name] = formulation_and_description
        if len(factor_to_formulation) != len(factor_dict):
            # 如果有因子缺失，构建新的提示，要求继续提取
            remain_df = factor_dict_df[~factor_dict_df["factor_name"].isin(factor_to_formulation)]
            current_user_prompt = (
                "部分因子缺失。请检查以下因子及其描述并继续提取。\n"
                "==========================剩余因子==========================\n" + remain_df.to_string()
            )
        else:
            break

    return factor_to_formulation


def __extract_factor_and_formulation_from_one_report(
    content: str,
) -> dict[str, dict[str, str]]:
    """
    从单个报告中完整地提取因子信息（名称、描述、公式、变量）的流水线函数。
    """
    final_factor_dict_to_one_report = {}
    factor_dict = __extract_factors_name_and_desc_from_content(content)
    if len(factor_dict) != 0:
        factor_to_formulation = __extract_factors_formulation_from_content(
            content,
            factor_dict,
        )
    for factor_name in factor_dict:
        if (
            factor_name not in factor_to_formulation
            or "formulation" not in factor_to_formulation[factor_name]
            or "variables" not in factor_to_formulation[factor_name]
        ):
            continue

        final_factor_dict_to_one_report.setdefault(factor_name, {})
        final_factor_dict_to_one_report[factor_name]["description"] = factor_dict[factor_name]

        # 修正公式中可能存在的下划线 `_` 导致的 Markdown 格式问题
        formulation = factor_to_formulation[factor_name]["formulation"]
        if factor_name in formulation:
            target_factor_name = factor_name.replace("_", r"\_")
            formulation = formulation.replace(factor_name, target_factor_name)
        for variable in factor_to_formulation[factor_name]["variables"]:
            if variable in formulation:
                target_variable = variable.replace("_", r"\_")
                formulation = formulation.replace(variable, target_variable)

        final_factor_dict_to_one_report[factor_name]["formulation"] = formulation
        final_factor_dict_to_one_report[factor_name]["variables"] = factor_to_formulation[factor_name]["variables"]

    return final_factor_dict_to_one_report


def extract_factors_from_report_dict(
    report_dict: dict[str, str],
    useful_no_dict: dict[str, dict[str, str]],
    n_proc: int = 11,
) -> dict[str, dict[str, dict[str, str]]]:
    """
    从经过分类筛选后的“有用”报告中并行提取因子。
    """
    useful_report_dict = {}
    for key, value in useful_no_dict.items():
        if isinstance(value, dict):
            if int(value.get("class")) == 1:
                useful_report_dict[key] = report_dict[key]
        else:
            logger.warning(f"输入格式无效: {key}")

    file_name_list = list(useful_report_dict.keys())

    # 使用多进程并行处理每个报告
    final_report_factor_dict = {}
    factor_dict_list = multiprocessing_wrapper(
        [
            (__extract_factor_and_formulation_from_one_report, (useful_report_dict[file_name],))
            for file_name in file_name_list
        ],
        n=RD_AGENT_SETTINGS.multi_proc_n,
    )
    for index, file_name in enumerate(file_name_list):
        final_report_factor_dict[file_name] = factor_dict_list[index]
    logger.info(f"为 {len(final_report_factor_dict)} 份报告完成了因子提取")

    return final_report_factor_dict


def merge_file_to_factor_dict_to_factor_dict(
    file_to_factor_dict: dict[str, dict],
) -> dict:
    """
    将从多个文件中提取的因子字典合并成一个总的因子字典，并进行简单的去重。
    """
    factor_dict = {}
    for file_name in file_to_factor_dict:
        for factor_name in file_to_factor_dict[file_name]:
            factor_dict.setdefault(factor_name, [])
            factor_dict[factor_name].append(file_to_factor_dict[file_name][factor_name])

    # 简单的去重逻辑：如果同一个因子在多个报告中出现，选择公式最长的那个
    factor_dict_simple_deduplication = {}
    for factor_name in factor_dict:
        if len(factor_dict[factor_name]) > 1:
            factor_dict_simple_deduplication[factor_name] = max(
                factor_dict[factor_name],
                key=lambda x: len(x["formulation"]),
            )
        else:
            factor_dict_simple_deduplication[factor_name] = factor_dict[factor_name][0]
    return factor_dict_simple_deduplication


def __check_factor_dict_relevance(
    factor_df_string: str,
) -> dict[str, dict[str, str]]:
    """
    调用 LLM 判断一批因子的相关性。
    """
    extract_result_resp = APIBackend().build_messages_and_create_chat_completion(
        system_prompt=T(".prompts:factor_relevance_system").r(),
        user_prompt=factor_df_string,
        json_mode=True,
    )
    return json.loads(extract_result_resp)


def check_factor_relevance(
    factor_dict: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    并行地检查所有因子的相关性，并筛选出相关的因子。
    """
    factor_relevance_dict = {}

    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    # 分批并行处理
    while factor_df.shape[0] > 0:
        result_list = multiprocessing_wrapper(
            [
                (__check_factor_dict_relevance, (factor_df.iloc[i : i + 50, :].to_string(),))
                for i in range(0, factor_df.shape[0], 50)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for result in result_list:
            for factor_name, relevance in result.items():
                factor_relevance_dict[factor_name] = relevance

        factor_df = factor_df[~factor_df.index.isin(factor_relevance_dict)]

    # 筛选出被认为相关的因子
    filtered_factor_dict = {
        factor_name: factor_dict[factor_name]
        for factor_name in factor_dict
        if factor_relevance_dict[factor_name]["relevance"]
    }

    return factor_relevance_dict, filtered_factor_dict


def __check_factor_dict_viability_simulate_json_mode(
    factor_df_string: str,
) -> dict[str, dict[str, str]]:
    """
    调用 LLM 判断一批因子的可行性（是否可以被代码实现）。
    """
    extract_result_resp = APIBackend().build_messages_and_create_chat_completion(
        system_prompt=T(".prompts:factor_viability_system").r(),
        user_prompt=factor_df_string,
        json_mode=True,
    )
    return json.loads(extract_result_resp)


def check_factor_viability(
    factor_dict: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    并行地检查所有因子的可行性，并筛选出可行的因子。
    """
    factor_viability_dict = {}

    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]

    # 分批并行处理
    while factor_df.shape[0] > 0:
        result_list = multiprocessing_wrapper(
            [
                (__check_factor_dict_viability_simulate_json_mode, (factor_df.iloc[i : i + 50, :].to_string(),))
                for i in range(0, factor_df.shape[0], 50)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for result in result_list:
            for factor_name, viability in result.items():
                factor_viability_dict[factor_name] = viability

        factor_df = factor_df[~factor_df.index.isin(factor_viability_dict)]

    # 筛选出被认为可行的因子
    filtered_factor_dict = {
        factor_name: factor_dict[factor_name]
        for factor_name in factor_dict
        if factor_viability_dict[factor_name]["viability"]
    }

    return factor_viability_dict, filtered_factor_dict


def __check_factor_duplication_simulate_json_mode(
    factor_df: pd.DataFrame,
) -> list[list[str]]:
    """
    调用 LLM 检查一批因子中的重复项。
    """
    current_user_prompt = factor_df.to_string()

    # 如果输入过长，则进行切分
    working_list = [factor_df]
    final_list = []

    while len(working_list) > 0:
        current_df = working_list.pop(0)
        if (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=current_df.to_string(), system_prompt=T(".prompts:factor_duplicate_system").r()
            )
            > APIBackend().chat_token_limit
        ):
            working_list.append(current_df.iloc[: current_df.shape[0] // 2, :])
            working_list.append(current_df.iloc[current_df.shape[0] // 2 :, :])
        else:
            final_list.append(current_df)

    # 多轮对话提取重复组
    generated_duplicated_groups = []
    for current_df in final_list:
        current_factor_to_string = current_df.to_string()
        session = APIBackend().build_chat_session(
            session_system_prompt=T(".prompts:factor_duplicate_system").r(),
        )
        for _ in range(10):
            extract_result_resp = session.build_chat_completion(
                user_prompt=current_factor_to_string,
                json_mode=True,
            )
            ret_dict = json.loads(extract_result_resp)
            if len(ret_dict) == 0:
                return generated_duplicated_groups
            else:
                generated_duplicated_groups.extend(ret_dict)
                current_factor_to_string = """继续提取重复组。如果找不到更多重复组，请响应空字典。"""
    return generated_duplicated_groups


def __kmeans_embeddings(embeddings: np.ndarray, k: int = 20) -> list[list[str]]:
    """
    使用 K-Means 和余弦相似度对因子嵌入进行聚类。
    """
    x_normalized = normalize(embeddings)
    np.random.seed(42)

    kmeans = KMeans(
        n_clusters=k,
        init="random",
        max_iter=100,
        n_init=10,
        random_state=42,
    )

    # 自定义函数以使用余弦相似度找到最近的簇中心
    def find_closest_cluster_cosine_similarity(
        data: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        similarity = cosine_similarity(data, centroids)
        return np.argmax(similarity, axis=1)

    # 初始化簇中心
    rng = np.random.default_rng(seed=42)
    centroids = rng.choice(x_normalized, size=k, replace=False)

    # 迭代直到收敛或达到最大迭代次数
    for _ in range(kmeans.max_iter):
        closest_clusters = find_closest_cluster_cosine_similarity(x_normalized, centroids)
        new_centroids = np.array([x_normalized[closest_clusters == i].mean(axis=0) for i in range(k)])
        new_centroids = normalize(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    clusters = find_closest_cluster_cosine_similarity(x_normalized, centroids)
    cluster_to_index = {}
    for index, cluster in enumerate(clusters):
        cluster_to_index.setdefault(cluster, []).append(index)
    return sorted(cluster_to_index.values(), key=lambda x: len(x), reverse=True)


def __deduplicate_factor_dict(factor_dict: dict[str, dict[str, str]]) -> list[list[str]]:
    """
    对因子字典进行去重的核心逻辑。
    首先使用 embedding 和 K-Means 进行粗粒度分组，然后对每个组内的因子调用 LLM 进行精细去重。
    """
    if len(factor_dict) == 0:
        return []
    factor_df = pd.DataFrame(factor_dict).T
    factor_df.index.names = ["factor_name"]
    factor_names = sorted(factor_dict)

    # 将每个因子的所有信息拼接成一个字符串
    factor_name_to_full_str = {}
    for factor_name in factor_dict:
        description = factor_dict[factor_name]["description"]
        formulation = factor_dict[factor_name]["formulation"]
        variables = factor_dict[factor_name]["variables"]
        factor_name_to_full_str[factor_name] = f"""因子名称: {factor_name}\n因子描述: {description}\n因子公式: {formulation}\n因子变量: {variables}\n"""

    # 获取所有因子的 embedding
    full_str_list = [factor_name_to_full_str[factor_name] for factor_name in factor_names]
    embeddings = APIBackend.create_embedding(full_str_list)

    # 使用 K-Means 将因子分组，确保每组的大小不超过 LLM 的输入限制
    target_k = None
    if len(full_str_list) < RD_AGENT_SETTINGS.max_input_duplicate_factor_group:
        kmeans_index_group = [list(range(len(full_str_list)))]
        target_k = 1
    else:
        for k in range(len(full_str_list) // RD_AGENT_SETTINGS.max_input_duplicate_factor_group, RD_AGENT_SETTINGS.max_kmeans_group_number):
            kmeans_index_group = __kmeans_embeddings(embeddings=embeddings, k=k)
            if len(kmeans_index_group[0]) < RD_AGENT_SETTINGS.max_input_duplicate_factor_group:
                target_k = k
                logger.info(f"K-means 组数: {k}")
                break
    factor_name_groups = [[factor_names[index] for index in index_group] for index_group in kmeans_index_group]

    # 对每个组并行调用 LLM 进行去重
    result_list = multiprocessing_wrapper(
        [
            (__check_factor_duplication_simulate_json_mode, (factor_df.loc[factor_name_group, :],))
            for factor_name_group in factor_name_groups
        ],
        n=RD_AGENT_SETTINGS.multi_proc_n,
    )

    duplication_names_list = []
    for deduplication_factor_names_list in result_list:
        filter_factor_names = [factor_name for factor_name in set(deduplication_factor_names_list) if factor_name in factor_dict]
        if len(filter_factor_names) > 1:
            duplication_names_list.append(filter_factor_names)

    return duplication_names_list


def deduplicate_factors_by_llm(  # noqa: C901, PLR0912
    factor_dict: dict[str, dict[str, str]],
    factor_viability_dict: dict[str, dict[str, str]] | None = None,
) -> list[list[str]]:
    """
    使用 LLM 对因子字典进行去重的高级封装。
    支持多轮去重，并根据可行性判断选择要保留的因子。
    """
    final_duplication_names_list = []
    current_round_factor_dict = factor_dict

    # 多轮去重，直到没有大的重复组
    for _ in range(10):
        duplication_names_list = __deduplicate_factor_dict(current_round_factor_dict)
        new_round_names = []
        for duplication_names in duplication_names_list:
            if len(duplication_names) < RD_AGENT_SETTINGS.max_output_duplicate_factor_group:
                final_duplication_names_list.append(duplication_names)
            else:
                new_round_names.extend(duplication_names)
        if len(new_round_names) != 0:
            current_round_factor_dict = {factor_name: factor_dict[factor_name] for factor_name in new_round_names}
        else:
            break

    # 按重复组的大小排序
    final_duplication_names_list = sorted(final_duplication_names_list, key=lambda x: len(x), reverse=True)

    # 创建一个映射，将重复的因子映射到要保留的目标因子
    to_replace_dict = {}
    for duplication_names in duplication_names_list:
        if factor_viability_dict is not None:
            viability_list = [factor_viability_dict[name]["viability"] for name in duplication_names]
            if True not in viability_list:
                continue
            target_factor_name = duplication_names[viability_list.index(True)]
        else:
            target_factor_name = duplication_names[0]
        for duplication_factor_name in duplication_names:
            if duplication_factor_name == target_factor_name:
                continue
            to_replace_dict[duplication_factor_name] = target_factor_name

    # 构建最终去重后的因子字典
    llm_deduplicated_factor_dict = {}
    added_lower_name_set = set()
    for factor_name in factor_dict:
        if factor_name not in to_replace_dict and factor_name.lower() not in added_lower_name_set:
            if factor_viability_dict is not None and not factor_viability_dict[factor_name]["viability"]:
                continue
            added_lower_name_set.add(factor_name.lower())
            llm_deduplicated_factor_dict[factor_name] = factor_dict[factor_name]

    return llm_deduplicated_factor_dict, final_duplication_names_list


class FactorExperimentLoaderFromPDFfiles(FactorExperimentLoader):
    """
    从 PDF 文件（或包含 PDF 的文件夹）加载因子实验的加载器。
    这是一个端到端的流水线。
    """
    def load(self, file_or_folder_path: str) -> QlibFactorExperiment:
        # 1. 加载和处理 PDF
        with logger.tag("docs"):
            docs_dict = load_and_process_pdfs_by_langchain(file_or_folder_path)
            logger.log_object(docs_dict)

        # 2. 对报告进行分类
        selected_report_dict = classify_report_from_dict(report_dict=docs_dict, vote_time=1)

        # 3. 从有用的报告中提取因子
        with logger.tag("file_to_factor_result"):
            file_to_factor_result = extract_factors_from_report_dict(docs_dict, selected_report_dict)
            logger.log_object(file_to_factor_result)

        # 4. 合并并初步去重
        with logger.tag("factor_dict"):
            factor_dict = merge_file_to_factor_dict_to_factor_dict(file_to_factor_result)
            logger.log_object(factor_dict)

        # 5. 检查因子的可行性并筛选
        with logger.tag("filtered_factor_dict"):
            factor_viability, filtered_factor_dict = check_factor_viability(factor_dict)
            logger.log_object(filtered_factor_dict)

        # (可选) 6. 使用 LLM 进行更精细的去重
        # factor_dict, duplication_names_list = deduplicate_factors_by_llm(factor_dict, factor_viability)

        # 7. 使用最终的因子字典加载实验
        return FactorExperimentLoaderFromDict().load(filtered_factor_dict)
