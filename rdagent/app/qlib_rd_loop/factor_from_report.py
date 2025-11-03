import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import LoopMeta


def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    """
    根据因子结果和报告内容生成假设。

    参数:
        factor_result (dict): 因子分析的结果。
        report_content (str): 报告的内容。

    返回:
        str: 生成的假设。
    """
    system_prompt = T(".prompts:hypothesis_generation.system").r()
    user_prompt = T(".prompts:hypothesis_generation.user").r(
        factor_descriptions=json.dumps(factor_result), report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=Dict[str, str],
    )

    response_json = json.loads(response)

    return Hypothesis(
        hypothesis=response_json.get("hypothesis", "未提供假设"),
        reason=response_json.get("reason", "未提供原因"),
        concise_reason=response_json.get("concise_reason", "未提供简明原因"),
        concise_observation=response_json.get("concise_observation", "未提供简明观察"),
        concise_justification=response_json.get("concise_justification", "未提供简明理由"),
        concise_knowledge=response_json.get("concise_knowledge", "未提供简明知识"),
    )


def extract_hypothesis_and_exp_from_reports(report_file_path: str) -> QlibFactorExperiment | None:
    """
    从报告文件中提取假设和实验细节。

    参数:
        report_file_path (str): 报告文件的路径。

    返回:
        QlibFactorExperiment: 包含提取细节的QlibFactorExperiment实例。
        None: 如果在报告中未找到有效的实验。
    """
    exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
    if exp is None or exp.sub_tasks == []:
        return None

    pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(pdf_screenshot, tag="加载PDF截图")

    docs_dict = load_and_process_pdfs_by_langchain(report_file_path)

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources,
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = generate_hypothesis(factor_result, report_content)
    exp.hypothesis = hypothesis
    return exp


class FactorReportLoop(FactorRDLoop, metaclass=LoopMeta):
    """从报告中提取因子的研发循环"""
    def __init__(self, report_folder: str = None):
        """
        初始化FactorReportLoop。

        参数:
            report_folder (str, optional): 包含报告PDF文件的文件夹。报告将从此文件夹加载。
        """
        super().__init__(PROP_SETTING=FACTOR_FROM_REPORT_PROP_SETTING)
        if report_folder is None:
            self.judge_pdf_data_items = json.load(
                open(FACTOR_FROM_REPORT_PROP_SETTING.report_result_json_file_path, "r")
            )
        else:
            self.judge_pdf_data_items = [i for i in Path(report_folder).rglob("*.pdf")]

        self.loop_n = min(len(self.judge_pdf_data_items), FACTOR_FROM_REPORT_PROP_SETTING.report_limit)
        self.shift_report = (
            0  # 某些报告不包含可行的因子，因此我们跳过其中一些以避免无限循环
        )

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """
        直接从报告生成实验。

        参数:
            prev_out (dict[str, Any]): 上一步的输出。

        返回:
            QlibFactorExperiment: 生成的实验。
        """
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                report_file_path = self.judge_pdf_data_items[self.loop_idx + self.shift_report]
                logger.info(f"正在处理第 {self.loop_idx} 个报告: {report_file_path}")
                exp = extract_hypothesis_and_exp_from_reports(str(report_file_path))
                if exp is None:
                    self.shift_report += 1
                    self.loop_n -= 1
                    if self.loop_n < 0:  # 注意: 在每一步，我们首先 self.loop_n -= 1。
                        raise self.LoopTerminationError("达到停止标准并停止循环")
                    continue
                exp.based_experiments = [QlibFactorExperiment(sub_tasks=[], hypothesis=exp.hypothesis)] + [
                    t[0] for t in self.trace.hist if t[1]
                ]
                exp.sub_workspace_list = exp.sub_workspace_list[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                exp.sub_tasks = exp.sub_tasks[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                logger.log_object(exp.hypothesis, tag="假设生成")
                logger.log_object(exp.sub_tasks, tag="实验生成")
                return exp
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """
        编码步骤。

        参数:
            prev_out (dict[str, Any]): 上一步的输出。

        返回:
            QlibFactorExperiment: 编码后的实验。
        """
        exp = self.coder.develop(prev_out["direct_exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="编码器结果")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """
        反馈步骤。

        参数:
            prev_out (dict[str, Any]): 上一步的输出。
        """
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="反馈")
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="反馈")
            self.trace.hist.append((prev_out["running"], feedback))


def main(report_folder=None, path=None, all_duration=None, checkout=True):
    """
    金融科技因子的自动研发演进循环（因子从金融报告中提取）。

    参数:
        report_folder (str, optional): 包含报告PDF文件的文件夹。报告将从此文件夹加载。
        path (str, optional): 用于加载会话的路径。如果提供，将加载会话。
        all_duration (str, optional): 总运行时间。
        checkout (bool, optional): 是否检出。
    """
    if path is None and report_folder is None:
        model_loop = FactorReportLoop()
    elif path is not None:
        model_loop = FactorReportLoop.load(path, checkout=checkout)
    else:
        model_loop = FactorReportLoop(report_folder=report_folder)

    asyncio.run(model_loop.run(all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
