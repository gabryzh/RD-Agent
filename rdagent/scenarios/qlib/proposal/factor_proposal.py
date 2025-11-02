# 导入 json 模块
import json
# 从 typing 模块导入 List, Tuple
from typing import List, Tuple

# 导入因子实验相关的类
from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
# 导入假设生成和转换相关的基类
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
# 导入核心组件：假设、场景、追踪记录
from rdagent.core.proposal import Hypothesis, Scenario, Trace
# 导入 Qlib 因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
# 导入 Qlib 模型实验类
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
# 导入 Qlib 量化场景类
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
# 导入模板工具
from rdagent.utils.agent.tpl import T

# 为通用的 Hypothesis 类创建一个别名，用于 Qlib 因子场景
QlibFactorHypothesis = Hypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
    """
    Qlib 因子假设生成器。
    负责根据历史实验的追踪记录，生成用于调用大语言模型的上下文，以提出新的研究假设。
    """
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        """准备用于生成假设的上下文。"""
        # 如果有历史记录，则渲染历史的假设和反馈
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "由于这是第一轮，没有先前的假设和反馈可用。"
        )
        # 渲染最近一次的假设和反馈
        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
            )
            if len(trace.hist) > 0
            else "由于这是第一轮，没有先前的假设和反馈可用。"
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            # RAG (Retrieval-Augmented Generation) 提示：根据实验轮次给出不同策略建议
            "RAG": (
                "首先尝试从不同角度进行最简单、最快速的因子实验。"
                if len(trace.hist) < 15
                else "现在，你需要尝试能够实现高 IC 的因子（例如，基于机器学习的因子）。"
            ),
            "hypothesis_output_format": T("scenarios.qlib.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:factor_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        """将大语言模型的响应字符串转换为 Hypothesis 对象。"""
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    """
    将 Qlib 因子假设转换为可执行的实验。
    负责根据给定的假设，生成用于调用大语言模型的上下文，以设计具体的实验任务（即因子）。
    """
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        """准备用于将假设转换为实验的上下文。"""
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="factor")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        experiment_output_format = T("scenarios.qlib.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "由于这是第一轮，没有先前的假设和反馈可用。"
        else:
            # 筛选出仅与因子相关的历史记录
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "factor":
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "没有可用的先前假设和反馈。"

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        """将大语言模型的响应字符串转换为 FactorExperiment 对象。"""
        response_dict = json.loads(response)
        tasks = []

        # 从响应中解析出每个因子的具体任务
        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        # 创建 QlibFactorExperiment 实例
        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        # 将历史上的因子实验作为基线实验
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        # 筛选出本次实验中不与历史重复的新因子任务
        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        # 更新实验的任务列表为去重后的列表
        exp.tasks = unique_tasks
        return exp
