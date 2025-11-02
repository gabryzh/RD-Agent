# 导入 json 模块
import json
# 从 typing 模块导入 List, Tuple
from typing import List, Tuple

# 导入模型实验相关的类
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
# 导入模型假设生成和转换相关的基类
from rdagent.components.proposal import ModelHypothesis2Experiment, ModelHypothesisGen
# 导入核心组件：假设、场景、追踪记录
from rdagent.core.proposal import Hypothesis, Scenario, Trace
# 导入 Qlib 模型实验类
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
# 导入 Qlib 量化场景类
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
# 导入模板工具
from rdagent.utils.agent.tpl import T

# 为通用的 Hypothesis 类创建一个别名，用于 Qlib 模型场景
QlibModelHypothesis = Hypothesis


class QlibModelHypothesisGen(ModelHypothesisGen):
    """
    Qlib 模型假设生成器。
    负责根据历史实验的追踪记录，生成用于调用大语言模型的上下文，以提出新的研究假设。
    """
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        """准备用于生成假设的上下文。"""
        # 渲染历史的假设和反馈
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

        # 找到最近一个被接受的（SOTA）假设和反馈
        sota_hypothesis_and_feedback = ""
        if len(trace.hist) == 0:
            sota_hypothesis_and_feedback = "由于是第一轮，没有 SOTA 假设和反馈可用。"
        else:
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][1].decision:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break
            else:
                sota_hypothesis_and_feedback = "由于先前的实验未被接受，没有 SOTA 假设和反馈可用。"

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            # RAG 提示：提供模型设计和超参数调整的建议
            "RAG": "1. 在量化金融中，市场数据可能是时间序列，GRU 模型/LSTM 模型适用于它们。目前不要生成 GNN 模型。\n2. 训练数据包含不到100万个训练样本和约25万个验证样本。请相应地设计超参数并控制模型大小。这对训练结果有重大影响。如果您认为以前的模型本身很好，但训练超参数或模型超参数不是最优的，您可以返回相同的模型并调整这些参数。",
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:model_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        """将大语言模型的响应字符串转换为 Hypothesis 对象。"""
        response_dict = json.loads(response)
        hypothesis = QlibModelHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    """
    将 Qlib 模型假设转换为可执行的实验。
    负责根据给定的假设，生成用于调用大语言模型的上下文，以设计具体的实验任务（即模型）。
    """
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        """准备用于将假设转换为实验的上下文。"""
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="model")
        else:
            scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = T("scenarios.qlib.prompts:model_experiment_output_format").r()

        last_experiment = None
        last_feedback = None
        sota_experiment = None
        sota_feedback = None

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "由于这是第一轮，没有先前的假设和反馈可用。"
        else:
            # 筛选出仅与模型相关的历史记录
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "model":
                    if last_experiment is None:
                        last_experiment = trace.hist[i][0]
                        last_feedback = trace.hist[i][1]
                    if trace.hist[i][1].decision is True and sota_experiment is None:
                        sota_experiment = trace.hist[i][0]
                        sota_feedback = trace.hist[i][1]
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "没有可用的先前假设和反馈。"

        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=last_experiment, feedback=last_feedback
            )
            if last_experiment is not None
            else "由于这是第一轮，没有先前的假设和反馈可用。"
        )

        sota_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                experiment=sota_experiment, feedback=sota_feedback
            )
            if sota_experiment is not None
            else "由于先前的实验未被接受，没有 SOTA 假设和反馈可用。"
        )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": "注意，训练数据包含不到100万个训练样本和约25万个验证样本。请相应地设计超参数并控制模型大小。这对训练结果有重大影响。如果您认为以前的模型本身很好，但训练超参数或模型超参数不是最优的，您可以返回相同的模型并调整这些参数。",
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        """将大语言模型的响应字符串转换为 ModelExperiment 对象。"""
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            # 从响应中解析出模型的具体任务
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            training_hyperparameters = response_dict[model_name]["training_hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    training_hyperparameters=training_hyperparameters,
                    model_type=model_type,
                )
            )
        # 创建 QlibModelExperiment 实例
        exp = QlibModelExperiment(tasks, hypothesis=hypothesis)
        # 将历史上的模型实验作为基线实验
        exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
        return exp
