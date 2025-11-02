# 导入 json 模块
import json
# 导入 random 模块
import random
# 从 typing 模块导入 Tuple
from typing import Tuple

# 导入量化提案的配置
from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
# 导入因子和模型假设生成的基类
from rdagent.components.proposal import FactorAndModelHypothesisGen
# 导入核心组件：假设、场景、追踪记录
from rdagent.core.proposal import Hypothesis, Scenario, Trace
# 导入大语言模型 API 后端
from rdagent.oai.llm_utils import APIBackend
# 导入 Bandit 算法相关的类和函数
from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    extract_metrics_from_experiment,
)
# 导入模板工具
from rdagent.utils.agent.tpl import T


class QuantTrace(Trace):
    """
    量化研究的追踪记录类，继承自通用的 Trace 类。
    它额外包含了一个 Bandit 环境控制器。
    """
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        # 使用默认权重初始化控制器
        self.controller = EnvController()


class QlibQuantHypothesis(Hypothesis):
    """
    Qlib 量化研究假设类，继承自通用的 Hypothesis 类。
    它额外增加了一个 `action` 属性，用于指明该假设是关于 "factor" 还是 "model"。
    """
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        action: str,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.action = action

    def __str__(self) -> str:
        """重写字符串表示，包含 action 信息。"""
        return f"""选择的动作: {self.action}
假设: {self.hypothesis}
原因: {self.reason}
"""


class QlibQuantHypothesisGen(FactorAndModelHypothesisGen):
    """
    Qlib 量化研究假设生成器。
    这是提案生成的核心，它决定下一步是生成因子假设还是模型假设，
    然后准备相应的上下文来调用大语言模型。
    """
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        """准备用于生成假设的上下文。"""

        # ========= 动作选择 =========
        # 1. 使用 Bandit 算法
        if QUANT_PROP_SETTING.action_selection == "bandit":
            if len(trace.hist) > 0:
                # 从上一个实验中提取指标
                metric = extract_metrics_from_experiment(trace.hist[-1][0])
                prev_action = trace.hist[-1][0].hypothesis.action
                # 记录上一个动作的结果并更新 Bandit
                trace.controller.record(metric, prev_action)
                # Bandit 决定下一个动作
                action = trace.controller.decide(metric)
            else:
                # 第一轮默认选择因子
                action = "factor"
        # 2. 使用大语言模型
        elif QUANT_PROP_SETTING.action_selection == "llm":
            hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=trace)
                if len(trace.hist) > 0
                else "由于这是第一轮，没有先前的假设和反馈可用。"
            )
            last_hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
                )
                if len(trace.hist) > 0
                else "由于这是第一轮，没有先前的假设和反馈可用。"
            )
            system_prompt = T("scenarios.qlib.prompts:action_gen.system").r()
            user_prompt = T("scenarios.qlib.prompts:action_gen.user").r(
                hypothesis_and_feedback=hypothesis_and_feedback,
                last_hypothesis_and_feedback=last_hypothesis_and_feedback,
            )
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
            action = json.loads(resp).get("action", "factor")
        # 3. 随机选择
        elif QUANT_PROP_SETTING.action_selection == "random":
            action = random.choice(["factor", "model"])
        self.targets = action

        # ========= 准备 RAG (Retrieval-Augmented Generation) 上下文 =========
        qaunt_rag = None
        if action == "factor":
            if len(trace.hist) < 6:
                qaunt_rag = "首先尝试从不同角度进行最简单、最快速的因子实验。"
            else:
                qaunt_rag = "现在，你需要尝试能够实现高 IC 的因子（例如，基于机器学习的因子）！不要包含与 SOTA 因子库中相似的因子！"
        elif action == "model":
            qaunt_rag = "1. 在量化金融中，市场数据可能是时间序列，GRU 模型/LSTM 模型适用于它们。目前不要生成 GNN 模型。\n2. 训练数据包含约47.8万个训练样本和约12.8万个验证样本。请相应地设计超参数并控制模型大小。这对训练结果有重大影响。如果您认为以前的模型本身很好，但训练超参数或模型超参数不是最优的，您可以返回相同的模型并调整这些参数。\n"

        # ========= 准备历史记录上下文 =========
        if len(trace.hist) == 0:
            hypothesis_and_feedback = "由于这是第一轮，没有先前的假设和反馈可用。"
        else:
            # 根据当前动作（factor/model）筛选相关的历史记录
            specific_trace = Trace(trace.scen)
            if action == "factor":
                # 包含所有因子实验和 SOTA 模型实验
                model_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):
                    if trace.hist[i][0].hypothesis.action == "factor":
                        specific_trace.hist.insert(0, trace.hist[i])
                    elif (
                        trace.hist[i][0].hypothesis.action == "model"
                        and trace.hist[i][1].decision is True
                        and not model_inserted
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        model_inserted = True
            elif action == "model":
                # 包含所有模型实验和所有 SOTA 因子实验
                factor_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):
                    if trace.hist[i][0].hypothesis.action == "model":
                        specific_trace.hist.insert(0, trace.hist[i])
                    elif (
                        trace.hist[i][0].hypothesis.action == "factor"
                        and trace.hist[i][1].decision is True
                        and not factor_inserted
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        factor_inserted = True

            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(trace=specific_trace)
            else:
                hypothesis_and_feedback = "没有可用的先前假设和反馈。"

        # 找到与当前动作相关的最近一次假设和反馈
        last_hypothesis_and_feedback = None
        for i in range(len(trace.hist) - 1, -1, -1):
            if trace.hist[i][0].hypothesis.action == action:
                last_hypothesis_and_feedback = T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                )
                break

        # 如果是模型动作，找到 SOTA 模型的假设和反馈
        sota_hypothesis_and_feedback = None
        if action == "model":
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][0].hypothesis.action == "model" and trace.hist[i][1].decision is True:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "RAG": qaunt_rag,
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format_with_action").r(),
            "hypothesis_specification": (
                T("scenarios.qlib.prompts:factor_hypothesis_specification").r()
                if action == "factor"
                else T("scenarios.qlib.prompts:model_hypothesis_specification").r()
            ),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        """将大语言模型的响应字符串转换为 QlibQuantHypothesis 对象。"""
        response_dict = json.loads(response)
        hypothesis = QlibQuantHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
            action=response_dict.get("action"),
        )
        return hypothesis
