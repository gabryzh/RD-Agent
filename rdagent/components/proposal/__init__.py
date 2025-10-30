from abc import abstractmethod
from typing import Tuple

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    ExperimentPlan,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class LLMHypothesisGen(HypothesisGen):
    """
    使用大型语言模型（LLM）生成假设的基类。
    """
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # 以下方法与具体场景相关，应在子类中实现
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        """
        准备 LLM 请求的上下文。

        Args:
            trace (Trace): 包含历史信息的追踪对象。

        Returns:
            Tuple[dict, bool]: 包含上下文信息的字典和是否使用 JSON 模式的标志。
        """
        ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis:
        """
        将 LLM 的响应字符串转换为 Hypothesis 对象。

        Args:
            response (str): LLM 返回的响应。

        Returns:
            Hypothesis: 转换后的假设对象。
        """
        ...

    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        """
        生成假设的核心逻辑。

        Args:
            trace (Trace): 包含历史信息的追踪对象。
            plan (ExperimentPlan | None): 可选的实验计划。

        Returns:
            Hypothesis: 生成的假设对象。
        """
        # 1. 准备上下文
        context_dict, json_flag = self.prepare_context(trace)

        # 2. 构建 system 和 user prompt
        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=self.scen.get_scenario_all_desc(filtered_tag=self.targets),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=context_dict.get("last_hypothesis_and_feedback", ""),
            sota_hypothesis_and_feedback=context_dict.get("sota_hypothesis_and_feedback", ""),
            RAG=context_dict["RAG"],
        )

        # 3. 调用 LLM API
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
        )

        # 4. 转换并返回结果
        hypothesis = self.convert_response(resp)
        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    """因子生成的假设生成器。"""
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    """模型调优的假设生成器。"""
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    """特征工程和模型构建的假设生成器。"""
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    """
    使用 LLM 将假设转换为实验的基类。
    """
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        """
        为 LLM 请求准备上下文。

        Args:
            hypothesis (Hypothesis): 当前要转换的假设。
            trace (Trace): 包含历史信息的追踪对象。

        Returns:
            Tuple[dict, bool]: 包含上下文的字典和是否使用 JSON 模式的标志。
        """
        ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        """
        将 LLM 的响应转换为 Experiment 对象。

        Args:
            response (str): LLM 的响应。
            hypothesis (Hypothesis): 原始假设。
            trace (Trace): 追踪对象。

        Returns:
            Experiment: 转换后的实验对象。
        """
        ...

    @wait_retry(retry_n=5)  # 转换失败时最多重试5次
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        """
        将假设转换为实验的核心逻辑。
        """
        # 1. 准备上下文
        context, json_flag = self.prepare_context(hypothesis, trace)

        # 2. 构建 system 和 user prompt
        system_prompt = T(".prompts:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(".prompts:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=context.get("hypothesis_and_feedback", ""),
            last_hypothesis_and_feedback=context.get("last_hypothesis_and_feedback", ""),
            sota_hypothesis_and_feedback=context.get("sota_hypothesis_and_feedback", ""),
            target_list=context["target_list"],
            RAG=context["RAG"],
        )

        # 3. 调用 LLM API
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, dict[str, str | dict]]
        )

        # 4. 转换并返回结果
        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    """将因子生成假设转换为实验。"""
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    """将模型调优假设转换为实验。"""
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    """将特征工程和模型构建假设转换为实验。"""
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
