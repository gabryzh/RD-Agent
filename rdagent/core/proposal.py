# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import (
    ASpecificExp,
    ASpecificPlan,
    Experiment,
    ExperimentPlan,
)
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.scenario import Scenario

# 用于类型检查时导入，避免循环依赖
if TYPE_CHECKING:
    from rdagent.utils.workflow.loop import LoopBase


class Hypothesis:
    """
    假设类，用于表示一个待验证的想法。
    TODO: 我们可以为它取一个更好的名字，例如Belief。
    """

    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
    ) -> None:
        self.hypothesis: str = hypothesis
        self.reason: str = reason
        self.concise_reason: str = concise_reason
        self.concise_observation: str = concise_observation
        self.concise_justification: str = concise_justification
        self.concise_knowledge: str = concise_knowledge

    def __str__(self) -> str:
        return f"""假设: {self.hypothesis}
理由: {self.reason}"""


class ExperimentFeedback(Feedback):
    """实验反馈类，用于表示对实验的反馈信息。"""

    def __init__(
        self,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.decision = decision
        self.eda_improvement = eda_improvement
        self.reason = reason
        # Exception不为None表示由于异常而未能生成可运行的实验。
        # 可运行的结果并不总是好的。
        self.exception: Exception | None = exception
        self.code_change_summary = code_change_summary

    def __bool__(self) -> bool:
        return self.decision

    def __str__(self) -> str:
        res = f"决策: {self.decision}\n理由: {self.reason}"
        code_change_summary = getattr(self, "code_change_summary", None)
        if code_change_summary is not None:
            res += "\n代码变更摘要: " + code_change_summary
        return res

    @classmethod
    def from_exception(cls, e: Exception) -> ExperimentFeedback:
        """
        从异常创建反馈的便捷方法。
        """
        return cls(decision=False, reason=f"实验因 {e!s} 失败", exception=e)


class HypothesisFeedback(ExperimentFeedback):
    """假设反馈类，用于表示对假设的反馈信息。"""

    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
        acceptable: bool | None = None,
    ) -> None:
        super().__init__(
            reason,
            decision=decision,
            code_change_summary=code_change_summary,
            eda_improvement=eda_improvement,
        )
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis
        self.acceptable = acceptable

    def __str__(self) -> str:
        return f"""{super().__str__()}
观察: {self.observations}
假设评估: {self.hypothesis_evaluation}
新假设: {self.new_hypothesis}"""


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)
ASpecificKB = TypeVar("ASpecificKB", bound=KnowledgeBase)


class Trace(Generic[ASpecificScen, ASpecificKB]):
    """轨迹类，用于记录实验的演进过程。"""
    NodeType = tuple[Experiment, ExperimentFeedback]
    NEW_ROOT: tuple = ()

    def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
        self.scen: ASpecificScen = scen

        # 图结构
        self.hist: list[Trace.NodeType] = []
        self.dag_parent: list[tuple[int, ...]] = []

        self.idx2loop_id: dict[int, int] = {}

        self.knowledge_base: ASpecificKB | None = knowledge_base
        self.current_selection: tuple[int, ...] = (-1,)

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """获取当前最优的假设和实验。"""
        for experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return experiment.hypothesis, experiment

        return None, None

    def is_selection_new_tree(self, selection: tuple[int, ...] | None = None) -> bool:
        """检查当前选择是否是一个新的树。"""
        if selection is None:
            selection = self.get_current_selection()

        return selection == self.NEW_ROOT or len(self.dag_parent) == 0

    def get_current_selection(self) -> tuple[int, ...]:
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...]) -> None:
        self.current_selection = selection

    def get_parent_exps(
        self,
        selection: tuple[int, ...] | None = None,
    ) -> list[Trace.NodeType]:
        """获取给定选择的所有祖先实验。"""
        if selection is None:
            selection = self.get_current_selection()

        if self.is_selection_new_tree(selection):
            return []

        return [self.hist[i] for i in self.get_parents(selection[0])]

    def exp2idx(self, exp: Experiment | list[Experiment]) -> int | list[int] | None:
        if isinstance(exp, list):
            exps: list[Experiment] = exp
            exp_to_index: dict[Experiment, int] = {_exp: i for i, (_exp, _) in enumerate(self.hist)}
            return [exp_to_index[_exp] for _exp in exps]
        for i, (_exp, _) in enumerate(self.hist):
            if _exp == exp:
                return i
        return None

    def idx2exp(self, idx: int | list[int]) -> Experiment | list[Experiment]:
        if isinstance(idx, list):
            idxs: list[int] = idx
            return [self.hist[_idx][0] for _idx in idxs]
        return self.hist[idx][0]

    def is_parent(self, parent_idx: int, child_idx: int) -> bool:
        ancestors = self.get_parents(child_idx)
        return parent_idx in ancestors

    def get_parents(self, child_idx: int) -> list[int]:
        if self.is_selection_new_tree((child_idx,)):
            return []

        ancestors: list[int] = []
        curr = child_idx
        while True:
            ancestors.insert(0, curr)
            parent_tuple = self.dag_parent[curr]
            if not parent_tuple or parent_tuple[0] == curr:
                break
            curr = parent_tuple[0]

        return ancestors


class CheckpointSelector:
    """检查点选择器，用于从轨迹中选择一个检查点开始。"""

    @abstractmethod
    def get_selection(self, trace: Trace) -> tuple[int, ...] | None:
        """
        获取一个选择，表示从哪个检查点开始。
        返回`(-1,)`表示从最新的试验开始，`(idx,)`表示从第idx个试验开始，`None`表示从头开始。
        """


class SOTAexpSelector:
    """最优实验选择器，用于从轨迹中选择最优的实验进行提交。"""

    @abstractmethod
    def get_sota_exp_to_submit(self, trace: Trace) -> Experiment | None:
        """从轨迹中选择最优的实验进行提交。"""


class ExpPlanner(ABC, Generic[ASpecificPlan]):
    """实验规划器的抽象基类。"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def plan(self, trace: Trace) -> ASpecificPlan:
        """根据轨迹生成一个实验计划。"""


class ExpGen(ABC):
    """实验生成器的抽象基类。"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Experiment:
        """
        根据轨迹生成一个实验。
        """

    async def async_gen(self, trace: Trace, loop: LoopBase) -> Experiment:
        """
        异步生成实验，并决定是否暂停生成以将控制权交给其他例程。
        """
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                return self.gen(trace)
            await asyncio.sleep(1)

    def reset(self) -> None:
        """将生成器重置为初始状态。"""
        return


class HypothesisGen(ABC):
    """假设生成器的抽象基类。"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        """根据轨迹生成一个假设。"""


class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """将假设转换为实验的抽象基类。"""

    @abstractmethod
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ASpecificExp:
        """将假设转换为一个具体的实验。"""
        ...


class Experiment2Feedback(ABC):
    """从实验生成反馈的抽象基类。"""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> ExperimentFeedback:
        """
        为已执行的实验生成反馈。
        """
        error_message = "generate_feedback 方法未实现。"
        raise NotImplementedError(error_message)
