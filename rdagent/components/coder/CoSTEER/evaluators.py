from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator, Feedback
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario

# TODO:
# 1. 逻辑上合理，但目前缺少应用场景。
# 2. 如果证明有用，应将其移至更通用的位置。
#
# class FBWorkspaceExeFeedback(Feedback):
#     """
#     在抽象层面上与 FBWorkspace 配对。
#     """
#     # ws: FBWorkspace   # 潜在属性
#     stdout: str


@dataclass
class CoSTEERSingleFeedback(Feedback):
    """
    针对单个实现的 CoSTEER 反馈。
    这是一个通用的反馈类，其设计与代码实现的不同阶段对齐：
    - 执行 -> 返回值检查 -> 代码 -> 最终决策
    """
    execution: str  # 执行阶段的反馈
    return_checking: Optional[str]  # 返回值检查阶段的反馈（包括对生成值的各种约束检查）
    code: str  # 对代码本身的反馈
    final_decision: Optional[bool] = None  # 最终决策 (True: 成功, False: 失败)

    @staticmethod
    def val_and_update_init_dict(data: dict) -> dict:
        """在初始化之前验证和转换数据字典。"""
        if "final_decision" not in data:
            raise ValueError("缺少 'final_decision' 字段")

        # 将字符串布尔值转换为实际的布尔值
        if isinstance(data["final_decision"], str):
            val = data["final_decision"].lower()
            if val in ("false", "0"):
                data["final_decision"] = False
            elif val in ("true", "1"):
                data["final_decision"] = True

        if not isinstance(data["final_decision"], bool):
            raise ValueError(f"'final_decision' 必须是布尔值, 而不是 {type(data['final_decision'])}")

        for attr in "execution", "return_checking", "code":
            if data.get(attr) is not None and not isinstance(data[attr], str):
                raise ValueError(f"'{attr}' 必须是字符串, 而不是 {type(data[attr])}")
        return data

    @classmethod
    def merge(cls, feedback_li: list["CoSTEERSingleFeedback"]) -> "CoSTEERSingleFeedback":
        """
        将多个反馈实例合并为一个。
        注意：此方法仅基于 CoSTEERSingleFeedback 的属性进行合并，
        如果子类有更复杂的结构，应重写此方法以避免信息丢失。
        """
        fb = deepcopy(feedback_li[0])

        # 最终决策是所有反馈决策的“与”操作
        fb.final_decision = all(f.final_decision for f in feedback_li)

        # 将文本反馈拼接在一起
        for attr in "execution", "return_checking", "code":
            setattr(
                fb,
                attr,
                "\n\n".join([getattr(f, attr) for f in feedback_li if getattr(f, attr) is not None]),
            )
        return fb

    def __str__(self) -> str:
        return f"""------------------执行------------------
{self.execution}
------------------返回值检查------------------
{self.return_checking if self.return_checking is not None else '没有返回值检查'}
------------------代码------------------
{self.code}
------------------最终决策------------------
此实现为 {'成功' if self.final_decision else '失败'}。
"""

    def __bool__(self):
        return self.final_decision


class CoSTEERSingleFeedbackDeprecated(CoSTEERSingleFeedback):
    """
    一个已弃用的反馈基类，用于所有对单一实现的代码生成器反馈。
    它具有更具体的字段，并通过属性映射到新版 `CoSTEERSingleFeedback` 的通用字段，以实现向后兼容。
    """
    def __init__(
        self,
        execution_feedback: str = None,
        shape_feedback: str = None,
        code_feedback: str = None,
        value_feedback: str = None,
        final_decision: bool = None,
        final_feedback: str = None,
        value_generated_flag: bool = None,
        final_decision_based_on_gt: bool = None,
    ) -> None:
        self.execution_feedback = execution_feedback
        self.code_feedback = code_feedback
        self.value_feedback = value_feedback
        self.final_decision = final_decision
        self.final_feedback = final_feedback
        self.value_generated_flag = value_generated_flag
        self.final_decision_based_on_gt = final_decision_based_on_gt
        self.shape_feedback = shape_feedback  # TODO: 不够通用，应考虑移入子类

    # --- 属性映射，用于兼容 CoSTEERSingleFeedback ---
    @property
    def execution(self):
        return self.execution_feedback
    @execution.setter
    def execution(self, value):
        self.execution_feedback = value

    @property
    def return_checking(self):
        if self.value_generated_flag:
            return f"数值反馈: {self.value_feedback}\n\n形状反馈: {self.shape_feedback}"
        return None
    @return_checking.setter
    def return_checking(self, value):
        # 由于 return_checking 是派生属性，这里仅做简单赋值
        self.value_feedback = value
        self.shape_feedback = value

    @property
    def code(self):
        return self.code_feedback
    @code.setter
    def code(self, value):
        self.code_feedback = value

    def __str__(self) -> str:
        return f"""------------------执行反馈------------------
{self.execution_feedback or '无执行反馈'}
------------------形状反馈------------------
{self.shape_feedback or '无形状反馈'}
------------------代码反馈------------------
{self.code_feedback or '无代码反馈'}
------------------数值反馈------------------
{self.value_feedback or '无数值反馈'}
------------------最终反馈------------------
{self.final_feedback or '无最终反馈'}
------------------最终决策------------------
此实现为 {'成功' if self.final_decision else '失败'}。
"""


class CoSTEERMultiFeedback(Feedback):
    """包含一个列表的反馈，每个元素对应一个因子实现的反馈。"""
    def __init__(self, feedback_list: List[CoSTEERSingleFeedback]) -> None:
        self.feedback_list = feedback_list

    def __getitem__(self, index: int) -> CoSTEERSingleFeedback:
        return self.feedback_list[index]
    def __len__(self) -> int:
        return len(self.feedback_list)
    def append(self, feedback: CoSTEERSingleFeedback) -> None:
        self.feedback_list.append(feedback)
    def __iter__(self):
        return iter(self.feedback_list)

    def is_acceptable(self) -> bool:
        return all(feedback.is_acceptable() for feedback in self.feedback_list)

    def finished(self) -> bool:
        """检查所有非None的反馈是否都成功。"""
        return all(feedback.final_decision for feedback in self.feedback_list if feedback is not None)

    def __bool__(self) -> bool:
        return all(feedback.final_decision for feedback in self.feedback_list)


class CoSTEEREvaluator(Evaluator):
    """CoSTEER 评估器的抽象基类。"""
    def __init__(self, scen: "Scenario") -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(self, target_task: Task, implementation: Workspace, gt_implementation: Workspace, **kwargs) -> CoSTEERSingleFeedback:
        raise NotImplementedError("请实现 `evaluate` 方法")


class CoSTEERMultiEvaluator(CoSTEEREvaluator):
    """用于实验的评估器。由于有多个任务，因此返回一个评估反馈列表。"""
    def __init__(self, single_evaluator: CoSTEEREvaluator | list[CoSTEEREvaluator], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.single_evaluator = single_evaluator

    def evaluate(self, evo: EvolvingItem, queried_knowledge: QueriedKnowledge = None, **kwargs) -> CoSTEERMultiFeedback:
        eval_l = self.single_evaluator if isinstance(self.single_evaluator, list) else [self.single_evaluator]

        # 1) 并行评估每个子任务
        tasks_to_run = [
            (
                ev.evaluate,
                (
                    evo.sub_tasks[index],
                    evo.sub_workspace_list[index],
                    evo.sub_gt_implementations[index] if evo.sub_gt_implementations else None,
                    queried_knowledge,
                ),
            )
            for index in range(len(evo.sub_tasks))
            for ev in eval_l
        ]

        all_feedbacks = multiprocessing_wrapper(tasks_to_run, n=RD_AGENT_SETTINGS.multi_proc_n)

        # 将反馈按评估器分组
        num_evaluators = len(eval_l)
        num_tasks = len(evo.sub_tasks)
        task_li_feedback_li = [all_feedbacks[i:i + num_tasks] for i in range(0, len(all_feedbacks), num_tasks)]

        # 2) 合并针对每个子任务的多个评估反馈
        merged_task_feedback = []
        for task_id in range(num_tasks):
            feedbacks_for_task = [fb_li[task_id] for fb_li in task_li_feedback_li]
            if feedbacks_for_task and feedbacks_for_task[0] is not None:
                 merged_feedback = feedbacks_for_task[0].merge(feedbacks_for_task)
                 merged_task_feedback.append(merged_feedback)
            else:
                 merged_task_feedback.append(None)

        final_decisions = [fb.final_decision if fb else None for fb in merged_task_feedback]
        logger.info(f"最终决策: {final_decisions} 成功数量: {final_decisions.count(True)}")

        # 兼容性代码：如果任务成功，更新任务对象的 `factor_implementation` 标志
        for index, task in enumerate(evo.sub_tasks):
            if final_decisions[index] and hasattr(task, 'factor_implementation'):
                task.factor_implementation = True

        return CoSTEERMultiFeedback(merged_task_feedback)
