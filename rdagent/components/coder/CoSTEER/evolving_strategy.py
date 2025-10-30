from __future__ import annotations

from abc import abstractmethod

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evolving_framework import EvolvingStrategy, EvoStep, QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.core.utils import multiprocessing_wrapper


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    """
    使用多进程来加速演进策略的基类。
    """
    # 可选的特殊键，用于表示演进主体变化的摘要
    KEY_CHANGE_SUMMARY = "__change_summary__"

    def __init__(self, scen: Scenario, settings: CoSTEERSettings, improve_mode: bool = False):
        super().__init__(scen)
        self.settings = settings
        self.improve_mode = improve_mode  # 改进模式意味着我们只实现之前失败过的任务。主要区别在于第一个循环不会实现所有任务。

    @abstractmethod
    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: QueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """
        抽象方法：实现单个任务。
        输入任务和当前工作空间，输出对工作空间的修改。

        Args:
            target_task (Task): 目标任务。
            queried_knowledge (QueriedKnowledge | None): 查询到的知识。
            workspace (FBWorkspace | None): 当前工作空间。
            prev_task_feedback (CoSTEERSingleFeedback | None): 上一个演进步骤的任务反馈。None表示这是第一个循环。

        Returns:
            dict[str, str]: 用于更新工作空间的新文件字典 {<文件名>: <内容>}。
                           - 特殊键: self.KEY_CHANGE_SUMMARY。
        """
        raise NotImplementedError

    @abstractmethod
    def assign_code_list_to_evo(self, code_list: list[dict], evo: EvolvingItem) -> None:
        """
        抽象方法：将生成的代码列表分配给演进项。
        代码列表与演进项的子任务对齐。如果某个任务未实现，则在列表中放置None。
        """
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> EvolvingItem:
        """
        执行一次演进。

        Args:
            evo (EvolvingItem): 当前要演进的实验项。
            queried_knowledge (CoSTEERQueriedKnowledge | None): 查询到的知识。
            evolving_trace (list[EvoStep]): 历史演进记录。

        Returns:
            EvolvingItem: 演进后的实验项。
        """
        code_list = [None] * len(evo.sub_tasks)

        last_feedback = evolving_trace[-1].feedback if evolving_trace else None
        if last_feedback:
            assert isinstance(last_feedback, CoSTEERMultiFeedback)

        # 1. 找出需要演进（即实现或修复）的任务
        to_be_finished_task_index: list[int] = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                # 如果是已知的成功任务，直接使用知识库中的实现
                code_list[index] = queried_knowledge.success_task_to_knowledge_dict[target_task_desc].implementation.file_dict
            else:
                # 确定是否要跳过任务
                # 在改进模式下，如果上一步没有失败反馈，则跳过
                skip_for_improve_mode = self.improve_mode and (
                    last_feedback is None or last_feedback[index] is None
                )

                # 如果任务未被标记为失败且不需要跳过，则安排执行
                if target_task_desc not in queried_knowledge.failed_task_info_set and not skip_for_improve_mode:
                    to_be_finished_task_index.append(index)

                if skip_for_improve_mode:
                    code_list[index] = {}  # 为跳过的任务设置空实现

        # 2. 使用多进程并行实现需要完成的任务
        tasks_to_run = [
            (
                self.implement_one_task,
                (
                    evo.sub_tasks[target_index],
                    queried_knowledge,
                    evo.experiment_workspace,
                    last_feedback[target_index] if last_feedback else None,
                ),
            )
            for target_index in to_be_finished_task_index
        ]

        results = multiprocessing_wrapper(tasks_to_run, n=RD_AGENT_SETTINGS.multi_proc_n)

        # 3. 将结果填入 code_list
        for i, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = results[i]

        # 4. 将生成的代码分配给演进项并返回
        evo = self.assign_code_list_to_evo(code_list, evo)
        return evo
