import re

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorFinalDecisionEvaluator,
    FactorValueEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

# 类型别名
FactorSingleFeedback = CoSTEERSingleFeedback


class FactorEvaluatorForCoder(CoSTEEREvaluator):
    """
    该类是用于单个因子实现的v1版本评估器。
    它调用共享模块中的多个评估器来评估因子实现。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始化各个子评估器
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback | None:
        """
        对单个因子实现进行全面评估。

        Args:
            target_task (FactorTask): 目标因子任务。
            implementation (Workspace): 当前的代码实现。
            gt_implementation (Workspace, optional): 标准实现（Ground Truth）。
            queried_knowledge (QueriedKnowledge, optional): 查询到的知识。

        Returns:
            FactorSingleFeedback | None: 包含所有评估维度的反馈对象，如果实现为None则返回None。
        """
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        # 检查知识库，看是否可以跳过评估
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorSingleFeedback(
                execution_feedback="此任务已失败多次，跳过实现。",
                value_generated_flag=False,
                code_feedback="此任务已失败多次，跳过代码评估。",
                value_feedback="此任务已失败多次，跳过数值评估。",
                final_decision=False,
                final_feedback="此任务已失败多次，跳过最终决策评估。",
                final_decision_based_on_gt=False,
            )

        # 开始评估流程
        factor_feedback = FactorSingleFeedback()

        # 1. 执行代码并获取反馈
        execution_feedback, gen_df = implementation.execute()
        # 从执行反馈中移除过长的数字列表，以简化反馈信息
        execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
        # 移除警告信息
        factor_feedback.execution_feedback = "\n".join(
            [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
        )

        # 2. 进行数值评估
        if gen_df is None:
            factor_feedback.value_feedback = "没有生成因子值，跳过数值评估。"
            factor_feedback.value_generated_flag = False
            decision_from_value_check = None
        else:
            factor_feedback.value_generated_flag = True
            factor_feedback.value_feedback, decision_from_value_check = self.value_evaluator.evaluate(
                implementation=implementation, gt_implementation=gt_implementation, version=target_task.version
            )

        factor_feedback.final_decision_based_on_gt = gt_implementation is not None

        # 3. 根据数值评估结果决定后续步骤
        if decision_from_value_check is True:
            # 如果数值评估通过，则无需进行代码评估
            factor_feedback.code_feedback = "最终决策为True，没有代码批评。"
            factor_feedback.final_decision = True
            factor_feedback.final_feedback = "数值评估通过，跳过最终决策评估。"
        elif decision_from_value_check is False:
            # 如果数值评估失败，需要进行代码评估以找出原因
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=gt_implementation,
            )
            factor_feedback.final_decision = False
            factor_feedback.final_feedback = "数值评估失败，跳过最终决策评估。"
        else:  # decision_from_value_check is None
            # 如果数值评估不确定（例如没有标准实现），则需要代码评估和最终决策评估
            factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                gt_implementation=gt_implementation,
            )
            factor_feedback.final_decision, factor_feedback.final_feedback = self.final_decision_evaluator.evaluate(
                target_task=target_task,
                execution_feedback=factor_feedback.execution_feedback,
                value_feedback=factor_feedback.value_feedback,
                code_feedback=factor_feedback.code_feedback,
            )

        return factor_feedback


# TODO: 实现一个通用的prompt缩短函数
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """当prompt过长时，我们需要缩短它。
    我们不应直接截断prompt，而应找到要缩短的关键部分并缩短它。
    """
    # TODO: 这个函数应该替换掉大部分在以下类中的代码：
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
    pass
