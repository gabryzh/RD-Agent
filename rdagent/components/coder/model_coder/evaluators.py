from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.model_coder.eva_utils import (
    ModelCodeEvaluator,
    ModelFinalEvaluator,
    shape_evaluator,
    value_evaluator,
)
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace

# 为模型评估反馈定义类型别名
ModelSingleFeedback = CoSTEERSingleFeedback
ModelMultiFeedback = CoSTEERMultiFeedback


class ModelCoSTEEREvaluator(CoSTEEREvaluator):
    """
    模型 CoSTEER 评估器。
    这是一个综合性的评估器，协调执行、形状、数值、代码和最终评估。
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelSingleFeedback:
        """
        对模型实现进行全面评估。

        Args:
            target_task (Task): 目标任务。
            implementation (Workspace): 当前的代码实现。
            gt_implementation (Workspace): 真实的标准实现（Ground Truth）。
            queried_knowledge (QueriedKnowledge, optional): 查询到的相关知识。

        Returns:
            ModelSingleFeedback: 包含所有评估维度的单一反馈对象。
        """
        target_task_information = target_task.get_task_information()

        # 检查知识库中是否已有成功或失败的记录
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            # 如果是已知的成功任务，直接返回缓存的反馈
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            # 如果是已知的重复失败任务，则跳过实现
            return ModelSingleFeedback(
                execution_feedback="此任务已失败多次，跳过实现。",
                shape_feedback="此任务已失败多次，跳过实现。",
                value_feedback="此任务已失败多次，跳过实现。",
                code_feedback="此任务已失败多次，跳过实现。",
                final_feedback="此任务已失败多次，跳过实现。",
                final_decision=False,
            )

        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)

        # NOTE: 使用固定的输入来测试模型，以避免随机性
        batch_size = 8
        num_features = 30
        num_timesteps = 40
        input_value = 0.4
        param_init_value = 0.6

        # 1. 执行评估：运行生成的代码
        model_execution_feedback, gen_np_array = implementation.execute(
            batch_size=batch_size,
            num_features=num_features,
            num_timesteps=num_timesteps,
            input_value=input_value,
            param_init_value=param_init_value,
        )

        # 如果有标准实现，也执行它以获取对比结果
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)
            _, gt_np_array = gt_implementation.execute(
                batch_size=batch_size,
                num_features=num_features,
                num_timesteps=num_timesteps,
                input_value=input_value,
                param_init_value=param_init_value,
            )
        else:
            gt_np_array = None

        # 2. 形状评估
        shape_feedback, shape_decision = shape_evaluator(
            gen_np_array,
            (batch_size, self.scen.model_output_channel if hasattr(self.scen, "model_output_channel") else 1),
        )

        # 3. 数值评估（如果标准实现存在）
        value_feedback, value_decision = value_evaluator(gen_np_array, gt_np_array)

        # 4. 代码评估
        code_feedback, _ = ModelCodeEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            gt_implementation=gt_implementation,
            model_execution_feedback=model_execution_feedback,
            model_value_feedback="\n".join([shape_feedback, value_feedback]),
        )

        # 5. 最终评估
        final_feedback, final_decision = ModelFinalEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            gt_implementation=gt_implementation,
            model_execution_feedback=model_execution_feedback,
            model_shape_feedback=shape_feedback,
            model_value_feedback=value_feedback,
            model_code_feedback=code_feedback,
        )

        # 组装并返回所有反馈信息
        return ModelSingleFeedback(
            execution_feedback=model_execution_feedback,
            shape_feedback=shape_feedback,
            value_feedback=value_feedback,
            code_feedback=code_feedback,
            final_feedback=final_feedback,
            final_decision=final_decision,
            value_generated_flag=(gen_np_array is not None),
            final_decision_based_on_gt=(gt_implementation is not None),
        )
