import json
from typing import Dict, Tuple

import numpy as np

from rdagent.components.coder.CoSTEER.evaluators import CoSTEEREvaluator
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.experiment import Task, Workspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


# 该形状评估器也用于 data_science 模块
def shape_evaluator(prediction: np.ndarray, target_shape: Tuple = None) -> Tuple[str, bool]:
    """
    评估预测结果的形状是否正确。

    Args:
        prediction (np.ndarray): 模型的预测输出。
        target_shape (Tuple, optional): 目标形状。

    Returns:
        Tuple[str, bool]: 评估反馈信息和评估结果（True表示正确，False表示错误）。
    """
    if target_shape is None or prediction is None:
        return (
            "模型没有生成输出。未进行形状评估。",
            False,
        )
    pre_shape = prediction.shape

    if pre_shape == target_shape:
        return "输出的形状是正确的。", True
    else:
        return (
            f"输出的形状不正确。预期为 {target_shape}，但得到 {pre_shape}。",
            False,
        )


def value_evaluator(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[str, bool]:
    """
    评估预测结果的数值是否接近目标值。

    Args:
        prediction (np.ndarray): 模型的预测输出。
        target (np.ndarray): 真实目标值。

    Returns:
        Tuple[str, bool]: 评估反馈信息和评估结果。
    """
    if prediction is None:
        return "模型没有生成输出。跳过数值评估", False
    elif target is None:
        return (
            "未提供真实目标输出。数值评估不切实际",
            False,
        )
    else:
        # 计算平均绝对差
        diff = np.mean(np.abs(target - prediction))
        # 如果差值小于0.1，则认为数值正确
        return (
            f"输出的数值是正确的。平均绝对差为 {diff}。",
            diff < 0.1,
        )


class ModelCodeEvaluator(CoSTEEREvaluator):
    """
    模型代码评估器。
    使用 LLM 评估生成的代码质量。
    """
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        model_execution_feedback: str = "",
        model_value_feedback: str = "",
    ) -> Tuple[str, None]:
        """
        执行代码评估。

        Args:
            target_task (Task): 目标任务。
            implementation (Workspace): 当前的代码实现。
            gt_implementation (Workspace): 真实的标准实现（Ground Truth）。
            model_execution_feedback (str): 模型执行的反馈。
            model_value_feedback (str): 模型数值评估的反馈。

        Returns:
            Tuple[str, None]: LLM 生成的代码评估反馈，以及一个固定的 None 值。
        """
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)

        model_task_information = target_task.get_task_information()
        code = implementation.all_codes

        system_prompt = T(".prompts:evaluator_code_feedback.system").r(
            scenario=(
                self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                if self.scen is not None
                else "没有场景描述。"
            )
        )

        # 为了防止 prompt 超出 token 限制，循环缩减执行反馈的内容
        execution_feedback_to_render = model_execution_feedback
        for _ in range(10):  # 循环10次足以缩减内容
            user_prompt = T(".prompts:evaluator_code_feedback.user").r(
                model_information=model_task_information,
                code=code,
                model_execution_feedback=execution_feedback_to_render,
                model_value_feedback=model_value_feedback,
                gt_code=gt_implementation.all_codes if gt_implementation else None,
            )
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                > APIBackend().chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        return critic_response, None


class ModelFinalEvaluator(CoSTEEREvaluator):
    """
    模型最终评估器。
    综合所有反馈，给出最终的评估决策。
    """
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        model_execution_feedback: str,
        model_shape_feedback: str,
        model_value_feedback: str,
        model_code_feedback: str,
    ) -> Tuple[str, bool]:
        """
        执行最终评估。

        Args:
            target_task: 目标任务。
            implementation: 当前实现。
            gt_implementation: 标准实现。
            model_execution_feedback: 执行反馈。
            model_shape_feedback: 形状反馈。
            model_value_feedback: 数值反馈。
            model_code_feedback: 代码反馈。

        Returns:
            Tuple[str, bool]: 最终的反馈信息和最终决策（True/False）。
        """
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)

        system_prompt = T(".prompts:evaluator_final_feedback.system").r(
            scenario=(
                self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                if self.scen is not None
                else "没有场景描述。"
            )
        )

        # 同样，为了防止 prompt 超限而缩减执行反馈
        execution_feedback_to_render = model_execution_feedback
        for _ in range(10):
            user_prompt = T(".prompts:evaluator_final_feedback.user").r(
                model_information=target_task.get_task_information(),
                model_execution_feedback=execution_feedback_to_render,
                model_shape_feedback=model_shape_feedback,
                model_code_feedback=model_code_feedback,
                model_value_feedback=model_value_feedback,
            )

            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                > APIBackend().chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        # 调用 LLM 获取 JSON 格式的最终评估
        final_evaluation_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str | bool | int],
            ),
        )

        # 将字符串类型的布尔值转换为真实的布尔值
        if isinstance(final_evaluation_dict["final_decision"], str) and final_evaluation_dict[
            "final_decision"
        ].lower() in ("true", "false"):
            final_evaluation_dict["final_decision"] = final_evaluation_dict["final_decision"].lower() == "true"

        return (
            final_evaluation_dict["final_feedback"],
            final_evaluation_dict["final_decision"],
        )
