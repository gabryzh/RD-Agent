import io
import json
from abc import abstractmethod
from typing import Dict, Tuple

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.experiment import Task, Workspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class FactorEvaluator:
    """因子评估器的基类。"""

    def __init__(self, scen=None) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> Tuple[str, object]:
        """
        评估方法的抽象定义。

        您可以通过以下方式获取DataFrame:
        .. code-block:: python
            _, gen_df = implementation.execute()
            _, gt_df = gt_implementation.execute()

        Returns:
            Tuple[str, object]:
            - str: 评估结果的文本描述。
            - object: 一个可比较的指标（布尔值、整数、浮点数等）。如果评估器只有文本结果，则为None。
        """
        raise NotImplementedError("请实现 `evaluate` 方法")

    def _get_df(self, gt_implementation: Workspace, implementation: Workspace) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """辅助方法，用于从工作空间执行并获取DataFrame。"""
        if gt_implementation is not None:
            _, gt_df = gt_implementation.execute()
            if isinstance(gt_df, pd.Series):
                gt_df = gt_df.to_frame("gt_factor")
            if isinstance(gt_df, pd.DataFrame):
                gt_df = gt_df.sort_index()
        else:
            gt_df = None

        _, gen_df = implementation.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gen_df, pd.DataFrame):
            gen_df = gen_df.sort_index()
        return gt_df, gen_df

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorCodeEvaluator(FactorEvaluator):
    """使用 LLM 评估因子代码的质量。"""
    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        execution_feedback: str,
        value_feedback: str = "",
        gt_implementation: Workspace = None,
        **kwargs,
    ) -> Tuple[str, None]:
        factor_information = target_task.get_task_information()
        code = implementation.all_codes

        system_prompt = T(".prompts:evaluator_code_feedback_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(
                    target_task,
                    filtered_tag="feature",
                    simple_background=FACTOR_COSTEER_SETTINGS.simple_background,
                )
                if self.scen is not None
                else "没有场景描述。"
            )
        )

        # 循环缩减执行反馈以避免prompt超长
        execution_feedback_to_render = execution_feedback
        for _ in range(10):
            user_prompt = T(".prompts:evaluator_code_feedback_v1_user").r(
                factor_information=factor_information,
                code=code,
                execution_feedback=execution_feedback_to_render,
                value_feedback=value_feedback,
                gt_code=gt_implementation.code if gt_implementation else None,
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


class FactorInfEvaluator(FactorEvaluator):
    """评估因子值中是否存在无穷大值。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", False

        inf_count = gen_df.isin([float("inf"), -float("inf")]).sum().sum()
        if inf_count == 0:
            return "源DataFrame中没有任何无穷大值。", True
        else:
            return f"源DataFrame中有 {inf_count} 个无穷大值。请检查实现。", False


class FactorSingleColumnEvaluator(FactorEvaluator):
    """评估因子是否为单列。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", False

        if len(gen_df.columns) == 1:
            return "源DataFrame只有一列，这是正确的。", True
        else:
            return "源DataFrame有多于一列。请检查实现。我们只评估第一列。", False


class FactorOutputFormatEvaluator(FactorEvaluator):
    """使用 LLM 评估输出DataFrame的格式。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。跳过输出格式评估。", False

        buffer = io.StringIO()
        gen_df.info(buf=buffer)
        gen_df_info_str = f"用户当前正在处理一个特征相关的任务。\n输出的DataFrame信息如下:\n{buffer.getvalue()}"
        system_prompt = T(".prompts:evaluator_output_format_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(implementation.target_task, filtered_tag="feature")
                if self.scen is not None
                else "没有场景描述。"
            )
        )

        # 重试机制，以防LLM返回格式错误
        max_attempts = 3
        for attempts in range(max_attempts):
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                resp = api.build_messages_and_create_chat_completion(
                    user_prompt=gen_df_info_str,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str | bool | int],
                )
                resp_dict = json.loads(resp)
                decision = str(resp_dict["output_format_decision"]).lower() in ["true", "1"]
                return str(resp_dict["output_format_feedback"]), decision
            except (KeyError, json.JSONDecodeError) as e:
                if attempts >= max_attempts - 1:
                    raise KeyError("多次尝试后，JSON响应错误或缺少键。") from e

        return "多次尝试后评估输出格式失败。", False


class FactorDatetimeDailyEvaluator(FactorEvaluator):
    """评估因子的时间索引是否为日度数据。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。跳过日期时间格式评估。", False

        if "datetime" not in gen_df.index.names:
            return "源DataFrame没有datetime索引。请检查实现。", False

        try:
            datetimes = pd.to_datetime(gen_df.index.get_level_values("datetime"))
        except Exception:
            return f"datetime索引格式不正确。\n 输出DataFrame的头部信息: \n{gen_df.head()}", False

        time_diffs = datetimes.to_series().diff().dropna().unique()
        if pd.Timedelta(minutes=1) in time_diffs:
            return "生成的DataFrame不是日度的。实现肯定有误。", False

        return "生成的DataFrame是日度的。", True


class FactorRowCountEvaluator(FactorEvaluator):
    """评估生成因子的行数与标准实现的比例。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, float]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", 0.0

        ratio = min(len(gen_df), len(gt_df)) / max(len(gen_df), len(gt_df))
        feedback = f"源DataFrame与真实DataFrame的行数比例为 {ratio:.2f}。"
        if ratio <= 0.99:
            feedback += " 请验证实现。"
        return feedback, ratio


class FactorIndexEvaluator(FactorEvaluator):
    """评估生成因子的索引与标准实现的相似度。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, float]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", 0.0

        gen_index_set, gt_index_set = set(gen_df.index), set(gt_df.index)
        similarity = len(gen_index_set.intersection(gt_index_set)) / len(gen_index_set.union(gt_index_set))
        feedback = f"索引相似度为 {similarity:.2%}。"
        if similarity <= 0.99:
            feedback += " 索引相似度由共享索引数除以联合索引数计算得出。请检查实现。"
        return feedback, similarity


class FactorMissingValuesEvaluator(FactorEvaluator):
    """评估缺失值的数量是否一致。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", False

        gen_na_count = gen_df.isna().sum().sum()
        gt_na_count = gt_df.isna().sum().sum()
        if gen_na_count == gt_na_count:
            return "两个DataFrame具有相同的缺失值。", True
        else:
            return f"缺失值数量不同。源DataFrame有 {gen_na_count} 个，而真实DataFrame有 {gt_na_count} 个。", False


class FactorEqualValueRatioEvaluator(FactorEvaluator):
    """评估值相等的比例（在一定容忍度内）。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, float]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", -1.0

        try:
            close_values = gen_df.sub(gt_df).abs().lt(1e-6)
            acc_rate = close_values.sum().sum() / close_values.size
            if close_values.all().iloc[0]:
                return "所有值在1e-6的容忍度内相等。", acc_rate
            else:
                return "部分值的差异超过1e-6的容忍度。请检查舍入误差或计算方法的差异。", acc_rate
        except Exception:
            return "计算值相等比例时出错。", -1.0


class FactorCorrelationEvaluator(FactorEvaluator):
    """评估因子值的相关性（IC 和 RankIC）。"""
    def __init__(self, hard_check: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hard_check = hard_check  # 是否进行硬性检查（相关性必须高于阈值）

    def evaluate(self, implementation: Workspace, gt_implementation: Workspace) -> Tuple[str, bool | float]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "源DataFrame为None。请检查实现。", False

        concat_df = pd.concat([gen_df, gt_df], axis=1)
        concat_df.columns = ["source", "gt"]
        ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
        ric = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"], method="spearman")).dropna().mean()

        if self.hard_check:
            if ic > 0.99 and ric > 0.99:
                return f"DataFrame高度相关。IC为 {ic:.6f}，RankIC为 {ric:.6f}。", True
            else:
                return f"DataFrame相关性不足。IC为 {ic:.6f}，RankIC为 {ric:.6f}。", False
        else:
            return f"IC为 ({ic:.6f})，RankIC为 ({ric:.6f})。", ic


class FactorValueEvaluator(FactorEvaluator):
    """综合性的因子值评估器，调用多个子评估器。"""
    def evaluate(self, implementation: Workspace, gt_implementation: Workspace, version: int = 1, **kwargs) -> Tuple[str, bool | None]:
        conclusions = []

        # 检查无穷大值
        feedback_str, inf_evaluate_res = FactorInfEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        # 检查输出格式
        feedback_str, _ = FactorOutputFormatEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        if version == 1:
            feedback_str, daily_check_result = FactorDatetimeDailyEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)
        else:
            daily_check_result = None

        # 如果有标准实现，进行更详细的比较
        if gt_implementation is not None:
            feedback_str, row_result = FactorRowCountEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, index_result = FactorIndexEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, equal_value_ratio_result = FactorEqualValueRatioEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            if index_result > 0.99:
                feedback_str, high_correlation_result = FactorCorrelationEvaluator(hard_check=True, scen=self.scen).evaluate(implementation, gt_implementation)
            else:
                high_correlation_result = False
                feedback_str = "索引不同，放弃比较值和相关性。"
            conclusions.append(feedback_str)

        conclusion_str = "\n".join(conclusions)

        # 根据子评估结果得出综合决策
        if gt_implementation is not None and (equal_value_ratio_result > 0.99 or high_correlation_result):
            decision_from_value_check = True
        elif (row_result is not None and row_result <= 0.99 or daily_check_result is False or inf_evaluate_res is False):
            decision_from_value_check = False
        else:
            decision_from_value_check = None # 不确定

        return conclusion_str, decision_from_value_check


class FactorFinalDecisionEvaluator(FactorEvaluator):
    """使用 LLM 综合所有反馈，给出最终决策。"""
    def evaluate(self, target_task: FactorTask, execution_feedback: str, value_feedback: str, code_feedback: str, **kwargs) -> Tuple[bool | None, str | None]:
        system_prompt = T(".prompts:evaluator_final_decision_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")
                if self.scen is not None
                else "没有场景描述。"
            )
        )

        execution_feedback_to_render = execution_feedback
        # 缩减prompt
        for _ in range(10):
            user_prompt = T(".prompts:evaluator_final_decision_v1_user").r(
                factor_information=target_task.get_task_information(),
                execution_feedback=execution_feedback_to_render,
                code_feedback=code_feedback,
                value_feedback=value_feedback or "未提供真实值，未进行数值评估。",
            )
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                > APIBackend().chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        # 带重试的LLM调用
        max_attempts = 3
        for attempts in range(max_attempts):
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                resp = api.build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    seed=attempts,
                    json_target_type=Dict[str, str | bool | int],
                )
                final_evaluation_dict = json.loads(resp)
                final_decision = str(final_evaluation_dict["final_decision"]).lower() in ["true", "1"]
                final_feedback = final_evaluation_dict["final_feedback"]
                return final_decision, final_feedback
            except json.JSONDecodeError as e:
                raise ValueError("无法解码API的JSON响应。") from e
            except KeyError as e:
                if attempts >= max_attempts - 1:
                    raise KeyError("多次尝试后，API响应中缺少 'final_decision' 或 'final_feedback' 键。") from e

        return None, None
