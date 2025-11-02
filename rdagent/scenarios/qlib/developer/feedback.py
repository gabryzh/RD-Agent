# 导入 json 库，用于处理 JSON 数据
import json
# 导入 Path 类，用于处理文件系统路径
from pathlib import Path
# 导入类型提示 Dict
from typing import Dict

# 导入 pandas 用于数据处理
import pandas as pd

# 导入 Experiment 基类
from rdagent.core.experiment import Experiment
# 导入反馈相关的类
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入与大语言模型 API 交互的后端
from rdagent.oai.llm_utils import APIBackend
# 导入 Qlib 量化场景类
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
# 导入布尔值转换工具
from rdagent.utils import convert2bool
# 导入模板工具
from rdagent.utils.agent.tpl import T

# 获取当前文件所在的目录
DIRNAME = Path(__file__).absolute().resolve().parent

# 定义重要的评估指标
IMPORTANT_METRICS = [
    "IC",  # 信息系数
    "1day.excess_return_with_cost.annualized_return",  # 考虑成本的单日超额收益的年化回报率
    "1day.excess_return_with_cost.max_drawdown",  # 考虑成本的单日超额收益的最大回撤
]


def process_results(current_result, sota_result):
    """
    处理并比较当前实验结果和 SOTA（State-of-the-art）实验结果。
    :param current_result: 当前实验结果。
    :param sota_result: SOTA 实验结果。
    :return: 格式化后的重要指标对比字符串。
    """
    # 将结果转换为 DataFrame
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # 设置索引名称为 'metric'
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # 重命名列名以区分不同结果
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # 合并两个 DataFrame
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # 筛选出重要的指标
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        """
        将筛选后的 DataFrame 格式化为字符串。
        """
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    """
    将 Qlib 因子实验结果转换为反馈。
    """
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        为给定的因子实验和假设生成反馈。

        Args:
            exp (QlibFactorExperiment): 需要生成反馈的因子实验。
            trace (Trace): 实验的追踪记录。

        Returns:
            HypothesisFeedback: 为实验和假设生成的反馈。
        """
        hypothesis = exp.hypothesis
        logger.info("正在生成反馈...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        # 获取实验中所有子任务的信息和实现结果
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # 处理并比较结果
        combined_result = process_results(current_result, sota_result)

        # 生成系统提示
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # 生成用户提示
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # 调用大语言模型 API 生成反馈
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # 解析 JSON 响应
        response_json = json.loads(response)

        # 从 JSON 响应中提取字段
        observations = response_json.get("Observations", "未提供观察结果")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "未提供假设反馈")
        new_hypothesis = response_json.get("New Hypothesis", "未提供新假设")
        reason = response_json.get("Reasoning", "未提供原因")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    """
    将 Qlib 模型实验结果转换为反馈。
    """
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        为给定的模型实验和假设生成反馈。

        Args:
            exp (QlibModelExperiment): 需要生成反馈的模型实验。
            trace (Trace): 实验的追踪记录。

        Returns:
            HypothesisFeedback: 为实验和假设生成的反馈。
        """
        hypothesis = exp.hypothesis
        logger.info("正在生成反馈...")

        # 生成系统提示
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # 获取 SOTA 的假设和实验结果
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        # 生成用户提示
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "执行失败",
        )

        # 调用大语言模型 API 生成反馈
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # 解析 JSON 响应
        response_json_hypothesis = json.loads(response)

        # 再次调用大语言模型 API，此处的逻辑似乎有重复，可能需要审查
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # 解析 JSON 响应
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "未提供观察结果"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "未提供假设反馈"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "未提供新假设"),
            reason=response_json_hypothesis.get("Reasoning", "未提供原因"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
