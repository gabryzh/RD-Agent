# 导入类型提示 List
from typing import List

# 导入 pandas 用于数据处理
import pandas as pd

# 导入 CoSTEERMultiFeedback 类，用于处理多重反馈
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
# 导入 RD_AGENT 的配置
from rdagent.core.conf import RD_AGENT_SETTINGS
# 导入自定义异常类
from rdagent.core.exception import FactorEmptyError
# 导入多进程包装器
from rdagent.core.utils import multiprocessing_wrapper
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def process_factor_data(exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
    """
    处理并合并来自实验实现的因子数据。

    Args:
        exp_or_list (List[QlibFactorExperiment] | QlibFactorExperiment): 包含因子数据的实验或实验列表。

    Returns:
        pd.DataFrame: 合并后的不含 NaN 值的因子数据。
    """
    # 如果传入的是单个实验，则将其放入列表中
    if isinstance(exp_or_list, QlibFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []

    # 收集所有实验的 DataFrame
    for exp in exp_or_list:
        if isinstance(exp, QlibFactorExperiment):
            if len(exp.sub_tasks) > 0:
                # 如果没有子任务，说明实验是模板项目的结果。
                # 否则，它是根据设计的任务开发的，应该有反馈。
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                # 迭代子实现并执行它们以获取每个因子数据
                # 使用多进程并行执行
                message_and_df_list = multiprocessing_wrapper(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ],  # 只执行有成功反馈的实现
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                error_message = ""
                for message, df in message_and_df_list:
                    # 检查因子生成是否成功
                    if df is not None and "datetime" in df.index.names:
                        # 检查时间戳的差异，排除分钟级别的数据
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(df)
                            logger.info(
                                f"来自 {exp.hypothesis.concise_justification} 的因子数据已成功生成。"
                            )
                        else:
                            logger.warning(f"来自 {exp.hypothesis.concise_justification} 的因子数据未生成。")
                    else:
                        error_message += f"来自 {exp.hypothesis.concise_justification} 的因子数据因 {message} 未能生成"
                        logger.warning(
                            f"来自 {exp.hypothesis.concise_justification} 的因子数据因 {message} 未能生成"
                        )

    # 合并所有成功的因子数据
    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        # 如果没有有效的因子数据，则抛出异常
        raise FactorEmptyError(
            f"在 process_factor_data 中没有找到有效的因子数据进行合并，原因: {error_message}。"
        )
