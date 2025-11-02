"""
用于实验追踪的追踪模块，使用 MLflow。

该模块提供了一个清晰的接口来追踪指标和参数，
同时根据配置使 MLflow 成为一个可选依赖。
"""

import datetime
from typing import TYPE_CHECKING

import pytz

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper

# 使用 TYPE_CHECKING 来避免循环依赖，这只在类型检查时导入
if TYPE_CHECKING:
    from rdagent.utils.workflow.loop import LoopBase

from rdagent.log import rdagent_logger as logger

# 为 mlflow 定义一个占位符，以防它不可用
mlflow = None

# 条件导入，使 MLflow 成为可选的
if RD_AGENT_SETTINGS.enable_mlflow:
    try:
        import mlflow  # type: ignore[assignment]
    except ImportError:
        logger.warning("MLflow 在设置中已启用，但无法导入。")
        RD_AGENT_SETTINGS.enable_mlflow = False


class WorkflowTracker:
    """
    一个特定于工作流的追踪系统，用于记录与工作流执行相关的指标。

    此类处理指标记录，同时保持 MLflow 依赖的可选性。
    如果 MLflow 在设置中未启用，则追踪调用将不执行任何操作 (no-ops)。
    """

    def __init__(self, loop_base: "LoopBase"):
        """
        使用一个 LoopBase 实例初始化 WorkflowTracker。

        Args:
            loop_base: 要为其追踪指标的 LoopBase 实例。
        """
        self.loop_base = loop_base

    @staticmethod
    def is_enabled() -> bool:
        """检查追踪功能是否已启用。"""
        return RD_AGENT_SETTINGS.enable_mlflow

    @staticmethod
    def _datetime_to_float(dt: datetime.datetime) -> float:
        """
        将 datetime 对象转换为一个结构化的浮点数表示。
        例如：2023-10-27 10:30:45 -> 20231027103045.0 (近似)
        """
        return dt.second + dt.minute * 1e2 + dt.hour * 1e4 + dt.day * 1e6 + dt.month * 1e8 + dt.year * 1e10

    def log_workflow_state(self) -> None:
        """
        从关联的 LoopBase 实例中记录所有工作流状态指标。
        """
        # 如果未启用 mlflow 或 mlflow 未成功导入，则直接返回
        if not RD_AGENT_SETTINGS.enable_mlflow or mlflow is None:
            return

        try:
            # 记录工作流进度
            mlflow.log_metric("loop_index", self.loop_base.loop_idx)
            mlflow.log_metric("step_index", self.loop_base.step_idx[self.loop_base.loop_idx])

            # 记录当前时间（上海时区）
            current_local_datetime = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
            float_like_datetime = self._datetime_to_float(current_local_datetime)
            mlflow.log_metric("current_datetime", float_like_datetime)

            # 记录 API 状态
            mlflow.log_metric("api_fail_count", RD_Agent_TIMER_wrapper.api_fail_count)
            latest_api_fail_time = RD_Agent_TIMER_wrapper.latest_api_fail_time
            if latest_api_fail_time is not None:
                float_like_datetime = self._datetime_to_float(latest_api_fail_time)
                mlflow.log_metric("lastest_api_fail_time", float_like_datetime)

            # 如果计时器已启动，则记录计时器状态
            if self.loop_base.timer.started:
                remain_time = self.loop_base.timer.remain_time()
                assert remain_time is not None, "计时器已启动但剩余时间为 None"
                mlflow.log_metric("remain_time", remain_time.total_seconds())
                # 记录剩余时间百分比
                if self.loop_base.timer.all_duration:
                    mlflow.log_metric(
                        "remain_percent",
                        remain_time / self.loop_base.timer.all_duration * 100,
                    )

        except Exception as e:
            # 捕获并记录在追踪过程中可能发生的任何异常
            logger.warning(f"方法 log_workflow_state 中发生错误: {e}")
