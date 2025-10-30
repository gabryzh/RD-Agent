import re
from datetime import datetime, timedelta

from rdagent.core.utils import SingletonBaseClass
from rdagent.log import rdagent_logger as logger


class RDAgentTimer:
    """一个简单的倒计时器类。"""
    def __init__(self) -> None:
        self.started: bool = False
        self.target_time: datetime | None = None
        self.all_duration: timedelta | None = None
        self._remain_time_duration: timedelta | None = None

    def reset(self, all_duration: str | timedelta) -> None:
        """
        重置计时器。
        可以接受字符串（如 "10s", "5m", "1h"）或 timedelta 对象作为总时长。
        """
        if isinstance(all_duration, str):
            pattern = re.compile(r"^\s*(\d*\.?\d+)\s*([smhd]?)\s*$")
            match = pattern.match(all_duration)
            if not match:
                return
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "s":
                self.all_duration = timedelta(seconds=value)
            elif unit == "m":
                self.all_duration = timedelta(minutes=value)
            elif unit == "h":
                self.all_duration = timedelta(hours=value)
            elif unit == "d":
                self.all_duration = timedelta(days=value)
            else:
                self.all_duration = timedelta(seconds=value)
        elif isinstance(all_duration, timedelta):
            self.all_duration = all_duration

        self.target_time = datetime.now() + self.all_duration
        logger.info(f"计时器设置为 {self.all_duration} 并开始倒计时。")
        self.started = True

    def restart_by_remain_time(self) -> None:
        """使用剩余时间重新启动计时器。"""
        if self._remain_time_duration is not None:
            self.target_time = datetime.now() + self._remain_time_duration
            self.started = True
            logger.info(f"计时器以剩余时间重新启动: {self._remain_time_duration}")
        else:
            logger.warning("没有剩余时间来重新启动计时器。")

    def add_duration(self, duration: timedelta) -> None:
        """为计时器增加额外的持续时间。"""
        if self.started and self.target_time is not None:
            logger.info(f"为计时器增加 {duration}。当前剩余 {self.remain_time()}。")
            self.target_time = self.target_time + duration
            self.update_remain_time()

    def is_timeout(self) -> bool:
        """检查计时器是否已超时。"""
        if self.started and self.target_time is not None:
            self.update_remain_time()
            if datetime.now() > self.target_time:
                return True
        return False

    def update_remain_time(self) -> None:
        """更新剩余时间。"""
        if self.started and self.target_time is not None:
            self._remain_time_duration = self.target_time - datetime.now()

    def remain_time(self) -> timedelta | None:
        """获取剩余时间。"""
        if self.started:
            self.update_remain_time()
            return self._remain_time_duration
        return None


class RDAgentTimerWrapper(SingletonBaseClass):
    """
    RDAgentTimer 的单例包装器。
    还增加了 API 失败计数的功能。
    """
    def __init__(self) -> None:
        self.timer: RDAgentTimer = RDAgentTimer()
        self.api_fail_count: int = 0
        self.latest_api_fail_time: datetime | None = None

    def replace_timer(self, timer: RDAgentTimer) -> None:
        """替换内部的计时器实例。"""
        self.timer = timer
        logger.info("计时器替换成功。")


# 创建一个全局的计时器包装器实例
RD_Agent_TIMER_wrapper: RDAgentTimerWrapper = RDAgentTimerWrapper()
