# 导入 CoSTEER 基类
from rdagent.components.coder.CoSTEER import CoSTEER


class DSCoSTEER(CoSTEER):
    """
    数据科学领域的 CoSTEER 类。
    继承自通用的 CoSTEER 框架，并针对数据科学场景进行了特定的调整。
    """
    def get_develop_max_seconds(self) -> int | None:
        """
        获取开发过程的最大允许秒数。

        在数据科学场景中，编码器使用场景的真实调试超时时间乘以一个系数
        作为开发过程的最大时间限制。

        返回:
            int | None: 最大开发秒数，如果未设置则返回 None。
        """
        return int(self.scen.real_debug_timeout() * self.settings.max_seconds_multiplier)
