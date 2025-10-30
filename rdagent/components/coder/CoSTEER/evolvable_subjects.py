from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.experiment import Experiment, FBWorkspace, Task
from rdagent.log import rdagent_logger as logger


class EvolvingItem(Experiment, EvolvableSubjects):
    """
    因子实现的中间产物，代表一个可演进的实验项。

    该类同时继承自 `Experiment` 和 `EvolvableSubjects`，
    使其既能像实验一样包含任务和工作空间，又能作为演进框架中的主体进行迭代。
    """

    def __init__(
        self,
        sub_tasks: list[Task],
        sub_gt_implementations: list[FBWorkspace] = None,
    ):
        """
        初始化一个可演进项。

        Args:
            sub_tasks (list[Task]): 包含的子任务列表。
            sub_gt_implementations (list[FBWorkspace], optional):
                与子任务对应的标准实现（Ground Truth）工作空间列表。
                如果长度与子任务列表不匹配，将被忽略。
        """
        # 初始化父类 Experiment
        Experiment.__init__(self, sub_tasks=sub_tasks)

        # 验证并设置标准实现
        if sub_gt_implementations is not None and len(sub_gt_implementations) != len(self.sub_tasks):
            self.sub_gt_implementations = None
            logger.warning(
                "标准实现（sub_gt_implementations）的长度与子任务（sub_tasks）的长度不相等，"
                "已将 sub_gt_implementations 设置为 None",
            )
        else:
            self.sub_gt_implementations = sub_gt_implementations

    @classmethod
    def from_experiment(cls, exp: Experiment) -> "EvolvingItem":
        """
        从一个标准的 Experiment 对象创建一个 EvolvingItem 实例。

        Args:
            exp (Experiment): 源实验对象。

        Returns:
            EvolvingItem: 新创建的可演进项实例。
        """
        ei = cls(sub_tasks=exp.sub_tasks)
        # 复制实验的相关属性
        ei.based_experiments = exp.based_experiments
        ei.experiment_workspace = exp.experiment_workspace
        return ei
