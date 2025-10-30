from rdagent.core.experiment import Task


class CoSTEERTask(Task):
    """
    CoSTEER 框架中的任务基类。
    它继承自通用的 `Task` 类，并添加了一个可选的 `base_code` 属性。
    """
    def __init__(self, base_code: str = None, *args, **kwargs) -> None:
        """
        初始化 CoSTEER 任务。

        Args:
            base_code (str, optional): 基础代码。默认为 None。
        """
        super().__init__(*args, **kwargs)
        # TODO: 我们也许可以将 base_code 升级为一个类似工作空间的东西来了解之前的状态。
        # NOTE (xiao): 认为我们不再需要 base_code。信息应该从工作空间中检索。
        self.base_code = base_code
