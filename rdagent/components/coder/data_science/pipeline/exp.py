from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# 因为我们使用 isinstance 来区分不同类型的任务，
# 所以我们需要使用子类来表示不同类型的任务。
class PipelineTask(CoSTEERTask):
    """
    表示数据科学流程中的管道（Pipeline）构建任务。
    """
    def __init__(self, name: str = "Pipeline", package_info: str | None = None, *args, **kwargs) -> None:
        """
        初始化管道任务。

        Args:
            name (str): 任务名称，默认为 "Pipeline"。
            package_info (str | None): 与任务相关的包信息。
        """
        super().__init__(name=name, *args, **kwargs)
        self.package_info = package_info
