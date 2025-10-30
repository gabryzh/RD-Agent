# 导入类型提示相关的模块
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# 因为我们使用 isinstance 来区分不同类型的任务，所以我们需要使用子类来表示不同类型的任务
class ModelTask(CoSTEERTask):
    """
    模型任务类。
    继承自 CoSTEERTask，专门用于表示与数据科学模型相关的任务。
    """
    def __init__(
        self,
        name: str,
        description: str,
        *args,
        **kwargs,
    ) -> None:
        """
        初始化一个模型任务。

        参数:
            name (str): 任务的名称，通常对应于模型的名称。
            description (str): 任务的详细描述。
        """
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        """
        获取任务的详细信息。

        返回:
            str: 一个包含任务名称和描述的格式化字符串。
        """
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        return task_desc
