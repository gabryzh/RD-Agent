import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.utils import cache_with_pickle


# 因为我们使用 isinstance 来区分不同类型的任务，
# 所以我们需要使用子类来表示不同类型的任务。
class WorkflowTask(CoSTEERTask):
    """
    表示数据科学工作流构建的任务。
    继承自 CoSTEERTask，专门用于标识这是一个工作流级别的任务。
    """
    def __init__(self, name: str = "Workflow", *args, **kwargs) -> None:
        """
        初始化工作流任务。

        Args:
            name (str): 任务名称，默认为 "Workflow"。
        """
        super().__init__(name=name, *args, **kwargs)
