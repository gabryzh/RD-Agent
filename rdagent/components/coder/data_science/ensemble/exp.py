import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.utils import cache_with_pickle


# 因为我们使用 isinstance 来区分不同类型的任务，
# 所以我们需要使用子类来表示不同类型的任务。
class EnsembleTask(CoSTEERTask):
    """
    表示模型集成（Ensemble）的任务。
    继承自 CoSTEERTask，专门用于标识这是一个模型集成的任务。
    """
    pass
