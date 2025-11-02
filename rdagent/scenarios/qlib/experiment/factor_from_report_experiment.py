# 从 copy 模块导入 deepcopy 函数，用于创建对象的深拷贝
from copy import deepcopy

# 从 qlib 因子实验模块导入 QlibFactorScenario 类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
# 从 rdagent 的模板工具模块导入 T
from rdagent.utils.agent.tpl import T


class QlibFactorFromReportScenario(QlibFactorScenario):
    """
    一个特定的 Qlib 因子场景，专门用于从研究报告中提取和实现因子。
    它继承自通用的 QlibFactorScenario，并重写了富文本样式描述，
    以提供更针对性的指令。
    """
    def __init__(self) -> None:
        super().__init__()
        # 使用模板，重写富文本样式描述，使其适用于从报告中提取因子的任务
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_from_report_rich_style_description").r())

    @property
    def rich_style_description(self) -> str:
        """
        返回适用于“从报告中提取因子”场景的富文本样式描述。
        """
        return self._rich_style_description
