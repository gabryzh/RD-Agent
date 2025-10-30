# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入ABC（抽象基类）和abstractmethod（抽象方法）
from abc import ABC, abstractmethod
# 导入Trace类
from trace import Trace
# 导入TYPE_CHECKING和Generic，用于类型提示
from typing import TYPE_CHECKING, Generic

# 从rdagent.core.experiment模块导入ASpecificExp
from rdagent.core.experiment import ASpecificExp

# 如果在类型检查时，导入Scenario类，以避免循环依赖
if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario


class Interactor(ABC, Generic[ASpecificExp]):
    """交互器抽象基类，用于与实验进行交互。"""
    def __init__(self, scen: Scenario) -> None:
        """
        初始化交互器。
        :param scen: 场景实例。
        """
        self.scen: Scenario = scen

    @abstractmethod
    def interact(self, exp: ASpecificExp, trace: Trace | None = None) -> ASpecificExp:
        """
        与实验进行交互以获取反馈或确认。

        职责：
        - 展示实验的当前状态。
        - 收集输入以指导实验的后续步骤。
        - 根据反馈重写实验。
        """
