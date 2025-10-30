# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入ABC（抽象基类）和abstractmethod（抽象方法），用于定义抽象类
from abc import ABC, abstractmethod
# 导入TYPE_CHECKING（用于类型检查时避免循环导入）和Generic（用于创建泛型类）
from typing import TYPE_CHECKING, Generic

# 从rdagent.core.experiment模块导入ASpecificExp类
from rdagent.core.experiment import ASpecificExp

# 如果在类型检查时，导入Scenario类，避免循环依赖
if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario


# 定义一个名为Developer的抽象基类，它是一个泛型类，类型变量为ASpecificExp
class Developer(ABC, Generic[ASpecificExp]):
    # 类的构造函数
    def __init__(self, scen: Scenario) -> None:
        # 初始化场景实例
        self.scen: Scenario = scen

    # 定义一个抽象方法develop
    @abstractmethod
    def develop(self, exp: ASpecificExp) -> ASpecificExp:  # TODO: 移除返回值
        """
        任务生成器应该接受一个实验作为输入。

        因为不同任务的调度对最终性能至关重要，它会影响学习过程。

        当前约束：
        - 开发人员应该**就地**编辑实验，而不是返回值；
            - 因为我们有很多用例会引发错误，但我们需要实验中的中间结果。
        - 因此，我们将来应该移除返回值。

        职责：
        - 在开发之后生成一个新的实验。
        - 如果它试图为未来的开发传递消息，它应该设置一个ExperimentFeedback。
        """
        # 如果方法未实现，则引发NotImplementedError
        error_message = "generate方法未实现。"
        raise NotImplementedError(error_message)
