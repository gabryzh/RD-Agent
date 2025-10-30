"""
该模块定义了评估框架的核心组件，预计将在不同的框架之间共享。
"""

# 导入ABC（抽象基类）和abstractmethod（抽象方法），用于定义抽象类
from abc import ABC, abstractmethod


class Feedback:
    """
    设计原则：
        - 这个类更像一个**数据类**（dataclass）。
        - 反馈的构建过程应该在评估器（evaluator）中完成。
    """

    def is_acceptable(self) -> bool:
        """
        判断方案是否可接受。

        有时，方案已经可以接受，但我们仍然希望对其进行优化。
        因此，我们使用不同的逻辑来判断方案是可接受的还是已完成的。
        """
        return self.__bool__()

    def finished(self) -> bool:
        """
        判断任务是否已完成。

        在某些实现中，任务可能会多次失败，导致代理跳过该实现。
        因此，跳过和成功都表示任务已完成。
        """
        return self.__bool__()

    def __bool__(self) -> bool:
        """
        默认情况下，反馈被认为是积极的（True）。
        """
        return True


class EvaluableObj:
    """
    一个可评估的信息集合。可以包含以下内容：
    - 任务（Task）
    - 解决方案（Solution）
    - 真实情况（Ground Truth）
    """


class Evaluator(ABC):
    """
    设计原则：

        - 评估器应该涵盖从原始信息构建反馈的整个过程。
            通常，反馈的构建分为两个阶段：
            1. 原始信息，包括标准输出（stdout）和工作区（workspace）（这部分由反馈本身处理）。
            2. 高级/摘要的反馈信息（这部分由评估器处理）。
    """

    @abstractmethod
    def evaluate(
        self,
        eo: EvaluableObj,
    ) -> Feedback:
        """
        评估一个可评估对象并返回反馈。

        这是一个抽象方法，需要在子类中实现。
        """
        raise NotImplementedError
