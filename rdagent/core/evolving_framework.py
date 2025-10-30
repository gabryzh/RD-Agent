# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入copy模块，用于对象拷贝
import copy
# 导入ABC（抽象基类）和abstractmethod（抽象方法）
from abc import ABC, abstractmethod
# 导入dataclass，用于创建数据类
from dataclasses import dataclass
# 导入TYPE_CHECKING, Any, Generic, TypeVar，用于类型提示
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# 从rdagent.core.evaluation模块导入EvaluableObj
from rdagent.core.evaluation import EvaluableObj
# 从rdagent.core.knowledge_base模块导入KnowledgeBase
from rdagent.core.knowledge_base import KnowledgeBase

# 如果在类型检查时，导入Feedback和Scenario类，避免循环依赖
if TYPE_CHECKING:
    from rdagent.core.evaluation import Feedback
    from rdagent.core.scenario import Scenario


class Knowledge:
    """知识的基类"""
    pass


class QueriedKnowledge:
    """查询到的知识的基类"""
    pass


class EvolvingKnowledgeBase(KnowledgeBase):
    """可演进的知识库"""
    @abstractmethod
    def query(
        self,
    ) -> QueriedKnowledge | None:
        """查询知识库"""
        raise NotImplementedError


class EvolvableSubjects(EvaluableObj):
    """可演进的目标对象"""

    def clone(self) -> EvolvableSubjects:
        """克隆一个可演进的对象"""
        return copy.deepcopy(self)


# 定义一个类型变量，表示特定类型的EvolvableSubjects
ASpecificEvolvableSubjects = TypeVar("ASpecificEvolvableSubjects", bound=EvolvableSubjects)


@dataclass
class EvoStep(Generic[ASpecificEvolvableSubjects]):
    """
    在一个特定的步骤中，
    基于
    - 先前的轨迹
    - 新的RAG知识 `QueriedKnowledge`

    `EvolvableSubjects` 会演进成一个新的 `EvolvableSubjects`。

    （可选）评估后，我们会得到反馈 `feedback`。
    """

    evolvable_subjects: ASpecificEvolvableSubjects  # 可演进的对象

    queried_knowledge: QueriedKnowledge | None = None  # 查询到的知识
    feedback: Feedback | None = None  # 反馈


class EvolvingStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    """演进策略的抽象基类"""
    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    @abstractmethod
    def evolve(
        self,
        *evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]] | None = None,
        queried_knowledge: QueriedKnowledge | None = None,
        **kwargs: Any,
    ) -> ASpecificEvolvableSubjects:
        """
        演进轨迹是一个按时间排序的（可演进对象，反馈）列表。

        这些参数对演进很重要的原因：
        - evolving_trace: 历史反馈很重要。
        - queried_knowledge: 查询到的知识。
        """


class RAGStrategy(ABC, Generic[ASpecificEvolvableSubjects]):
    """检索增强生成策略"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.knowledgebase: EvolvingKnowledgeBase = self.load_or_init_knowledge_base(*args, **kwargs)

    @abstractmethod
    def load_or_init_knowledge_base(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> EvolvingKnowledgeBase:
        """加载或初始化知识库"""
        pass

    @abstractmethod
    def query(
        self,
        evo: ASpecificEvolvableSubjects,
        evolving_trace: list[EvoStep],
        **kwargs: Any,
    ) -> QueriedKnowledge | None:
        """查询知识"""
        pass

    @abstractmethod
    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]],
        *,
        return_knowledge: bool = False,
        **kwargs: Any,
    ) -> Knowledge | None:
        """
        基于演进轨迹生成新知识。
        - 鼓励在生成新知识之前查询相关知识。

        RAGStrategy应自行维护新知识。
        """

    @abstractmethod
    def dump_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        """转储知识库"""
        pass

    @abstractmethod
    def load_dumped_knowledge_base(self, *args: Any, **kwargs: Any) -> None:
        """
        加载转储的知识库。
        这主要用于并行编码，其中多个编码器共享同一个知识库。
        在这种情况下，代理应在更新知识库之前从其他代理加载知识库。
        """
