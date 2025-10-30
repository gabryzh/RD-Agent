# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入ABC（抽象基类）和abstractmethod（抽象方法）
from abc import ABC, abstractmethod
# 导入Generator，用于定义生成器类型
from collections.abc import Generator
# 导入nullcontext，用于在不需要时提供一个空的上下文管理器
from contextlib import nullcontext
# 导入Any, Generic, TypeVar，用于类型提示
from typing import Any, Generic, TypeVar

# 导入FileLock，用于文件锁
from filelock import FileLock
# 导入tqdm，用于显示进度条
from tqdm import tqdm

# 从rdagent.core.evaluation模块导入EvaluableObj, Evaluator, Feedback
from rdagent.core.evaluation import EvaluableObj, Evaluator, Feedback
# 从rdagent.core.evolving_framework模块导入EvolvableSubjects, EvolvingStrategy, EvoStep
from rdagent.core.evolving_framework import EvolvableSubjects, EvolvingStrategy, EvoStep
# 从rdagent.log模块导入rdagent_logger
from rdagent.log import rdagent_logger as logger

# 定义一个类型变量，表示特定类型的Evaluator
ASpecificEvaluator = TypeVar("ASpecificEvaluator", bound=Evaluator)
# 定义一个类型变量，表示特定类型的EvolvableSubjects
ASpecificEvolvableSubjects = TypeVar("ASpecificEvolvableSubjects", bound=EvolvableSubjects)


# 定义一个名为EvoAgent的抽象基类
class EvoAgent(ABC, Generic[ASpecificEvaluator, ASpecificEvolvableSubjects]):

    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        """
        初始化EvoAgent。

        :param max_loop: 最大循环次数。
        :param evolving_strategy: 演进策略。
        """
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: ASpecificEvolvableSubjects,
        eva: ASpecificEvaluator | Feedback,
    ) -> Generator[ASpecificEvolvableSubjects, None, None]:
        """
        多步演进。

        为了便于调用者进行过程控制和日志记录，该方法会yield EvolvableSubjects。
        """


# 定义一个名为RAGEvaluator的抽象基类，继承自Evaluator
class RAGEvaluator(Evaluator):

    @abstractmethod
    def evaluate(
        self,
        eo: EvaluableObj,
        queried_knowledge: object = None,
    ) -> Feedback:
        """
        评估一个可评估对象。

        :param eo: 可评估对象。
        :param queried_knowledge: 查询到的知识。
        :return: 评估反馈。
        """
        raise NotImplementedError


# 定义一个名为RAGEvoAgent的类，继承自EvoAgent
class RAGEvoAgent(EvoAgent[RAGEvaluator, ASpecificEvolvableSubjects], Generic[ASpecificEvolvableSubjects]):

    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: Any,
        *,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
        enable_filelock: bool = False,
        filelock_path: str | None = None,
    ) -> None:
        """
        初始化RAGEvoAgent。

        :param max_loop: 最大循环次数。
        :param evolving_strategy: 演进策略。
        :param rag: RAG模型实例。
        :param with_knowledge: 是否使用知识。
        :param with_feedback: 是否使用反馈。
        :param knowledge_self_gen: 是否自生成知识。
        :param enable_filelock: 是否启用文件锁。
        :param filelock_path: 文件锁路径。
        """
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep[ASpecificEvolvableSubjects]] = []
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.enable_filelock = enable_filelock
        self.filelock_path = filelock_path

    def multistep_evolve(
        self,
        evo: ASpecificEvolvableSubjects,
        eva: RAGEvaluator | Feedback,
    ) -> Generator[ASpecificEvolvableSubjects, None, None]:
        """
        多步演进。

        :param evo: 可演进对象。
        :param eva: 评估器或反馈。
        :return: 一个生成器，用于yield可演进对象。
        """
        for evo_loop_id in tqdm(range(self.max_loop), "Implementing"):
            with logger.tag(f"evo_loop_{evo_loop_id}"):
                # 1. RAG
                queried_knowledge = None
                if self.with_knowledge and self.rag is not None:
                    # TODO: 将演进轨迹放在这里实际上不起作用
                    queried_knowledge = self.rag.query(evo, self.evolving_trace)

                # 2. 演进
                evo = self.evolving_strategy.evolve(
                    evo=evo,
                    evolving_trace=self.evolving_trace,
                    queried_knowledge=queried_knowledge,
                )

                # 3. 打包演进结果
                es = EvoStep[ASpecificEvolvableSubjects](evo, queried_knowledge)

                # 4. 评估
                if self.with_feedback:
                    es.feedback = (
                        eva if isinstance(eva, Feedback) else eva.evaluate(evo, queried_knowledge=queried_knowledge)
                    )
                    logger.log_object(es.feedback, tag="evolving feedback")

                # 5. 更新轨迹
                self.evolving_trace.append(es)

                # 6. 知识自演进
                if self.knowledge_self_gen and self.rag is not None:
                    with FileLock(self.filelock_path) if self.enable_filelock else nullcontext():  # type: ignore[arg-type]
                        self.rag.load_dumped_knowledge_base()
                        self.rag.generate_knowledge(self.evolving_trace)
                        self.rag.dump_knowledge_base()

                yield evo  # 将控制权yield给调用者，以便进行过程控制和日志记录。

                # 7. 检查所有任务是否完成
                if self.with_feedback and es.feedback is not None and es.feedback.finished():
                    logger.info("演进主题中的所有任务都已完成。")
                    break
