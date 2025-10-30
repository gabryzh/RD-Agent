from __future__ import annotations

import copy
import json
import pickle
import random
import re
from itertools import combinations
from pathlib import Path
from typing import List, Union

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from rdagent.core.evolving_agent import Feedback
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvolvingKnowledgeBase,
    EvoStep,
    Knowledge,
    QueriedKnowledge,
    RAGStrategy,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import (
    APIBackend,
    calculate_embedding_distance_between_str_list,
)
from rdagent.utils.agent.tpl import T


class CoSTEERKnowledge(Knowledge):
    """封装单个CoSTEER知识单元，包含任务、实现和反馈。"""
    def __init__(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        feedback: Feedback,
    ) -> None:
        self.target_task = target_task
        self.implementation = implementation.copy()
        self.feedback = feedback

    def get_implementation_and_feedback_str(self) -> str:
        """获取实现代码和反馈的格式化字符串。"""
        return f"""------------------实现代码:------------------
{self.implementation.all_codes}
------------------实现反馈:------------------
{self.feedback!s}
"""


class CoSTEERRAGStrategy(RAGStrategy):
    """CoSTEER的检索增强生成（RAG）策略基类。"""
    def __init__(self, *args, dump_knowledge_base_path: Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump_knowledge_base_path = dump_knowledge_base_path

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, evolving_version: int = 2) -> EvolvingKnowledgeBase:
        """加载或初始化知识库，并检查版本兼容性。"""
        if former_knowledge_base_path and former_knowledge_base_path.exists():
            with open(former_knowledge_base_path, "rb") as f:
                knowledge_base = pickle.load(f)
            if (evolving_version == 1 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV1)) or \
               (evolving_version == 2 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV2)):
                raise ValueError("知识库版本不兼容。")
        else:
            knowledge_base = CoSTEERKnowledgeBaseV2() if evolving_version == 2 else CoSTEERKnowledgeBaseV1()
        return knowledge_base

    def dump_knowledge_base(self):
        """将当前知识库序列化到文件。"""
        if self.dump_knowledge_base_path:
            self.dump_knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dump_knowledge_base_path, "wb") as f:
                pickle.dump(self.knowledgebase, f)
        else:
            logger.warning("未设置知识库转储路径，跳过操作。")

    def load_dumped_knowledge_base(self):
        """从文件加载序列化的知识库。"""
        if self.dump_knowledge_base_path and self.dump_knowledge_base_path.exists():
            with open(self.dump_knowledge_base_path, "rb") as f:
                self.knowledgebase = pickle.load(f)
            logger.info(f"已从 {self.dump_knowledge_base_path} 加载知识库。")


class CoSTEERQueriedKnowledge(QueriedKnowledge):
    """封装查询到的CoSTEER知识的基类。"""
    def __init__(self, success_task_to_knowledge_dict: dict = {}, failed_task_info_set: set = set()) -> None:
        self.success_task_to_knowledge_dict = success_task_to_knowledge_dict
        self.failed_task_info_set = failed_task_info_set


class CoSTEERKnowledgeBaseV1(EvolvingKnowledgeBase):
    """V1版本的CoSTEER知识库。"""
    def __init__(self, path: str | Path = None) -> None:
        super().__init__(path)
        self.implementation_trace: dict[str, List[CoSTEERKnowledge]] = {}
        self.success_task_info_set: set[str] = set()


class CoSTEERQueriedKnowledgeV1(CoSTEERQueriedKnowledge):
    """封装V1版本查询到的知识。"""
    def __init__(self, *args, task_to_former_failed_traces: dict = {}, task_to_similar_task_successful_knowledge: dict = {}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_to_former_failed_traces = task_to_former_failed_traces
        self.task_to_similar_task_successful_knowledge = task_to_similar_task_successful_knowledge


class CoSTEERRAGStrategyV1(CoSTEERRAGStrategy):
    """V1版本的RAG策略（已不推荐使用）。"""
    def __init__(self, settings: CoSTEERSettings, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(self, evolving_trace: list[EvoStep], *, return_knowledge: bool = False) -> None:
        """从演进轨迹中生成并存储知识。"""
        raise NotImplementedError("V1版本已不推荐使用，请使用V2。")

    def query(self, evo: EvolvableSubjects, evolving_trace: list[EvoStep]) -> CoSTEERQueriedKnowledgeV1:
        """根据当前演进主体查询知识库。"""
        raise NotImplementedError("V1版本已不推荐使用，请使用V2。")


class CoSTEERQueriedKnowledgeV2(CoSTEERQueriedKnowledgeV1):
    """封装V2版本查询到的知识，增加了相似错误知识。"""
    def __init__(self, *args, task_to_similar_error_successful_knowledge: dict = {}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_to_similar_error_successful_knowledge = task_to_similar_error_successful_knowledge


class CoSTEERRAGStrategyV2(CoSTEERRAGStrategy):
    """V2版本的RAG策略，使用知识图谱进行更复杂的知识管理。"""
    def __init__(self, settings: CoSTEERSettings, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(self, evolving_trace: list[EvoStep], *, return_knowledge: bool = False) -> None:
        """从演进轨迹中生成知识，并更新知识图谱。"""
        if len(evolving_trace) == self.current_generated_trace_count:
            return

        for trace_index in range(self.current_generated_trace_count, len(evolving_trace)):
            evo_step = evolving_trace[trace_index]
            # ... 省略了复杂的知识生成和图更新逻辑 ...
        self.current_generated_trace_count = len(evolving_trace)

    def query(self, evo: EvolvableSubjects, evolving_trace: list[EvoStep]) -> CoSTEERQueriedKnowledgeV2:
        """执行多方面的知识查询：历史轨迹、组件和错误。"""
        queried_knowledge = CoSTEERQueriedKnowledgeV2(success_task_to_knowledge_dict=self.knowledgebase.success_task_to_knowledge_dict)
        queried_knowledge = self.former_trace_query(evo, queried_knowledge)
        queried_knowledge = self.component_query(evo, queried_knowledge)
        queried_knowledge = self.error_query(evo, queried_knowledge)
        return queried_knowledge

    def analyze_component(self, target_task_information: str) -> list[UndirectedNode]:
        """使用LLM分析任务描述，关联到已有的组件节点。"""
        # ... 省略LLM调用和节点匹配逻辑 ...
        return []

    def analyze_error(self, single_feedback: str, feedback_type="execution") -> list[Union[UndirectedNode, str]]:
        """从反馈中提取错误类型和信息。"""
        # ... 省略错误信息解析逻辑 ...
        return []

    def former_trace_query(self, evo: EvolvableSubjects, queried_knowledge_v2: CoSTEERQueriedKnowledgeV2) -> CoSTEERQueriedKnowledgeV2:
        """查询任务的历史失败轨迹。"""
        # ... 省略历史轨迹查询逻辑 ...
        return queried_knowledge_v2

    def component_query(self, evo: EvolvableSubjects, queried_knowledge_v2: CoSTEERQueriedKnowledgeV2) -> CoSTEERQueriedKnowledgeV2:
        """基于组件关联查询成功的实现案例。"""
        # ... 省略基于图的组件知识查询逻辑 ...
        return queried_knowledge_v2

    def error_query(self, evo: EvolvableSubjects, queried_knowledge_v2: CoSTEERQueriedKnowledgeV2) -> CoSTEERQueriedKnowledgeV2:
        """基于当前错误查询相似错误的解决方案。"""
        # ... 省略基于图的错误知识查询逻辑 ...
        return queried_knowledge_v2


class CoSTEERKnowledgeBaseV2(EvolvingKnowledgeBase):
    """V2版本的CoSTEER知识库，核心是一个无向图。"""
    def __init__(self, init_component_list: list = None, path: str | Path = None) -> None:
        self.graph: UndirectedGraph = UndirectedGraph(Path.cwd() / "graph.pkl")
        logger.info(f"CoSTEER知识图谱已加载, 大小={self.graph.size()}")

        # ... 省略了图更新、查询等多种复杂方法 ...
        # 这些方法共同构成了基于知识图谱的知识管理和推理能力。
