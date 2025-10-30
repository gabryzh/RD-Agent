from copy import deepcopy
from datetime import datetime
from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from rdagent.core.developer import Developer
from rdagent.core.evolving_agent import EvolvingStrategy, RAGEvaluator, RAGEvoAgent
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import RD_Agent_TIMER_wrapper


class CoSTEER(Developer[Experiment]):
    """
    CoSTEER (Code evolution with STrEER) 框架的核心实现。
    它是一个通用的开发者类，通过演进的方式来完成开发任务。
    """
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: RAGEvaluator,
        es: EvolvingStrategy,
        *args,
        evolving_version: int = 2,
        with_knowledge: bool = True,
        knowledge_self_gen: bool = True,
        max_loop: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings

        self.max_loop = settings.max_loop if max_loop is None else max_loop
        self.knowledge_base_path = Path(settings.knowledge_base_path) if settings.knowledge_base_path else None
        self.new_knowledge_base_path = Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path else None

        self.with_knowledge = with_knowledge
        self.knowledge_self_gen = knowledge_self_gen
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # 根据演进版本初始化 RAG (Retrieval-Augmented Generation) 策略
        self.rag = (
            CoSTEERRAGStrategyV2(settings=settings, former_knowledge_base_path=self.knowledge_base_path, dump_knowledge_base_path=self.new_knowledge_base_path, evolving_version=self.evolving_version)
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(settings=settings, former_knowledge_base_path=self.knowledge_base_path, dump_knowledge_base_path=self.new_knowledge_base_path, evolving_version=self.evolving_version)
        )

    def get_develop_max_seconds(self) -> int | None:
        """获取开发任务的最大秒数。子类可以重写此方法。"""
        return None

    def _get_last_fb(self) -> CoSTEERMultiFeedback:
        """获取演进轨迹中的最后一个反馈。"""
        fb = self.evolve_agent.evolving_trace[-1].feedback
        assert fb is not None, "反馈不应为None"
        assert isinstance(fb, CoSTEERMultiFeedback), "反馈必须是 CoSTEERMultiFeedback 类型"
        return fb

    def should_use_new_evo(self, base_fb: CoSTEERMultiFeedback | None, new_fb: CoSTEERMultiFeedback) -> bool:
        """
        比较新旧反馈，判断是否应采用新的演进结果作为备用方案。
        目前逻辑比较简单：只要新的反馈是可接受的（例如，没有导致程序崩溃），就采用。
        """
        return new_fb is not None and new_fb.is_acceptable()

    def develop(self, exp: Experiment) -> Experiment:
        """
        核心开发方法，通过多轮演进生成最终的代码实现。
        """
        max_seconds = self.get_develop_max_seconds()
        evo_exp = EvolvingItem.from_experiment(exp)

        self.evolve_agent = RAGEvoAgent[EvolvingItem](
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=True,
            knowledge_self_gen=self.knowledge_self_gen,
            enable_filelock=self.settings.enable_filelock,
            filelock_path=self.settings.filelock_path,
        )

        # 开始演进循环
        start_datetime = datetime.now()
        fallback_evo_exp = None  # 存储最佳的备用解决方案
        fallback_evo_fb = None
        reached_max_seconds = False

        for evo_exp in self.evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            evo_fb = self._get_last_fb()

            # 判断是否更新备用方案
            if self.should_use_new_evo(base_fb=fallback_evo_fb, new_fb=evo_fb):
                fallback_evo_exp = deepcopy(evo_exp)
                fallback_evo_fb = deepcopy(evo_fb)
                fallback_evo_exp.create_ws_ckp()  # 创建工作空间快照以防被修改

            # 检查是否超时
            if max_seconds and (datetime.now() - start_datetime).total_seconds() > max_seconds:
                logger.info(f"达到最大时间限制 {max_seconds} 秒，停止演进")
                reached_max_seconds = True
                break
            if RD_Agent_TIMER_wrapper.timer.started and RD_Agent_TIMER_wrapper.timer.is_timeout():
                logger.info("全局计时器超时，停止演进")
                break

        try:
            # 循环结束后，回退到最佳的备用解决方案
            if fallback_evo_exp is not None:
                logger.info("回退到备用解决方案。")
                evo_exp = fallback_evo_exp
                evo_exp.recover_ws_ckp() # 恢复工作空间快照
                evo_fb = fallback_evo_fb

            assert evo_fb is not None, "演进循环至少应运行一次"
            evo_exp = self._exp_postprocess_by_feedback(evo_exp, evo_fb)
        except CoderError as e:
            e.caused_by_timeout = reached_max_seconds
            raise e

        # 将最终的工作空间和代码更新回原始的实验对象
        exp.sub_workspace_list = evo_exp.sub_workspace_list
        exp.experiment_workspace = evo_exp.experiment_workspace
        return exp

    def _exp_postprocess_by_feedback(self, evo: Experiment, feedback: CoSTEERMultiFeedback) -> Experiment:
        """
        根据最终的反馈对实验进行后处理。
        如果所有任务都失败了，则抛出异常。
        """
        assert isinstance(evo, Experiment)
        assert isinstance(feedback, CoSTEERMultiFeedback)
        assert len(evo.sub_workspace_list) == len(feedback)

        failed_feedbacks = [
            f"- 反馈{index + 1:02d}:\n  - 执行: {f.execution}\n  - 返回值检查: {f.return_checking}\n  - 代码: {f.code}"
            for index, f in enumerate(feedback)
            if f is not None and not f.is_acceptable()
        ]

        if len(failed_feedbacks) == len(feedback):
            feedback_summary = "\n".join(failed_feedbacks)
            raise CoderError(f"所有任务都失败了:\n{feedback_summary}")

        return evo
