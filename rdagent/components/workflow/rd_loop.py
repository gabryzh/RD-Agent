"""
带会话控制的模型工作流。
该模块源自 `rdagent/app/qlib_rd_loop/model.py`，旨在替代 `rdagent/app/qlib_rd_loop/RDAgent.py`。
"""

import asyncio
from typing import Any

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.utils.workflow import LoopBase, LoopMeta


class RDLoop(LoopBase, metaclass=LoopMeta):
    """
    研发循环（Research and Development Loop）的核心实现。
    该类继承自 LoopBase，并使用 LoopMeta 元类来构建一个可执行的工作流。
    """

    def __init__(self, PROP_SETTING: BasePropSetting):
        """
        初始化 RDLoop。

        Args:
            PROP_SETTING (BasePropSetting): 包含工作流配置属性的设置对象。
        """
        # 动态导入并实例化场景类
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        # 记录 RDLoop 和全局 RD Agent 的设置
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        # 动态导入并实例化各个组件
        # 假设生成器
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
        # 假设到实验的转换器
        self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
        # 编码器（实现实验）
        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        # 运行器（执行实验）
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
        # 总结器（从实验结果生成反馈）
        self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)

        # 初始化追踪对象，用于记录历史信息
        self.trace = Trace(scen=scen)
        super().__init__()

    # --- 内部方法，不直接作为工作流步骤 ---

    def _propose(self) -> Hypothesis:
        """
        生成一个新的假设。

        Returns:
            Hypothesis: 生成的假设对象。
        """
        hypothesis = self.hypothesis_gen.gen(self.trace)
        logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def _exp_gen(self, hypothesis: Hypothesis):
        """
        将假设转换为实验。

        Args:
            hypothesis (Hypothesis): 输入的假设。

        Returns:
            Experiment: 生成的实验对象。
        """
        exp = self.hypothesis2experiment.convert(hypothesis, self.trace)
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    # --- 工作流步骤 ---

    async def direct_exp_gen(self, prev_out: dict[str, Any]) -> dict[str, Any]:
        """
        直接生成实验（包含假设生成和实验转换）。
        该步骤会检查当前的并行任务数量，并在达到上限时等待。
        """
        while True:
            # 如果未完成的循环数小于最大并行数，则继续
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                exp = self._exp_gen(hypo)
                return {"propose": hypo, "exp_gen": exp}
            # 否则，异步等待1秒
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """
        编码步骤：根据实验设计，生成具体的代码实现。
        """
        exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """
        运行步骤：执行生成的代码，并获取实验结果。
        """
        exp = self.runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """
        反馈步骤：根据实验结果生成反馈，并更新追踪历史。
        """
        # 检查上一步是否有异常
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            # 如果有异常，生成包含错误信息的反馈
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="feedback")
            # 将实验和失败的反馈记录到历史中
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            # 如果没有异常，使用总结器生成反馈
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="feedback")
            # 将实验和成功的反馈记录到历史中
            self.trace.hist.append((prev_out["running"], feedback))

    # TODO: `def record(self, prev_out: dict[str, Any]):` 已硬编码到 LoopBase 中。
    # 因此，我们应将其添加到 RDLoop 类中，以确保每个 RDLoop 子类都能意识到它的存在。
