"""
量化（因子和模型）工作流与会话控制
"""

import asyncio
from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace


class QuantRDLoop(RDLoop):
    """量化研发循环"""
    skip_loop_error = (
        FactorEmptyError,
        ModelEmptyError,
    )

    def __init__(self, PROP_SETTING: BasePropSetting):
        """
        初始化QuantRDLoop。

        :param PROP_SETTING: 基础属性设置。
        """
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="场景")

        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="量化假设生成器")

        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.factor_hypothesis2experiment
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="因子假设到实验转换器")
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.model_hypothesis2experiment
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="模型假设到实验转换器")

        self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="因子编码器")
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="模型编码器")

        self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="因子运行器")
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="模型运行器")

        self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="因子摘要器")
        self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="模型摘要器")

        self.trace = QuantTrace(scen=scen)
        super(RDLoop, self).__init__()

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """
        直接从假设生成实验。

        :param prev_out: 上一步的输出。
        :return: 包含假设和实验的字典。
        """
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                assert hypo.action in ["factor", "model"]
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                logger.log_object(exp.sub_tasks, tag="实验生成")
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """
        编码步骤。

        :param prev_out: 上一步的输出。
        :return: 编码后的实验。
        """
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp, tag="编码器结果")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """
        运行步骤。

        :param prev_out: 上一步的输出。
        :return: 运行后的实验。
        """
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_runner.develop(prev_out["coding"])
            if exp is None:
                logger.error(f"因子提取失败。")
                raise FactorEmptyError("因子提取失败。")
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="运行器结果")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """
        反馈步骤。

        :param prev_out: 上一步的输出。
        """
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="反馈")
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="反馈")
            self.trace.hist.append((prev_out["running"], feedback))


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
):
    """
    金融科技因子的自动研发演进循环。
    你可以通过以下方式继续运行会话：
    .. code-block:: python
        dotenv run -- python rdagent/app/qlib_rd_loop/quant.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` 是一个可选参数

    :param path: 会话路径。
    :param step_n: 要运行的步骤数。
    :param loop_n: 要运行的循环数。
    :param all_duration: 总运行时间。
    :param checkout: 是否检出。
    """
    if path is None:
        quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
    else:
        quant_loop = QuantRDLoop.load(path, checkout=checkout)

    asyncio.run(quant_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
