"""
具有会话控制的因子工作流
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire
import typer
from typing_extensions import Annotated

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    """因子研发循环"""
    skip_loop_error = (FactorEmptyError,)

    def running(self, prev_out: dict[str, Any]):
        """
        运行因子研发循环的单个步骤。

        :param prev_out: 上一步的输出。
        :return: 实验结果。
        """
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error(f"因子提取失败。")
            raise FactorEmptyError("因子提取失败。")
        logger.log_object(exp, tag="运行器结果")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """
    金融科技因子的自动研发演进循环。

    你可以通过以下方式继续运行会话：

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` 是一个可选参数

    :param path: 会话路径。
    :param step_n: 要运行的步骤数。
    :param loop_n: 要运行的循环数。
    :param all_duration: 总运行时间。
    :param checkout: 是否检出。
    :param checkout_path: 检出路径。
    """
    if not checkout_path is None:
        checkout = Path(checkout_path)

    if path is None:
        model_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        model_loop = FactorRDLoop.load(path, checkout=checkout)
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
