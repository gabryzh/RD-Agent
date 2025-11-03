"""
具有会话控制的模型工作流
"""

import asyncio

import fire

from rdagent.app.qlib_rd_loop.conf import MODEL_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError


class ModelRDLoop(RDLoop):
    """模型研发循环"""
    skip_loop_error = (ModelEmptyError,)


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
):
    """
    金融科技模型的自动研发演进循环

    你可以通过以下方式继续运行会话：

    .. code-block:: python

        dotenv run -- python rdagent.app.qlib_rd_loop.model $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` 是一个可选参数

    :param path: 会话路径。
    :param step_n: 要运行的步骤数。
    :param loop_n: 要运行的循环数。
    :param all_duration: 总运行时间。
    :param checkout: 是否检出。
    """
    if path is None:
        model_loop = ModelRDLoop(MODEL_PROP_SETTING)
    else:
        model_loop = ModelRDLoop.load(path, checkout=checkout)
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
