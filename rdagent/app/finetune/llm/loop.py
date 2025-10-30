import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.finetune.llm.conf import update_settings
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    model: str | None = None,
    dataset: str | None = None,
):
    """
    用于模型微调的自动研发演进循环。

    参数：
        dataset (str): 用于微调的数据集名称。

    您可以使用以下命令继续运行一个会话：
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh
    """
    if not dataset:
        raise Exception("请指定数据集名称。")

    model_folder = Path(DS_RD_SETTING.local_data_path) / dataset / "prev_model"
    if not model_folder.exists():
        raise Exception(f"请将模型路径放置在 {model_folder}。")
    update_settings(dataset)
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
