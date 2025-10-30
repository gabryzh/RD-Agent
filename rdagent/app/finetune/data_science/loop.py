import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.finetune.data_science.conf import update_settings
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    model: str | None = None,
    competition: str | None = None,
):
    """
    用于模型微调的自动研发演进循环。

    参数：
        competition (str): 竞赛名称。

    您可以使用以下命令继续运行一个会话：
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/data_science/loop.py --competition aerial-cactus-identification
    """
    if not competition:
        raise Exception("请指定竞赛名称。")

    model_folder = Path(DS_RD_SETTING.local_data_path) / competition / "prev_model"
    if not model_folder.exists():
        raise Exception(f"请将模型路径放置在 {model_folder}。")
    update_settings(competition)
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
