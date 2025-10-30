from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.utils.agent.tpl import T


class LLMFinetuneScen(DataScienceScen):
    """LLM微调场景"""

    def __init__(self, competition: str) -> None:
        """初始化LLM微调场景"""
        self._download_data(competition=competition)
        super().__init__(competition)
        self._analysis_competition_description()

    def _get_data_folder_description(self) -> str:
        """获取数据文件夹的描述"""
        folder_desc = describe_data_folder_v2(
            Path(DS_RD_SETTING.local_data_path) / self.competition, show_nan_columns=DS_RD_SETTING.show_nan_columns
        )
        return folder_desc

    def _download_data(self, competition: str):
        """
        从Hugging Face Hub下载数据集。
        :param competition: 数据集ID，例如 "shibing624/alpaca-zh"。
        """
        save_path = f"{DS_RD_SETTING.local_data_path}/{competition}"
        if Path(save_path).exists():
            logger.info(f"{save_path} 已存在。")
        else:
            logger.info(f"正在下载 {competition} 到 {save_path}")
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=competition,
                    repo_type="dataset",
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                )
            except ImportError:
                raise ImportError(
                    "请先安装huggingface_hub。"
                    '您可以使用 `pip install -U "huggingface_hub[cli]"` 进行安装。'
                )
            except Exception as e:
                logger.error(f"下载 {competition} 时出错: {e}")
                raise e

    def _get_description(self):
        """获取数据集的描述"""
        if (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}/README.md")).exists():
            logger.info(f"{self.competition}/找到README.md，从本地文件加载。")
            return fp.read_text()

    def _get_direction(self):
        """获取方向"""
        return True

    @property
    def rich_style_description(self) -> str:
        """富文本样式的描述"""
        raise NotImplementedError

    @property
    def background(self) -> str:
        """获取背景信息"""
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            raw_description=self.raw_description,
        )
        return background_prompt

    def get_competition_full_desc(self) -> str:
        """获取竞赛的完整描述"""
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """
        获取场景的所有描述。
        eda_output依赖于当前工作区中的动态.md文件，不是固定的。
        """
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )
