import os

from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS, ExtendedBaseSettings


class LLMFinetuneScen(ExtendedBaseSettings):
    """LLM微调场景的配置类"""
    model_config = SettingsConfigDict(env_prefix="FT_", protected_namespaces=())
    scen: str = "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    """
    数据科学任务的场景类。
    - 对于Kaggle竞赛，使用: "rdagent.scenarios.data_science.scen.KaggleScen"
    - 对于自定义数据科学场景，使用: "rdagent.scenarios.data_science.scen.DataScienceScen"
    - 对于LLM微调场景，使用: "rdagent.app.finetune.llm.scen.LLMFinetuneScen"
    - 对于数据科学微调场景，使用: "rdagent.app.finetune.data_science.scen.DSFinetuneScen"
    """

    hypothesis_gen: str = "rdagent.app.finetune.llm.proposal.FinetuneExpGen"
    """假设生成类"""

    debug_timeout: int = 36000
    """在调试数据上运行的超时限制"""
    full_timeout: int = 360000
    """在完整数据上运行的超时限制"""

    coder_on_whole_pipeline: bool = True
    enable_model_dump: bool = True
    app_tpl: str = "app/finetune/llm/tpl"


def update_settings(competition: str):
    """
    使用LLM_FINETUNE_SETTINGS中的值更新RD_AGENT_SETTINGS。
    """
    LLM_FINETUNE_SETTINGS = LLMFinetuneScen()
    RD_AGENT_SETTINGS.app_tpl = LLM_FINETUNE_SETTINGS.app_tpl
    os.environ["DS_CODER_COSTEER_EXTRA_EVALUATOR"] = '["rdagent.app.finetune.share.eval.PrevModelLoadEvaluator"]'
    for field_name, new_value in LLM_FINETUNE_SETTINGS.model_dump().items():
        if hasattr(DS_RD_SETTING, field_name):
            setattr(DS_RD_SETTING, field_name, new_value)
    DS_RD_SETTING.competition = competition
