from typing import Literal

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import (
    CondaConf,
    DockerEnv,
    DSDockerConf,
    Env,
    LocalEnv,
    MLEBDockerConf,
    MLECondaConf,
)


class DSCoderCoSTEERSettings(CoSTEERSettings):
    """数据科学 CoSTEER 设置"""

    class Config:
        env_prefix = "DS_Coder_CoSTEER_"

    max_seconds_multiplier: int = 4
    env_type: str = "docker"

    extra_evaluator: list[str] = []
    """要使用的额外评估器"""

    extra_eval: list[str] = []
    """
    额外的评估器。

    评估器遵循以下假设:
    - 它在前一个评估器之后运行（因此运行结果已经存在）。

    这不是一个完整的功能，因为它只在数据科学管道和编码器中实现。

    TODO: 完整版本应在 CoSTEERSettings 中实现。
    """


def get_ds_env(
    conf_type: Literal["kaggle", "mlebench"] = "kaggle",
    extra_volumes: dict = {},
    running_timeout_period: int | None = DS_RD_SETTING.debug_timeout,
    enable_cache: bool | None = None,
) -> Env:
    """
    根据 env_type 设置检索适当的环境配置。

    Returns:
        Env: 配置为 DockerEnv 或 LocalEnv 的环境实例。

    Raises:
        ValueError: 如果 env_type 无法识别。
    """
    conf = DSCoderCoSTEERSettings()
    assert conf_type in ["kaggle", "mlebench"], f"未知的 conf_type: {conf_type}"

    # 根据配置选择 Docker 或 Conda 环境
    if conf.env_type == "docker":
        env_conf = DSDockerConf() if conf_type == "kaggle" else MLEBDockerConf()
        env = DockerEnv(conf=env_conf)
    elif conf.env_type == "conda":
        env_conf = CondaConf(conda_env_name=conf_type) if conf_type == "kaggle" else MLECondaConf(conda_env_name=conf_type)
        env = LocalEnv(conf=env_conf)
    else:
        raise ValueError(f"未知的 env_type: {conf.env_type}")

    # 应用额外的配置
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache

    env.prepare()
    return env


def get_clear_ws_cmd(stage: Literal["before_training", "before_inference"] = "before_training") -> str:
    """
    获取用于将工作空间清理到特定阶段的命令。

    Args:
        stage: "before_training" 或 "before_inference"。

    Returns:
        str: 清理命令。
    """
    assert stage in ["before_training", "before_inference"], f"未知的阶段: {stage}"

    # 根据阶段和是否启用模型转储来确定要删除的文件
    if DS_RD_SETTING.enable_model_dump and stage == "before_training":
        cmd = "rm -rf submission.csv scores.csv models trace.log"
    else:
        cmd = "rm -f submission.csv scores.csv trace.log"
    return cmd
