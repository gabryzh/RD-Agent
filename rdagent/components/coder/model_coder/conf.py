from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import Env, QlibCondaConf, QlibCondaEnv, QTDockerEnv


class ModelCoSTEERSettings(CoSTEERSettings):
    """
    模型 CoSTEER 的特定配置。
    继承自通用的 CoSTEERSettings。
    """
    # 设置 pydantic 模型从环境变量加载时的前缀
    model_config = SettingsConfigDict(env_prefix="MODEL_COSTEER_")

    # 环境类型，可以是 "conda" 或 "docker"
    env_type: str = "conda"
    """用于在 coder 和 runner 中运行模型代码的环境：
    'conda' 表示本地 conda 环境, 'docker' 表示 Docker 容器"""


def get_model_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    """
    获取并准备用于运行模型代码的环境。

    Args:
        conf_type (Optional[str]): 配置类型（当前未使用）。
        extra_volumes (dict): 需要挂载到环境中的额外卷（主要用于 Docker）。
        running_timeout_period (int): 代码运行的超时时间（秒）。
        enable_cache (Optional[bool]): 是否启用缓存。如果为 None，则使用环境的默认设置。

    Returns:
        Env: 准备好的环境对象。

    Raises:
        ValueError: 如果配置中的 env_type 无效。
    """
    # 加载模型 CoSTEER 的配置
    conf = ModelCoSTEERSettings()

    # 根据配置的 env_type 创建相应的环境实例
    if conf.env_type == "docker":
        env = QTDockerEnv()
    elif conf.env_type == "conda":
        env = QlibCondaEnv(conf=QlibCondaConf())
    else:
        raise ValueError(f"未知的环境类型: {conf.env_type}")

    # 更新环境配置
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache

    # 准备环境（例如，启动 Docker 容器或激活 conda 环境）
    env.prepare()
    return env


# 创建一个全局的配置实例，以便在其他地方导入和使用
MODEL_COSTEER_SETTINGS = ModelCoSTEERSettings()
