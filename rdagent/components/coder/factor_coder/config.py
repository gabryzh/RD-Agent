import os
from typing import Optional

from pantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import CondaConf, Env, LocalEnv


class FactorCoSTEERSettings(CoSTEERSettings):
    """
    因子 CoSTEER 的特定配置。
    """
    # 设置 pydantic 模型从环境变量加载时的前缀
    model_config = SettingsConfigDict(env_prefix="FACTOR_COSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """包含金融数据的文件夹路径（默认为 Qlib 中的基本面数据）"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """包含部分金融数据的文件夹路径（用于调试）"""

    simple_background: bool = False
    """是否为代码反馈使用简单的背景信息"""

    file_based_execution_timeout: int = 3600
    """每个因子实现执行的超时时间（秒）"""

    select_method: str = "random"
    """因子实现的选择方法"""

    python_bin: str = "python"
    """Python 可执行文件的路径"""


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    """
    获取并准备用于运行因子代码的环境。

    Args:
        conf_type (Optional[str]): 配置类型（当前未使用）。
        extra_volumes (dict): 额外卷（当前未使用，为兼容性保留）。
        running_timeout_period (int): 代码运行的超时时间（秒）。
        enable_cache (Optional[bool]): 是否启用缓存。

    Returns:
        Env: 准备好的环境对象。
    """
    conf = FactorCoSTEERSettings()
    # 如果配置了 python_bin，则使用本地 conda 环境
    if hasattr(conf, "python_bin"):
        # 使用当前激活的 conda 环境
        env = LocalEnv(conf=(CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))))

    # 更新环境配置
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache

    # 准备环境
    env.prepare()
    return env


# 创建一个全局的配置实例
FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
