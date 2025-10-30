# 该文件遵循Python 3.9的语法，并启用了 postponed evaluation of type annotations (PEP 563)
from __future__ import annotations

# 导入Path类，用于处理文件系统路径
from pathlib import Path
# 导入cast，用于类型转换
from typing import cast

# 从pydantic-settings导入所需类，用于管理配置
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)

# 定义一个扩展的基础设置类
class ExtendedBaseSettings(BaseSettings):

    # 定义一个类方法，用于自定义设置源的加载顺序
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 1) 从基类开始遍历
        def base_iter(settings_cls: type[ExtendedBaseSettings]) -> list[type[ExtendedBaseSettings]]:
            bases = []
            for cl in settings_cls.__bases__:
                if issubclass(cl, ExtendedBaseSettings) and cl is not ExtendedBaseSettings:
                    bases.append(cl)
                    bases.extend(base_iter(cl))
            return bases

        # 2) 从基类构建EnvSettingsSource，以便添加父级的环境变量源
        parent_env_settings = [
            EnvSettingsSource(
                base_cls,
                case_sensitive=base_cls.model_config.get("case_sensitive"),
                env_prefix=base_cls.model_config.get("env_prefix"),
                env_nested_delimiter=base_cls.model_config.get("env_nested_delimiter"),
            )
            for base_cls in base_iter(cast("type[ExtendedBaseSettings]", settings_cls))
        ]
        # 返回所有设置源，按优先级排序
        return init_settings, env_settings, *parent_env_settings, dotenv_settings, file_secret_settings


class RDAgentSettings(ExtendedBaseSettings):

    # azure document intelligence 配置
    azure_document_intelligence_key: str = ""  # Azure Document Intelligence 的密钥
    azure_document_intelligence_endpoint: str = ""  # Azure Document Intelligence 的终端节点

    # 因子提取配置
    max_input_duplicate_factor_group: int = 300  # 最大输入重复因子组
    max_output_duplicate_factor_group: int = 20  # 最大输出重复因子组
    max_kmeans_group_number: int = 40  # K-Means聚类的最大组数

    # 工作区配置
    workspace_path: Path = Path.cwd() / "git_ignore_folder" / "RD-Agent_workspace"  # 工作区的路径
    workspace_ckp_size_limit: int = 0  # 工作区检查点大小限制（0表示无限制）
    workspace_ckp_white_list_names: list[str] | None = None  # 工作区检查点文件白名单
    """
    工作区的检查点是一个zip文件。
    0（或任何小于等于0的值）表示对工作区检查点中的文件大小没有限制。
    """

    # 多进程配置
    multi_proc_n: int = 1  # 进程数

    # pickle缓存配置
    cache_with_pickle: bool = True  # 是否使用pickle缓存
    pickle_cache_folder_path_str: str = str(
        Path.cwd() / "pickle_cache/",
    )  # pickle缓存文件夹的路径
    use_file_lock: bool = (
        True  # 调用相同参数的函数时，是否使用文件锁避免重复执行
    )

    # 其他配置
    """上下文标准输出的限制"""
    stdout_context_len: int = 400  # 标准输出上下文长度
    stdout_line_len: int = 10000  # 标准输出行长度

    enable_mlflow: bool = False  # 是否启用MLflow

    initial_fator_library_size: int = 20  # 初始因子库大小

    # 并行循环配置
    step_semaphore: int | dict[str, int] = 1  # 每个步骤的信号量
    """每个步骤的信号量；可以指定一个总信号量
    或一个分步骤的信号量，如 {"coding": 3, "running": 2}"""

    def get_max_parallel(self) -> int:
        """根据信号量设置，返回最大并行循环数"""
        if isinstance(self.step_semaphore, int):
            return self.step_semaphore
        return max(self.step_semaphore.values())

    # 注意：用于调试
    # 以下函数仅用于调试，在主逻辑中是必需的。
    subproc_step: bool = False  # 是否在子进程中执行步骤

    def is_force_subproc(self) -> bool:
        """判断是否强制在子进程中执行"""
        return self.subproc_step or self.get_max_parallel() > 1

    # 模板配置
    app_tpl: str | None = None  # 用于应用程序覆盖默认模板，例如："app/fintune/tpl"


# 创建RDAgentSettings的实例
RD_AGENT_SETTINGS = RDAgentSettings()
