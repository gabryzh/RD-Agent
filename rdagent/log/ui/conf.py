from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class UIBasePropSetting(ExtendedBaseSettings):
    """UI 基础属性设置"""
    model_config = SettingsConfigDict(env_prefix="UI_", protected_namespaces=())

    # 默认的日志文件夹列表
    default_log_folders: list[str] = ["./log"]

    # 基线结果文件路径
    baseline_result_path: str = "./baseline.csv"

    # AIDE 相关文件的路径
    aide_path: str = "./aide"

    # AMLT 相关文件的路径
    amlt_path: str = "/data/share_folder_local/amlt"

    # 静态文件（如 CSS, JS）的路径
    static_path: str = "./git_ignore_folder/static"

    # 追踪日志文件夹的路径
    trace_folder: str = "./traces"

    # 是否启用缓存
    enable_cache: bool = True


# 创建一个全局的 UI 设置实例
UI_SETTING = UIBasePropSetting()
