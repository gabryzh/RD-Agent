from typing import Union

from rdagent.core.conf import ExtendedBaseSettings


class CoSTEERSettings(ExtendedBaseSettings):
    """
    CoSTEER 的设置。这个设置类不应该被直接使用！！！
    它作为因子和模型特定设置的基类。
    """

    class Config:
        # Pydantic 配置，用于从环境变量加载设置
        env_prefix = "CoSTEER_"

    coder_use_cache: bool = False
    """指示编码器是否使用缓存"""

    max_loop: int = 10
    """任务实现的最大循环次数"""

    fail_task_trial_limit: int = 20
    """单个任务失败的尝试次数上限"""

    # V1 知识查询限制
    v1_query_former_trace_limit: int = 3
    """V1: 查询历史失败记录的数量限制"""
    v1_query_similar_success_limit: int = 3
    """V1: 查询相似成功案例的数量限制"""

    # V2 知识查询与功能开关
    v2_query_component_limit: int = 1
    """V2: 查询组件（component）知识的数量限制"""
    v2_query_error_limit: int = 1
    """V2: 查询错误知识的数量限制"""
    v2_query_former_trace_limit: int = 3
    """V2: 查询历史失败记录的数量限制"""
    v2_add_fail_attempt_to_latest_successful_execution: bool = False
    """V2: 是否将失败尝试添加到最近的成功执行中"""
    v2_error_summary: bool = False
    """V2: 是否启用错误总结功能"""
    v2_knowledge_sampler: float = 1.0
    """V2: 知识采样率"""

    knowledge_base_path: Union[str, None] = None
    """知识库的路径"""

    new_knowledge_base_path: Union[str, None] = None
    """新知识库的路径"""

    enable_filelock: bool = False
    """是否启用文件锁"""
    filelock_path: Union[str, None] = None
    """文件锁的路径"""

    max_seconds_multiplier: int = 10**6
    """最大秒数乘数因子"""


# 创建一个全局的 CoSTEER 设置实例
CoSTEER_SETTINGS = CoSTEERSettings()
