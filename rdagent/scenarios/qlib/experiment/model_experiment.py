# 从 copy 模块导入 deepcopy 函数，用于创建对象的深拷贝
from copy import deepcopy
# 从 pathlib 模块导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 从 rdagent 的模型编码器配置中导入 get_model_env 函数
from rdagent.components.coder.model_coder.conf import get_model_env
# 从 rdagent 的模型编码器模型模块导入相关类
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
# 从 rdagent 核心实验模块导入 Task 类
from rdagent.core.experiment import Task
# 从 rdagent 核心场景模块导入 Scenario 类
from rdagent.core.scenario import Scenario
# 从 qlib 实验工作区模块导入 QlibFBWorkspace 类
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
# 从共享模块导入 get_runtime_environment_by_env 函数
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
# 从 rdagent 的模板工具模块导入 T
from rdagent.utils.agent.tpl import T


class QlibModelExperiment(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    """
    Qlib 模型实验类，继承自通用的模型实验类 ModelExperiment。
    它专门为 Qlib 场景定制，使用 QlibFBWorkspace 作为其工作区。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始化实验工作区，使用 "model_template" 目录下的模板
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")
        # 初始化标准输出为空字符串
        self.stdout = ""


class QlibModelScenario(Scenario):
    """
    Qlib 模型场景类，定义了模型开发任务的背景、接口、格式等信息。
    这些信息会用于生成提示（prompt），引导大语言模型完成任务。
    """
    def __init__(self) -> None:
        super().__init__()
        # 使用模板和运行时环境信息，初始化场景的各个组成部分
        self._background = deepcopy(
            T(".prompts:qlib_model_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        self._output_format = deepcopy(T(".prompts:qlib_model_output_format").r())
        self._interface = deepcopy(T(".prompts:qlib_model_interface").r())
        self._simulator = deepcopy(T(".prompts:qlib_model_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:qlib_model_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:qlib_model_experiment_setting").r())

    @property
    def background(self) -> str:
        """返回场景背景信息"""
        return self._background

    @property
    def source_data(self) -> str:
        """Qlib 模型场景的源数据未实现"""
        raise NotImplementedError("QlibModelScenario 的 source_data 尚未实现")

    @property
    def output_format(self) -> str:
        """返回输出格式要求"""
        return self._output_format

    @property
    def interface(self) -> str:
        """返回代码接口要求"""
        return self._interface

    @property
    def simulator(self) -> str:
        """返回模拟器（回测环境）信息"""
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        """返回富文本样式描述"""
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        """返回实验设置信息"""
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        """
        获取完整的场景描述，用于生成提示。
        """
        return f"""场景背景:
{self.background}
编写可运行代码时应遵循的接口:
{self.interface}
您的代码输出应遵循以下格式:
{self.output_format}
用户可用于测试您的模型的模拟器:
{self.simulator}
"""

    def get_runtime_environment(self):
        """
        获取并返回模型开发环境的运行时信息。
        """
        model_env = get_model_env()
        stdout = get_runtime_environment_by_env(env=model_env)
        return stdout
