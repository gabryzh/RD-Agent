# 从 copy 模块导入 deepcopy 函数，用于创建对象的深拷贝
from copy import deepcopy
# 从 pathlib 模块导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 从 rdagent 的因子编码器配置中导入 get_factor_env 函数
from rdagent.components.coder.factor_coder.config import get_factor_env
# 从 rdagent 的因子编码器因子模块导入相关类
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
# 从 rdagent 核心实验模块导入 Task 类
from rdagent.core.experiment import Task
# 从 rdagent 核心场景模块导入 Scenario 类
from rdagent.core.scenario import Scenario
# 从 qlib 实验工具模块导入 get_data_folder_intro 函数
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
# 从 qlib 实验工作区模块导入 QlibFBWorkspace 类
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
# 从共享模块导入 get_runtime_environment_by_env 函数
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
# 从 rdagent 的模板工具模块导入 T
from rdagent.utils.agent.tpl import T


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    """
    Qlib 因子实验类，继承自通用的因子实验类 FactorExperiment。
    它专门为 Qlib 场景定制，使用 QlibFBWorkspace 作为其工作区。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始化实验工作区，使用 "factor_template" 目录下的模板
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")
        # 初始化标准输出为空字符串
        self.stdout = ""


class QlibFactorScenario(Scenario):
    """
    Qlib 因子场景类，定义了因子开发任务的背景、数据、接口、格式等信息。
    这些信息会用于生成提示（prompt），引导大语言模型完成任务。
    """
    def __init__(self) -> None:
        super().__init__()
        # 使用模板和运行时环境信息，初始化场景的各个组成部分
        self._background = deepcopy(
            T(".prompts:qlib_factor_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        self._source_data = deepcopy(get_data_folder_intro())
        self._output_format = deepcopy(T(".prompts:qlib_factor_output_format").r())
        self._interface = deepcopy(T(".prompts:qlib_factor_interface").r())
        self._strategy = deepcopy(T(".prompts:qlib_factor_strategy").r())
        self._simulator = deepcopy(T(".prompts:qlib_factor_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:qlib_factor_experiment_setting").r())

    @property
    def background(self) -> str:
        """返回场景背景信息"""
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        """返回源数据描述"""
        return self._source_data

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
        :param simple_background: 如果为 True，只返回背景信息。
        :return: 完整的场景描述字符串。
        """
        if simple_background:
            return f"""场景背景:
{self.background}"""
        return f"""场景背景:
{self.background}
您可以使用的源数据:
{self.get_source_data_desc(task)}
编写可运行代码时应遵循的接口:
{self.interface}
您的代码输出应遵循以下格式:
{self.output_format}
用户可用于测试您的因子的模拟器:
{self.simulator}
"""

    def get_runtime_environment(self):
        """
        获取并返回因子开发环境的运行时信息。
        """
        factor_env = get_factor_env()
        stdout = get_runtime_environment_by_env(env=factor_env)
        return stdout
