# 从 copy 模块导入 deepcopy 函数，用于创建对象的深拷贝
from copy import deepcopy
# 从 pathlib 模块导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 因子相关导入
from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)

# 模型相关导入
from rdagent.components.coder.model_coder.conf import get_model_env
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    """
    Qlib 因子实验类 (在此文件中重新定义以便于 QlibQuantScenario 使用)。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")


class QlibModelExperiment(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    """
    Qlib 模型实验类 (在此文件中重新定义以便于 QlibQuantScenario 使用)。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class QlibQuantScenario(Scenario):
    """
    Qlib 量化研究场景类，它整合了因子和模型开发的场景描述。
    可以根据指定的 `tag` (如 "factor" 或 "model") 提供不同的上下文信息。
    """
    def __init__(self) -> None:
        super().__init__()
        self._source_data = deepcopy(get_data_folder_intro())
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:qlib_factor_experiment_setting").r())

    def background(self, tag=None) -> str:
        """
        根据 tag 返回不同的背景信息。
        :param tag: "factor", "model", 或 None (返回全部)。
        """
        assert tag in [None, "factor", "model"]
        quant_background = "场景背景如下:\n" + T(".prompts:qlib_quant_background").r(
            runtime_environment=self.get_runtime_environment(),
        )
        factor_background = "这次，我需要你帮助研究和开发因子。因子场景的背景如下:\n" + T(
            ".prompts:qlib_factor_background"
        ).r(
            runtime_environment=self.get_runtime_environment(tag="factor"),
        )
        model_background = "这次，我需要你帮助研究和开发模型。模型场景的背景如下:\n" + T(
            ".prompts:qlib_model_background"
        ).r(
            runtime_environment=self.get_runtime_environment(tag="model"),
        )

        # TODO: 这里存在一些问题
        if tag is None:
            return quant_background + "\n" + factor_background + "\n" + model_background
        elif tag == "factor":
            return factor_background
        else:
            return model_background

    def get_source_data_desc(self) -> str:
        """返回源数据描述"""
        return self._source_data

    def output_format(self, tag=None) -> str:
        """根据 tag 返回不同的输出格式要求"""
        assert tag in [None, "factor", "model"]
        factor_output_format = (
            "因子代码应输出以下格式:\n" + T(".prompts:qlib_factor_output_format").r()
        )
        model_output_format = (
            "模型代码应输出以下格式:\n" + T(".prompts:qlib_model_output_format").r()
        )

        if tag is None:
            return factor_output_format + "\n" + model_output_format
        elif tag == "factor":
            return factor_output_format
        else:
            return model_output_format

    def interface(self, tag=None) -> str:
        """根据 tag 返回不同的代码接口要求"""
        assert tag in [None, "factor", "model"]
        factor_interface = (
            "因子代码应遵循以下接口:\n" + T(".prompts:qlib_factor_interface").r()
        )
        model_interface = (
            "模型代码应遵循以下接口:\n" + T(".prompts:qlib_model_interface").r()
        )

        if tag is None:
            return factor_interface + "\n" + model_interface
        elif tag == "factor":
            return factor_interface
        else:
            return model_interface

    def simulator(self, tag=None) -> str:
        """根据 tag 返回不同的模拟器信息"""
        assert tag in [None, "factor", "model"]
        factor_simulator = "因子代码将被发送到模拟器:\n" + T(".prompts:qlib_factor_simulator").r()
        model_simulator = "模型代码将被发送到模拟器:\n" + T(".prompts:qlib_model_simulator").r()

        if tag is None:
            return factor_simulator + "\n" + model_simulator
        elif tag == "factor":
            return factor_simulator
        else:
            return model_simulator

    @property
    def rich_style_description(self) -> str:
        """返回富文本样式描述"""
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        """返回实验设置信息"""
        return self._experiment_setting

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
        action: str | None = None,
    ) -> str:
        """
        获取完整的、经过筛选和格式化的场景描述。
        """
        def common_description(action: str | None = None) -> str:
            return f"""\n------场景背景------
{self.background(action)}
------您可以使用的源数据集------
{self.get_source_data_desc()}
"""

        # TODO: 这里处理 source_data 仍然存在一些问题
        def source_data() -> str:
            return f"""
------您可以使用的源数据------
{self.get_source_data_desc()}
"""

        def interface(tag: str | None) -> str:
            return f"""
------编写可运行代码时应遵循的接口------
{self.interface(tag)}
"""

        def output(tag: str | None) -> str:
            return f"""
------您的代码输出应遵循的格式------
{self.output_format(tag)}
"""

        def simulator(tag: str | None) -> str:
            return f"""
------用户可用于测试您的解决方案的模拟器------
{self.simulator(tag)}
"""

        if simple_background:
            return common_description()
        elif filtered_tag == "hypothesis_and_experiment" or filtered_tag == "feedback":
            return common_description() + simulator(None)
        elif filtered_tag == "factor" or filtered_tag == "feature" or filtered_tag == "factors":
            return common_description("factor") + interface("factor") + output("factor") + simulator("factor")
        elif filtered_tag == "model" or filtered_tag == "model tuning":
            return common_description("model") + interface("model") + output("model") + simulator("model")
        elif action == "factor" or action == "model":
            return common_description(action) + interface(action) + output(action) + simulator(action)

    def get_runtime_environment(self, tag: str = None) -> str:
        """
        根据 tag 获取并返回因子或模型开发环境的运行时信息。
        """
        assert tag in [None, "factor", "model"]

        if tag is None or tag == "factor":
            # 使用因子环境获取运行时信息
            factor_env = get_factor_env()
            factor_stdout = get_runtime_environment_by_env(env=factor_env)
            if tag == "factor":
                stdout = factor_stdout

        if tag is None or tag == "model":
            # 使用模型环境获取运行时信息
            model_env = get_model_env()
            model_stdout = get_runtime_environment_by_env(env=model_env)
            if tag == "model":
                stdout = model_stdout

        if tag is None:
            # 合并两个环境的输出
            stdout = (
                "=== [生成因子的环境] ===\n"
                + factor_stdout.strip()
                + "\n\n=== [训练模型的环境] ===\n"
                + model_stdout.strip()
            )

        return stdout
