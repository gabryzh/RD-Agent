from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class ModelBasePropSetting(BasePropSetting):
    """模型基础属性设置"""
    model_config = SettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())

    # 1) 重写基础设置
    scen: str = "rdagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    """用于Qlib模型的场景类"""

    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    """假设生成类"""

    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    """假设到实验的转换类"""

    coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """编码器类"""

    runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """运行器类"""

    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    """摘要器类"""

    evolving_n: int = 10
    """演进次数"""


class FactorBasePropSetting(BasePropSetting):
    """因子基础属性设置"""
    model_config = SettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) 重写基础设置
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    """用于Qlib因子的场景类"""

    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    """假设生成类"""

    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    """假设到实验的转换类"""

    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """编码器类"""

    runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """运行器类"""

    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    """摘要器类"""

    evolving_n: int = 10
    """演进次数"""


class FactorFromReportPropSetting(FactorBasePropSetting):
    """从报告中提取因子的属性设置"""
    # 1) 重写scen属性
    scen: str = "rdagent.scenarios.qlib.experiment.factor_from_report_experiment.QlibFactorFromReportScenario"
    """用于从报告中提取Qlib因子的场景类"""

    # 2) 子任务特定设置:
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    """列出用于因子提取的研究报告的JSON文件路径"""

    max_factors_per_exp: int = 10000
    """每个实验实现的最大因子数"""

    report_limit: int = 10000
    """要处理的最大报告数"""


class QuantBasePropSetting(BasePropSetting):
    """量化基础属性设置"""
    model_config = SettingsConfigDict(env_prefix="QLIB_QUANT_", protected_namespaces=())

    # 1) 重写基础设置
    scen: str = "rdagent.scenarios.qlib.experiment.quant_experiment.QlibQuantScenario"
    """用于Qlib模型的场景类"""

    quant_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen"
    """假设生成类"""

    model_hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    """假设到实验的转换类"""

    model_coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """编码器类"""

    model_runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """运行器类"""

    model_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    """摘要器类"""

    factor_hypothesis2experiment: str = (
        "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    )
    """假设到实验的转换类"""

    factor_coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """编码器类"""

    factor_runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """运行器类"""

    factor_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    """摘要器类"""

    evolving_n: int = 10
    """演进次数"""

    action_selection: str = "bandit"
    """动作选择策略: 'bandit'表示基于bandit的选择, 'llm'表示基于LLM的选择, 'random'表示随机选择"""


FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
QUANT_PROP_SETTING = QuantBasePropSetting()
