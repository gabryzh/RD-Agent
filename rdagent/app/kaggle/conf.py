from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class KaggleBasePropSetting(ExtendedBaseSettings):
    """Kaggle竞赛场景的基础配置类"""
    model_config = SettingsConfigDict(env_prefix="KG_", protected_namespaces=())

    # 1) 覆盖默认设置
    scen: str = "rdagent.scenarios.kaggle.experiment.scenario.KGScenario"
    """数据挖掘模型的场景类"""

    hypothesis_gen: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesisGen"
    """假设生成类"""

    hypothesis2experiment: str = "rdagent.scenarios.kaggle.proposal.proposal.KGHypothesis2Experiment"
    """假设到实验的转换类"""

    feature_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGFactorCoSTEER"
    """特征编码器类"""

    model_feature_selection_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelFeatureSelectionCoder"
    """模型特征选择编码器类"""

    model_coder: str = "rdagent.scenarios.kaggle.developer.coder.KGModelCoSTEER"
    """模型编码器类"""

    feature_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGFactorRunner"
    """特征运行器类"""

    model_runner: str = "rdagent.scenarios.kaggle.developer.runner.KGModelRunner"
    """模型运行器类"""

    summarizer: str = "rdagent.scenarios.kaggle.developer.feedback.KGExperiment2Feedback"
    """摘要器类"""

    evolving_n: int = 10
    """演进次数"""

    competition: str = ""
    """Kaggle竞赛名称，例如 'sf-crime'"""

    template_path: str = "rdagent/scenarios/kaggle/experiment/templates"
    """Kaggle竞赛基础模板路径"""

    local_data_path: str = ""
    """存储Kaggle竞赛数据的文件夹"""

    # 测试评估相关
    if_using_mle_data: bool = False
    auto_submit: bool = False
    """自动将每个实验结果上传并提交到Kaggle平台"""

    knowledge_base: str = ""
    """知识库类，启用高级图RAG时使用'KGKnowledgeGraph'，否则为空。"""
    if_action_choosing_based_on_UCB: bool = False
    """启用基于UCB算法的决策机制"""

    domain_knowledge_path: str = "/data/userdata/share/kaggle/domain_knowledge"
    """存储领域知识文件的文件夹（.case格式）"""

    knowledge_base_path: str = "kg_graph.pkl"
    """高级版图RAG的路径"""

    rag_path: str = "git_ignore_folder/kaggle_vector_base.pkl"
    """基础版向量RAG的路径"""

    if_using_vector_rag: bool = False
    """启用基础版向量RAG"""

    if_using_graph_rag: bool = False
    """启用高级版图RAG"""

    mini_case: bool = False
    """为实验启用迷你案例研究"""

    time_ratio_limit_to_enable_hyperparameter_tuning: float = 1
    """
    启用超参数调整的运行器时间比例限制，如果不更改，则在第一次演进中始终启用超参数调整。
    """

    res_time_ratio_limit_to_enable_hyperparameter_tuning: float = 1
    """
    启用超参数调整的总体剩余时间比例限制，如果不更改，则在第一次演进中始终启用超参数调整。
    `1`表示当剩余时间为100%时启用超参数调整（因此始终启用）。
    """

    only_first_loop_enable_hyperparameter_tuning: bool = True
    """仅在评估的第一轮中启用超参数调整反馈。"""

    only_enable_tuning_in_merge: bool = False
    """仅在合并阶段启用超参数调整"""


KAGGLE_IMPLEMENT_SETTING = KaggleBasePropSetting()
