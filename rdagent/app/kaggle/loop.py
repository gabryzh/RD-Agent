import subprocess
from typing import Any

import fire

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.kaggle.experiment.scenario import (
    KG_ACTION_FEATURE_ENGINEERING,
    KG_ACTION_FEATURE_PROCESSING,
    KG_ACTION_MODEL_FEATURE_SELECTION,
)
from rdagent.scenarios.kaggle.experiment.utils import python_files_to_notebook
from rdagent.scenarios.kaggle.kaggle_crawler import download_data
from rdagent.scenarios.kaggle.proposal.proposal import KGTrace


class KaggleRDLoop(RDLoop):
    """Kaggle竞赛的自动研发循环"""
    def __init__(self, PROP_SETTING: BasePropSetting):
        """初始化Kaggle研发循环"""
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
        logger.log_object(scen, tag="场景")
        knowledge_base = (
            import_class(PROP_SETTING.knowledge_base)(PROP_SETTING.knowledge_base_path, scen)
            if PROP_SETTING.knowledge_base != ""
            else None
        )
        logger.log_object(knowledge_base, tag="知识库")
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="假设生成器")
        self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
        logger.log_object(self.hypothesis2experiment, tag="假设到实验")
        self.feature_coder: Developer = import_class(PROP_SETTING.feature_coder)(scen)
        logger.log_object(self.feature_coder, tag="特征编码器")
        self.model_feature_selection_coder: Developer = import_class(PROP_SETTING.model_feature_selection_coder)(scen)
        logger.log_object(self.model_feature_selection_coder, tag="模型特征选择编码器")
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="模型编码器")
        self.feature_runner: Developer = import_class(PROP_SETTING.feature_runner)(scen)
        logger.log_object(self.feature_runner, tag="特征运行器")
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="模型运行器")
        self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        logger.log_object(self.summarizer, tag="摘要器")
        self.trace = KGTrace(scen=scen, knowledge_base=knowledge_base)
        super(RDLoop, self).__init__()

    def coding(self, prev_out: dict[str, Any]):
        """编码阶段"""
        if prev_out["direct_exp_gen"]["propose"].action in [
            KG_ACTION_FEATURE_ENGINEERING,
            KG_ACTION_FEATURE_PROCESSING,
        ]:
            exp = self.feature_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        elif prev_out["direct_exp_gen"]["propose"].action == KG_ACTION_MODEL_FEATURE_SELECTION:
            exp = self.model_feature_selection_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        else:
            exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="编码器结果")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """运行阶段"""
        if prev_out["direct_exp_gen"]["propose"].action in [
            KG_ACTION_FEATURE_ENGINEERING,
            KG_ACTION_FEATURE_PROCESSING,
        ]:
            exp = self.feature_runner.develop(prev_out["coding"])
        else:
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="运行器结果")
        if KAGGLE_IMPLEMENT_SETTING.competition in [
            "optiver-realized-volatility-prediction",
            "covid19-global-forecasting-week-1",
        ]:
            try:
                python_files_to_notebook(KAGGLE_IMPLEMENT_SETTING.competition, exp.experiment_workspace.workspace_path)
            except Exception as e:
                logger.error(f"合并python文件到一个文件失败: {e}")
        if KAGGLE_IMPLEMENT_SETTING.auto_submit:
            csv_path = exp.experiment_workspace.workspace_path / "submission.csv"
            try:
                subprocess.run(
                    [
                        "kaggle",
                        "competitions",
                        "submit",
                        "-f",
                        str(csv_path.absolute()),
                        "-m",
                        str(csv_path.parent.absolute()),
                        KAGGLE_IMPLEMENT_SETTING.competition,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"自动提交失败: \n{e}")
            except Exception as e:
                logger.error(f"使用kaggle api时发生其他异常:\n{e}")

        return exp

    skip_loop_error = (ModelEmptyError, FactorEmptyError, CoderError)


def main(path=None, step_n=None, competition=None):
    """
    Kaggle场景中模型的自动研发演进循环。
    您可以通过以下方式继续运行会话：
    .. code-block:: bash
        dotenv run -- python rdagent/app/kaggle/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` 是可选参数
        rdagent kaggle --competition playground-series-s4e8  # 推荐使用此命令
    """
    if competition:
        KAGGLE_IMPLEMENT_SETTING.competition = competition
        download_data(competition=competition, settings=KAGGLE_IMPLEMENT_SETTING)
        if KAGGLE_IMPLEMENT_SETTING.if_using_graph_rag:
            KAGGLE_IMPLEMENT_SETTING.knowledge_base = (
                "rdagent.scenarios.kaggle.knowledge_management.graph.KGKnowledgeGraph"
            )
    else:
        logger.error("请指定竞赛名称。")
    if path is None:
        kaggle_loop = KaggleRDLoop(KAGGLE_IMPLEMENT_SETTING)
    else:
        kaggle_loop = KaggleRDLoop.load(path)
    kaggle_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
