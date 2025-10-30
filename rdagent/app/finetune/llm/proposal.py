from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen
from rdagent.utils.agent.tpl import T


class FinetuneExpGen(DSProposalV2ExpGen):
    """微调实验生成器"""
    def gen(
        self,
        trace: DSTrace,
    ) -> DSExperiment:
        """
        根据轨迹生成一个微调实验。
        """
        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()

        # 获取当前最优的实验和反馈
        if (sota_exp_fb := trace.sota_experiment_fb()) is None:
            sota_exp, fb_to_sota_exp = None, None
        else:
            sota_exp, fb_to_sota_exp = sota_exp_fb

        # 获取EDA输出
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = self.scen.get_scenario_all_desc(eda_output=eda_output)

        # TODO: 这是一个过于简化的版本，将在更多调研后添加更多功能
        sota_exp_desc = "没有可用的先前SOTA实验。"
        failed_exp_feedback_list_desc = "没有可用的先前实验。"

        # 生成任务
        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypotheses=[
                DSHypothesis(
                    component="Model",
                )
            ],
            pipeline=True,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
            fb_to_sota_exp=fb_to_sota_exp,
        )
