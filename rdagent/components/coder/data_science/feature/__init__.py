from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.conf import DSCoderCoSTEERSettings
from rdagent.components.coder.data_science.feature.eval import FeatureCoSTEEREvaluator
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class FeatureMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """特征工程的特定多进程演进策略。"""
    def implement_one_task(
        self,
        target_task: FeatureTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """实现单个特征工程任务，生成 `feature.py` 的代码。"""
        feature_information_str = target_task.get_task_information()

        # 1. 查询知识
        # ... (省略了与工作流相似的知识查询逻辑)

        # 2. 生成代码
        system_prompt = T(".prompts:feature_coder.system").r(
            competition_info=self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md")),
            task_desc=feature_information_str,
            data_loader_code=workspace.file_dict.get("load_data.py"),
            # ... (省略知识渲染)
            out_spec=PythonAgentOut.get_spec(),
        )

        # 根据配置决定代码规范的来源
        code_spec = (
            workspace.file_dict.get("spec/feature.md")
            if DS_RD_SETTING.spec_enabled
            else T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.FeatureEng").r(),
                test_code=(Path(__file__).parent / "eval_tests" / "feature_test.txt").read_text(),
            )
        )
        user_prompt = T(".prompts:feature_coder.user").r(
            code_spec=code_spec,
            latest_code=workspace.file_dict.get("feature.py"),
            latest_code_feedback=prev_task_feedback,
        )

        # 尝试生成与之前不同的代码
        for _ in range(5):
            feature_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if feature_code != workspace.file_dict.get("feature.py"):
                break
            user_prompt += "\n请避免生成与之前相同的代码！"
        else:
            raise CoderError("无法生成新的特征工程代码。")

        return {"feature.py": feature_code}

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """将生成的代码列表分配给演进项。"""
        for index, code_dict in enumerate(code_list):
            if code_dict is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_dict)
        return evo


class FeatureCoSTEER(DSCoSTEER):
    """专门用于特征工程的 CoSTEER 实现。"""
    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(FeatureCoSTEEREvaluator(scen=scen), scen=scen)
        es = FeatureMultiProcessEvolvingStrategy(scen=scen, settings=settings)
        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=DS_RD_SETTING.coder_max_loop,
            **kwargs,
        )
