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
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class WorkflowMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """
    数据科学工作流的特定多进程演进策略。
    """
    def implement_one_task(
        self,
        target_task: WorkflowTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """
        实现单个工作流任务，即生成 `main.py` 的代码。
        """
        workflow_information_str = target_task.get_task_information()

        # 1. 查询知识库
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge.get(workflow_information_str, [])
            if queried_knowledge
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces.get(workflow_information_str, ([], None))
            if queried_knowledge
            else ([], None)
        )
        # 过滤掉与当前代码相同的历史失败记录
        queried_former_failed_knowledge = (
            [k for k in queried_former_failed_knowledge[0] if k.implementation.file_dict.get("main.py") != workspace.file_dict.get("main.py")],
            queried_former_failed_knowledge[1],
        )

        # 2. 构建 prompt 并生成代码
        system_prompt = T(".prompts:workflow_coder.system").r(
            task_desc=workflow_information_str,
            competition_info=self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md")),
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonAgentOut.get_spec(),
        )
        user_prompt = T(".prompts:workflow_coder.user").r(
            load_data_code=workspace.file_dict.get("load_data.py"),
            feature_code=workspace.file_dict.get("feature.py"),
            model_codes=workspace.get_codes(r"^model_(?!test)\w+\.py$"),
            ensemble_code=workspace.file_dict.get("ensemble.py"),
            latest_code=workspace.file_dict.get("main.py"),
            code_spec=(
                workspace.file_dict.get("spec/workflow.md")
                if DS_RD_SETTING.spec_enabled
                else T("scenarios.data_science.share:component_spec.Workflow").r()
            ),
            latest_code_feedback=prev_task_feedback,
        )

        # 尝试最多5次生成与之前不同的代码
        for _ in range(5):
            workflow_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if workflow_code != workspace.file_dict.get("main.py"):
                break
            user_prompt += "\nPlease avoid generating same code to former code!"
        else:
            raise CoderError("无法生成新的工作流代码。")

        return {"main.py": workflow_code}

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """将生成的代码列表分配给演进项。"""
        for index, code_dict in enumerate(code_list):
            if code_dict is None:
                continue
            if evo.sub_workspace_list[index] is None:
                # 工作流任务共享同一个实验工作空间
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_dict)
        return evo


class WorkflowCoSTEER(DSCoSTEER):
    """
    专门用于数据科学工作流的 CoSTEER 实现。
    """
    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(WorkflowGeneralCaseSpecEvaluator(scen=scen), scen=scen)
        es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=settings)
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
