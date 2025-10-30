"""

循环不应有大的变化，排除
- 行动选择[当前数据加载器和规范]
- 其他应共享
    - 提议[选择] => 任务[选择] => CoSTEER =>
        -

额外特性:
- 缓存


文件结构
- ___init__.py: coder的入口/代理
- evaluator.py
- conf.py
- exp.py: 实验下的所有内容，例如
    - 任务
    - 实验
    - 工作空间
- test.py
    - 每个coder都可以被测试。
"""

import re
from pathlib import Path

# 导入 rdagent 内部模块
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
from rdagent.components.coder.data_science.conf import (
    DSCoderCoSTEERSettings,
    get_ds_env,
)
from rdagent.components.coder.data_science.raw_data_loader.eval import (
    DataLoaderCoSTEEREvaluator,
)
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent


class DataLoaderMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """
    数据加载器多进程演进策略。
    这个类实现了使用多进程方式来演进和实现数据加载器代码的逻辑。
    """
    def implement_one_task(
        self,
        target_task: DataLoaderTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """
        实现单个数据加载器任务。
        该方法的核心是生成代码规范、查询知识库、调用大语言模型生成代码，最终返回包含代码和规范文件的工作空间。
        """
        # 返回一个包含 "load_data.py", "spec/load_data.md" 的工作空间
        # 将实现的代码分配给新的工作空间。
        competition_info = self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None))
        data_folder_info = self.scen.processed_data_folder_description
        data_loader_task_info = target_task.get_task_information()

        # 查询相似的成功案例知识
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[data_loader_task_info]
            if queried_knowledge is not None
            else []
        )
        # 查询之前的失败尝试
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[data_loader_task_info]
            if queried_knowledge is not None
            else []
        )
        # 过滤掉与当前工作空间中代码相同的失败尝试
        queried_former_failed_knowledge = (
            [
                knowledge
                for knowledge in queried_former_failed_knowledge[0]
                if knowledge.implementation.file_dict.get("load_data.py") != workspace.file_dict.get("load_data.py")
            ],
            queried_former_failed_knowledge[1],
        )

        # 1. 生成规范
        # TODO: 我们可以将规范生成移至一个单独的 COSTEER 任务中
        if DS_RD_SETTING.spec_enabled:
            if "spec/data_loader.md" not in workspace.file_dict:  # 只生成一次规范
                system_prompt = T(".prompts:spec.system").r(
                    runtime_environment=self.scen.get_runtime_environment(),
                    task_desc=data_loader_task_info,
                    competition_info=competition_info,
                    folder_spec=data_folder_info,
                )
                data_loader_prompt = T(".prompts:spec.user.data_loader").r(
                    latest_spec=workspace.file_dict.get("spec/data_loader.md")
                )
                feature_prompt = T(".prompts:spec.user.feature").r(
                    latest_spec=workspace.file_dict.get("spec/feature.md")
                )
                model_prompt = T(".prompts:spec.user.model").r(latest_spec=workspace.file_dict.get("spec/model.md"))
                ensemble_prompt = T(".prompts:spec.user.ensemble").r(
                    latest_spec=workspace.file_dict.get("spec/ensemble.md")
                )
                workflow_prompt = T(".prompts:spec.user.workflow").r(
                    latest_spec=workspace.file_dict.get("spec/workflow.md")
                )

                spec_session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

                data_loader_spec = spec_session.build_chat_completion(user_prompt=data_loader_prompt)
                feature_spec = spec_session.build_chat_completion(user_prompt=feature_prompt)
                model_spec = spec_session.build_chat_completion(user_prompt=model_prompt)
                ensemble_spec = spec_session.build_chat_completion(user_prompt=ensemble_prompt)
                workflow_spec = spec_session.build_chat_completion(user_prompt=workflow_prompt)
            else:
                data_loader_spec = workspace.file_dict["spec/data_loader.md"]
                feature_spec = workspace.file_dict["spec/feature.md"]
                model_spec = workspace.file_dict["spec/model.md"]
                ensemble_spec = workspace.file_dict["spec/ensemble.md"]
                workflow_spec = workspace.file_dict["spec/workflow.md"]

        # 2. 生成代码
        system_prompt = T(".prompts:data_loader_coder.system").r(
            task_desc=data_loader_task_info,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonAgentOut.get_spec(),
        )
        code_spec = (
            data_loader_spec
            if DS_RD_SETTING.spec_enabled
            else T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.DataLoadSpec").r(),
                test_code=(DIRNAME / "eval_tests" / "data_loader_test.txt").read_text(),
            )
        )
        user_prompt = T(".prompts:data_loader_coder.user").r(
            competition_info=competition_info,
            code_spec=code_spec,
            folder_spec=data_folder_info,
            latest_code=workspace.file_dict.get("load_data.py"),
            latest_code_feedback=prev_task_feedback,
        )

        # 尝试最多5次来生成有效的代码
        for _ in range(5):
            data_loader_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if data_loader_code != workspace.file_dict.get("load_data.py"):
                break
            else:
                user_prompt = user_prompt + "\n请避免生成与之前代码相同的代码！"
        else:
            raise CoderError("生成新的数据加载器代码失败。")

        return (
            {
                "spec/data_loader.md": data_loader_spec,
                "spec/feature.md": feature_spec,
                "spec/model.md": model_spec,
                "spec/ensemble.md": ensemble_spec,
                "spec/workflow.md": workflow_spec,
                "load_data.py": data_loader_code,
            }
            if DS_RD_SETTING.spec_enabled
            else {
                "load_data.py": data_loader_code,
            }
        )

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """
        将代码列表分配给演进项。

        代码列表与演进项的子任务对齐。
        如果某个任务未实现，则在列表中放置一个 None。
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo


class DataLoaderCoSTEER(DSCoSTEER):
    """
    数据加载器 CoSTEER 类。
    这个类使用 CoSTEER 框架来协调数据加载器的代码生成过程。
    它负责初始化评估器（Evaluator）和演进策略（Evolving Strategy）。
    """
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(
            DataLoaderCoSTEEREvaluator(scen=scen), scen=scen
        )  # 请指定您是否同意并行运行评估器
        es = DataLoaderMultiProcessEvolvingStrategy(scen=scen, settings=settings)

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

    def develop(self, exp):
        """
        开发过程。
        在父类开发过程的基础上，增加了执行数据加载器测试和提取EDA（探索性数据分析）输出的步骤。
        """
        new_exp = super().develop(exp)

        env = get_ds_env(
            extra_volumes={
                f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": T(
                    "scenarios.data_science.share:scen.input_path"
                ).r()
            },
            running_timeout_period=self.scen.real_full_timeout(),
        )

        stdout = new_exp.experiment_workspace.execute(env=env, entry=f"python test/data_loader_test.py")
        match = re.search(r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===", stdout, re.DOTALL)
        eda_output = match.groups()[1] if match else None
        if eda_output is not None:
            new_exp.experiment_workspace.inject_files(**{"EDA.md": eda_output})
        else:
            eda_output = "无 EDA 输出。"
            new_exp.experiment_workspace.inject_files(**{"EDA.md": eda_output})
        return new_exp
