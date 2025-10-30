# 导入必要的库和模块
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
from rdagent.components.coder.data_science.model.eval import (
    ModelGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonBatchEditOut
from rdagent.utils.agent.tpl import T

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent


class ModelMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """
    模型多进程演进策略。
    这个类实现了使用多进程方式来演进和实现模型代码的逻辑。
    """

    def implement_one_task(
        self,
        target_task: ModelTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """
        实现单个模型任务。
        该方法的核心是通过查询知识库、调用大语言模型生成代码、并对生成结果进行后处理来完成一个具体的模型编码任务。
        """
        model_information_str = target_task.get_task_information()

        # 1. 查询知识库
        # 查询相似的成功案例知识
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[model_information_str]
            if queried_knowledge is not None
            else []
        )
        # 查询之前的失败尝试
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[model_information_str]
            if queried_knowledge is not None
            else []
        )
        # 过滤掉与当前工作空间中代码相同的失败尝试
        queried_former_failed_knowledge = (
            (
                [
                    knowledge
                    for knowledge in queried_former_failed_knowledge[0]
                    if knowledge.implementation.file_dict.get(f"{target_task.name}.py")
                    != workspace.file_dict.get(f"{target_task.name}.py")
                ],
                queried_former_failed_knowledge[1],
            )
        )

        # 2. 生成代码
        # 构建系统提示
        system_prompt = T(".prompts:model_coder.system").r(
            task_desc=model_information_str,
            competition_info=self.scen.get_scenario_all_desc(eda_output=workspace.file_dict.get("EDA.md", None)),
            data_loader_code=workspace.file_dict.get("load_data.py"),
            feature_code=workspace.file_dict["feature.py"],
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge[0],
            out_spec=PythonBatchEditOut.get_spec(),
        )

        # 根据设置确定代码规范
        code_spec = (
            workspace.file_dict["spec/model.md"]
            if DS_RD_SETTING.spec_enabled
            else T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.Model").r(),
                test_code=(DIRNAME / "eval_tests" / "model_test.txt").read_text().replace("model01", target_task.name),
            )
        )
        # 构建用户提示
        user_prompt = T(".prompts:model_coder.user_general").r(
            code_spec=code_spec,
            latest_model_code=workspace.get_codes(
                r"^model_(?!test)\w+\.py$"
            ),  # TODO: 如果此处失败率高，应清理此步骤，减少信息量。
            latest_code_feedback=prev_task_feedback,
        )

        # 尝试最多5次来生成有效的代码
        for _ in range(5):
            # 调用大语言模型生成代码
            batch_edit = PythonBatchEditOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )

            # 检查生成的文件是否都是模型文件
            if not all(i.startswith("model_") for i in batch_edit.keys()):
                user_prompt += "\n你应该只更新模型代码！"
                continue

            # 3. 后处理，将文件名与任务名对齐
            # 我们假设 batch_edit 只包含一个模型文件的更新。
            batch_edit = {
                (f"{target_task.name}.py" if value != "__DEL__" and key != f"{target_task.name}.py" else key): value
                for key, value in batch_edit.items()
            }

            user_prompt = user_prompt + "\n请避免生成与之前代码相同的代码！"
            # TODO: 除了代码重复问题，还应考虑其他导致重试的问题。
            if f"{target_task.name}.py" not in batch_edit:
                continue

            # 检查文件名长度是否合法
            if batch_edit and max(len(i.encode("utf-8")) for i in batch_edit.keys()) > 255:
                continue

            # 如果生成的代码与旧代码不同，则跳出循环
            if batch_edit[f"{target_task.name}.py"] != "__DEL__" and batch_edit[
                f"{target_task.name}.py"
            ] != workspace.file_dict.get(f"{target_task.name}.py"):
                break

            # 如果任务是删除模型，假设一次只处理一个模型。
            if len(batch_edit) == 1 and batch_edit[f"{target_task.name}.py"] == "__DEL__":
                break
        else:
            # 如果5次尝试都失败，则抛出异常
            raise CoderError("生成新的模型代码失败。")

        return batch_edit

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


class ModelCoSTEER(DSCoSTEER):
    """
    模型 CoSTEER 类。
    这个类使用 CoSTEER 框架来协调模型的代码生成过程。
    它负责初始化评估器（Evaluator）和演进策略（Evolving Strategy）。
    """
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        # 初始化设置
        settings = DSCoderCoSTEERSettings()
        # 初始化评估器，这里使用了多评估器来并行评估
        eva = CoSTEERMultiEvaluator(
            ModelGeneralCaseSpecEvaluator(scen=scen), scen=scen
        )
        # 初始化演进策略
        es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=settings)

        # 调用父类的构造函数
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
