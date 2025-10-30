"""
文件结构说明... (与 ensemble/__init__.py 相同)
"""

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
from rdagent.components.coder.data_science.pipeline.eval import PipelineCoSTEEREvaluator
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class PipelineMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """数据科学管道的特定多进程演进策略。"""
    def implement_one_task(
        self,
        target_task: PipelineTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """实现单个管道任务，生成 `main.py` 的代码。"""
        # ... (省略了与工作流和集成相似的知识查询和prompt构建逻辑)

        # 尝试生成与之前不同的代码
        for _ in range(5):
            pipeline_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if pipeline_code != workspace.file_dict.get("main.py"):
                break
            user_prompt += "\n请避免生成与之前相同的代码！"
        else:
            raise CoderError("无法生成新的管道代码。")

        return {"main.py": pipeline_code}

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """将生成的代码列表分配给演进项。"""
        # ... (与工作流和集成的实现相同)
        return evo


class PipelineCoSTEER(DSCoSTEER):
    """专门用于数据科学管道的 CoSTEER 实现。"""
    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = DSCoderCoSTEERSettings()

        # 动态构建评估器列表
        eval_l = [PipelineCoSTEEREvaluator(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="sample"))
        # 添加配置中指定的额外评估器
        for evaluator_path in settings.extra_evaluator:
            eval_l.append(import_class(evaluator_path)(scen=scen))
        for extra_eval_path in settings.extra_eval:
            kls = import_class(extra_eval_path)
            eval_l.append(kls(scen=scen))

        eva = CoSTEERMultiEvaluator(single_evaluator=eval_l, scen=scen)
        es = PipelineMultiProcessEvolvingStrategy(scen=scen, settings=settings)

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
