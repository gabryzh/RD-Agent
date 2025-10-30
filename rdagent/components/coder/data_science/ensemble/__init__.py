"""
文件结构说明:
- ___init__.py: coder的入口/代理
- evaluator.py: 评估器
- conf.py: 配置
- exp.py: 与实验相关的所有内容，例如：
    - Task (任务)
    - Experiment (实验)
    - Workspace (工作空间)
- test.py: 用于测试coder的脚本
"""

from pathlib import Path

from jinja2 import Environment, StrictUndefined

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
from rdagent.components.coder.data_science.ensemble.eval import EnsembleCoSTEEREvaluator
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.share.ds_costeer import DSCoSTEER
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class EnsembleMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """模型集成的特定多进程演进策略。"""
    def implement_one_task(
        self,
        target_task: EnsembleTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:
        """实现单个集成任务，生成 `ensemble.py` 的代码。"""
        ensemble_information_str = target_task.get_task_information()

        # 1. 查询知识
        # ... (省略了与工作流相似的知识查询逻辑)

        # 2. 生成代码
        system_prompt = T(".prompts:ensemble_coder.system").r(...)

        # 根据配置决定代码规范的来源
        if DS_RD_SETTING.spec_enabled:
            code_spec = workspace.file_dict["spec/ensemble.md"]
        else:
            # 动态生成测试代码和规范
            test_code = Environment(undefined=StrictUndefined).from_string(
                (Path(__file__).parent / "eval_tests" / "ensemble_test.txt").read_text()
            ).render(...)
            code_spec = T("scenarios.data_science.share:component_spec.general").r(
                spec=T("scenarios.data_science.share:component_spec.Ensemble").r(), test_code=test_code
            )

        user_prompt = T(".prompts:ensemble_coder.user").r(
            code_spec=code_spec,
            latest_code=workspace.file_dict.get("ensemble.py"),
            latest_code_feedback=prev_task_feedback,
        )

        # 尝试生成与之前不同的代码
        for _ in range(5):
            ensemble_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            if ensemble_code != workspace.file_dict.get("ensemble.py"):
                break
            user_prompt += "\n请避免生成与之前相同的代码！"
        else:
            raise CoderError("无法生成新的集成代码。")

        return {"ensemble.py": ensemble_code}

    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """将生成的代码列表分配给演进项。"""
        for index, code_dict in enumerate(code_list):
            if code_dict is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code_dict)
        return evo


class EnsembleCoSTEER(DSCoSTEER):
    """专门用于模型集成的 CoSTEER 实现。"""
    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = DSCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator(EnsembleCoSTEEREvaluator(scen=scen), scen=scen)
        es = EnsembleMultiProcessEvolvingStrategy(scen=scen, settings=settings)
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
