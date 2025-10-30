import json
import re
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 类型别名
EnsembleEvalFeedback = CoSTEERSingleFeedback


class EnsembleCoSTEEREvaluator(CoSTEEREvaluator):
    """模型集成（Ensemble）的CoSTEER评估器。"""

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> EnsembleEvalFeedback:
        """
        评估生成的集成代码。

        流程:
        1. 检查知识库以确定是否可以跳过评估。
        2. 准备执行环境。
        3. 动态生成并注入单元测试代码。
        4. 执行测试并捕获输出和返回码。
        5. （可选）如果存在 main.py，则执行完整工作流。
        6. 使用LLM分析执行结果和代码，生成反馈。
        7. 结合测试返回码和LLM的判断，得出最终决策。
        """
        target_task_information = target_task.get_task_information()
        metric_name = self.scen.metric_name

        # 1. 知识库检查
        if queried_knowledge and target_task_information in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge and target_task_information in queried_knowledge.failed_task_info_set:
            return EnsembleEvalFeedback(
                execution="此任务已失败多次，跳过实现。",
                code="此任务已失败多次，跳过实现。",
                return_checking="此任务已失败多次，跳过实现。",
                final_decision=False,
            )

        # 2. 准备环境
        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        # 3. 动态生成测试代码
        fname = "test/ensemble_test.py"  # 注意：文件名从 .txt 改为 .py
        test_code_template = (Path(__file__).parent / "eval_tests" / "ensemble_test.txt").read_text()
        test_code = Environment(undefined=StrictUndefined).from_string(test_code_template).render(
            model_names=[fn[:-3] for fn in implementation.file_dict if fn.startswith("model_") and "test" not in fn],
            metric_name=metric_name,
        )
        implementation.inject_files(**{fname: test_code})

        # 4. 执行测试
        result = implementation.run(env=env, entry=f"python {fname}")
        stdout = result.get_truncated_stdout()
        ret_code = result.exit_code
        stdout += f"\n注意: 上述脚本运行的返回码为 {ret_code}"

        # 5. （可选）执行工作流
        workflow_stdout = None
        if "main.py" in implementation.file_dict and ret_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            workflow_stdout = remove_eda_part(workflow_stdout)

        # 6. LLM评估
        system_prompt = T(".prompts:ensemble_eval.system").r(
            task_desc=target_task_information,
            test_code=test_code,
            metric_name=metric_name,
            code=implementation.file_dict.get("ensemble.py"),
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:ensemble_eval.user").r(
            stdout=stdout,
            workflow_stdout=workflow_stdout,
        )
        efb = build_cls_from_json_with_retry(
            EnsembleEvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=EnsembleEvalFeedback.val_and_update_init_dict,
        )

        # 7. 最终决策
        efb.final_decision = efb.final_decision and ret_code == 0
        return efb
