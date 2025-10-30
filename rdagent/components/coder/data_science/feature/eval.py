import json
import re
from pathlib import Path

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
from rdagent.utils.fmt import shrink_text

# 类型别名
FeatureEvalFeedback = CoSTEERSingleFeedback


class FeatureCoSTEEREvaluator(CoSTEEREvaluator):
    """特征工程的CoSTEER评估器。"""

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FeatureEvalFeedback:
        """
        评估生成的特征工程代码。

        流程:
        1. 检查知识库。
        2. 准备执行环境。
        3. 注入并执行单元测试。
        4. （可选）执行完整工作流。
        5. 使用LLM分析结果并生成反馈。
        6. 结合测试结果得出最终决策。
        """
        target_task_information = target_task.get_task_information()

        # 1. 知识库检查
        if queried_knowledge and target_task_information in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge and target_task_information in queried_knowledge.failed_task_info_set:
            return FeatureEvalFeedback(
                execution="此任务已失败多次，跳过实现。",
                return_checking="此任务已失败多次，跳过实现。",
                code="此任务已失败多次，跳过实现。",
                final_decision=False,
            )

        # 2. 准备环境
        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        # 3. 注入并执行测试
        fname = "test/feature_test.py"
        test_code = (Path(__file__).parent / "eval_tests" / "feature_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})
        result = implementation.run(env=env, entry=f"python {fname}")

        # 4. （可选）执行工作流
        workflow_stdout = None
        if "main.py" in implementation.file_dict and result.exit_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            workflow_stdout = remove_eda_part(workflow_stdout)

        # 5. LLM评估
        system_prompt = T(".prompts:feature_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            code=implementation.file_dict.get("feature.py"),
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:feature_eval.user").r(
            stdout=result.get_truncated_stdout(),
            workflow_stdout=workflow_stdout,
        )

        fb = build_cls_from_json_with_retry(
            FeatureEvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=FeatureEvalFeedback.val_and_update_init_dict,
        )

        # 6. 最终决策
        fb.final_decision = fb.final_decision and result.exit_code == 0
        return fb
