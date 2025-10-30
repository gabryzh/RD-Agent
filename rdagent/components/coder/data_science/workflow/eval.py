import json
import re
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 类型别名
WorkflowSingleFeedback = CoSTEERSingleFeedback
WorkflowMultiFeedback = CoSTEERMultiFeedback


class WorkflowGeneralCaseSpecEvaluator(CoSTEEREvaluator):
    """
    工作流通用场景评估器。

    动机:
    - 在最简单的情况下，我们已将数据拆分为训练、验证和测试集。我们要求模型学习（可选地在验证数据上验证），并在测试数据上进行推断。

    测试工作流:
    - 构建训练、验证和测试数据来运行它，并测试输出（例如，形状等）。
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        # 检查知识库，看是否可以跳过评估
        target_task_information = target_task.get_task_information()
        if queried_knowledge and target_task_information in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge and target_task_information in queried_knowledge.failed_task_info_set:
            return WorkflowSingleFeedback(
                execution="此任务已失败多次，跳过实现。",
                return_checking="此任务已失败多次，跳过实现。",
                code="此任务已失败多次，跳过实现。",
                final_decision=False,
            )

        # 获取数据科学执行环境
        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        # 1. 执行前清理工作空间
        implementation.execute(env=env, entry=get_clear_ws_cmd())

        # 2. 运行主脚本并捕获输出
        stdout = implementation.execute(env=env, entry="python -m coverage run main.py")
        stdout = remove_eda_part(stdout)  # 移除EDA相关的输出

        # 3. 检查分数文件 (scores.csv)
        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        if not score_fp.exists():
            score_check_text = "[错误] 未生成指标文件 (scores.csv)！"
            score_ret_code = 1
            # 检查代码覆盖率，以判断是否有代码被执行
            implementation.execute(env=env, entry="python -m coverage json -o coverage.json")
            coverage_report_path = implementation.workspace_path / "coverage.json"
            if coverage_report_path.exists():
                used_files = set(json.loads(coverage_report_path.read_text())["files"].keys())
                coverage_report_path.unlink()
                if len(used_files) == 1:
                    score_check_text += f"\n[错误] 唯一使用的脚本是 {used_files}。\n请检查您是否在 'main.py' 中实现了入口点。"
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                # 检查索引、列名和值是否正确
                # ...
            except Exception as e:
                score_check_text += f"\n[错误] 检查 scores.csv 文件时出错: {e}"
                score_ret_code = 1

        # 4. 检查提交文件 (submission.csv)
        base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
        implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
        submission_result = implementation.run(env=env, entry="python test/submission_format_test.py")
        stdout += "\n" + submission_result.get_truncated_stdout()

        # 5. 使用LLM进行代码和逻辑评估
        system_prompt = T(".prompts:workflow_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=None),
            task_desc=target_task.get_task_information(),
            spec=(
                implementation.file_dict.get("spec/workflow.md")
                if DS_RD_SETTING.spec_enabled
                else T("scenarios.data_science.share:component_spec.Workflow").r()
            ),
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.file_dict.get("main.py"),
        )
        wfb = build_cls_from_json_with_retry(
            WorkflowSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=WorkflowSingleFeedback.val_and_update_init_dict,
        )

        # 6. 综合文件检查结果和LLM评估结果，得出最终决策
        if score_ret_code != 0:
            wfb.final_decision = False
            wfb.return_checking = (wfb.return_checking or "") + "\n" + score_check_text
        if submission_result.exit_code != 0:
            wfb.final_decision = False
            wfb.return_checking = (wfb.return_checking or "") + "\n提交文件检查失败。"

        return wfb
