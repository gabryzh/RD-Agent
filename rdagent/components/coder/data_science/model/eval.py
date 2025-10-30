"""
扩展之前的测试
-
"""

# 导入必要的库和模块
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
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent
# 定义模型单次反馈的别名
ModelSingleFeedback = CoSTEERSingleFeedback


# 以下是用于测试已实现模型规范的单元测试 ------------------
class ModelGeneralCaseSpecEvaluator(CoSTEEREvaluator):
    """
    动机案例：
    - 最简单的情况，我们已经将数据分成了训练数据、验证数据和测试数据。我们要求模型学习（可选地在验证数据上验证），并在测试数据上进行推断。

    测试工作流程：
    - 构建训练、验证和测试数据来运行它，并测试输出（例如，形状等）。
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelSingleFeedback:
        """
        评估模型实现。
        此方法通过运行测试代码、检查工作流执行情况，并利用大语言模型生成反馈来评估模型代码的质量。
        """
        target_task_information = target_task.get_task_information()
        # 检查是否已有成功的知识记录，如果有则直接返回
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        # 检查任务是否失败次数过多，如果是则跳过
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return ModelSingleFeedback(
                execution="此任务失败次数过多，跳过实现。",
                return_checking="此任务失败次数过多，跳过实现。",
                code="此任务失败次数过多，跳过实现。",
                final_decision=False,
            )

        # 获取数据科学环境配置
        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        if_model_removed = False

        # 如果模型文件存在，则运行单元测试
        if f"{target_task.name}.py" in implementation.file_dict:
            fname = "test/model_test.py"
            # 从文件中读取测试代码模板并替换模型名称
            test_code = (
                (DIRNAME / "eval_tests" / "model_test.txt").read_text().replace("model01", target_task.name)
            )  # 只检查本次更改的模型
            implementation.inject_files(**{fname: test_code})
            # 运行测试
            result = implementation.run(env=env, entry=f"python {fname}")
            stdout = result.get_truncated_stdout()
            ret_code = result.exit_code

            if stdout is None:
                raise CoderError(
                    "执行输出包含过多的进度条，导致LLM的token大小超出限制。"
                )
        # 如果模型文件不存在，则认为是模型删除操作
        else:
            ret_code = 0
            if_model_removed = True
            stdout = f"模型 {target_task.name} 删除成功。"

        # 如果主工作流文件存在且之前的测试通过，则执行主工作流
        if "main.py" in implementation.file_dict and ret_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            # 移除探索性数据分析部分的输出
            workflow_stdout = remove_eda_part(workflow_stdout)
        else:
            workflow_stdout = None

        # 根据模型是否被移除，构建不同的提示信息
        if if_model_removed:
            system_prompt = T(".prompts:model_eval_rm.system").r(
                task_desc=target_task.get_task_information(),
                workflow_stdout=workflow_stdout,
                workflow_code=implementation.all_codes,
            )
            user_prompt = T(".prompts:model_eval_rm.user").r(
                stdout=stdout,
                workflow_stdout=workflow_stdout,
            )
        else:
            system_prompt = T(".prompts:model_eval.system").r(
                task_desc=target_task.get_task_information(),
                test_code=test_code,
                code=implementation.file_dict[f"{target_task.name}.py"],
                workflow_stdout=workflow_stdout,
                workflow_code=implementation.all_codes,
            )
            user_prompt = T(".prompts:model_eval.user").r(
                stdout=stdout,
                workflow_stdout=workflow_stdout,
            )

        # 使用大语言模型从JSON构建反馈类，并带重试机制
        fb = build_cls_from_json_with_retry(
            ModelSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=ModelSingleFeedback.val_and_update_init_dict,
        )
        # 最终决定取决于大语言模型的判断和测试代码的执行结果
        fb.final_decision = fb.final_decision and ret_code == 0

        return fb
