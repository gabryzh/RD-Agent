# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
# 导入必要的库和模块
import json
import re
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent

# 定义数据加载器评估反馈的别名
DataLoaderEvalFeedback = CoSTEERSingleFeedback


class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):
    """
    数据加载器 CoSTEER 评估器。
    这个类负责评估数据加载器代码的质量。
    """
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledgeV2 = None,
        **kwargs,
    ) -> DataLoaderEvalFeedback:
        """
        评估数据加载器实现。
        此方法通过运行测试代码、检查工作流执行情况，并利用大语言模型生成反馈来评估数据加载器代码的质量。
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
            return DataLoaderEvalFeedback(
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

        # TODO: 我们需要清理生成的临时内容吗？
        fname = "test/data_loader_test.py"
        test_code = (DIRNAME / "eval_tests" / "data_loader_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})
        # 运行测试
        result = implementation.run(env=env, entry=f"python {fname}")
        stdout = result.get_truncated_stdout()
        ret_code = result.exit_code
        # 从输出中分离EDA（探索性数据分析）部分
        match = re.search(r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===(.*)", stdout, re.DOTALL)
        stdout_part_1, eda_output, stdout_part_2 = match.groups() if match else (stdout, None, "")
        stdout = stdout_part_1 + stdout_part_2
        # 如果EDA输出过长，则进行截断并提示
        if eda_output is not None and len(eda_output.split(" ")) > 10000:
            eda_output += "EDA输出长度过长，已截断。请拒绝此实现并激励其减少EDA输出的长度。"

        # 如果主工作流文件存在且之前的测试通过，则执行主工作流
        if "main.py" in implementation.file_dict and ret_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            workflow_stdout = remove_eda_part(workflow_stdout)
        else:
            workflow_stdout = None

        # 构建发送给大语言模型的提示
        system_prompt = T(".prompts:data_loader_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            code=implementation.file_dict["load_data.py"],
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:data_loader_eval.user").r(
            stdout=stdout,
            eda_output=eda_output,
            workflow_stdout=workflow_stdout,
        )

        # 使用大语言模型从JSON构建反馈类，并带重试机制
        fb = build_cls_from_json_with_retry(
            DataLoaderEvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=DataLoaderEvalFeedback.val_and_update_init_dict,
        )
        # 最终决定取决于大语言模型的判断和测试代码的执行结果
        fb.final_decision = fb.final_decision and ret_code == 0

        return fb
