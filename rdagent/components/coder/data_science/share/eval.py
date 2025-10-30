# 导入必要的库和模块
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent

# 定义反馈类的别名
PipelineSingleFeedback = CoSTEERSingleFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback

# 定义未找到文件时的占位符
NO_SUB = "<未找到 submission.csv 文件>"
NO_SCORE = "<未找到 scores.csv 文件>"


class ModelDumpEvaluator(CoSTEEREvaluator):
    """
    模型转储评估器。
    这个评估器假设它在模型训练和预测之后运行，用于验证模型是否被正确保存（转储），
    并且可以在不重新训练的情况下被加载并用于推理。
    """

    def __init__(self, scen: Scenario, data_type: Literal["sample", "full"]):
        """
        初始化评估器。

        参数:
            scen (Scenario): 当前的场景实例。
            data_type (Literal["sample", "full"]): 使用的数据类型，"sample"表示调试用的采样数据，"full"表示完整数据。
        """
        super().__init__(scen)
        self.data_type = data_type

    def evaluate(
        self, target_task: Task, implementation: FBWorkspace, gt_implementation: FBWorkspace, *kargs, **kwargs
    ) -> CoSTEERSingleFeedback:
        """
        执行评估。
        """

        model_folder = implementation.workspace_path / "models"
        # 1) 检查模型文件夹是否不为空
        if not model_folder.exists() or not any(model_folder.iterdir()):
            err_msg = "模型文件夹（`models`子文件夹）为空或不存在。模型未被转储。"
            return CoSTEERSingleFeedback(
                execution=err_msg,
                return_checking=err_msg,
                code=err_msg,
                final_decision=False,
            )

        # 根据数据类型设置数据源路径和环境
        data_source_path = (
            f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}"
            if self.data_type == "full"
            else self.scen.debug_path
        )
        env = get_ds_env(
            extra_volumes={data_source_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=(
                self.scen.real_full_timeout() if self.data_type == "full" else self.scen.real_debug_timeout()
            ),
        )

        # 2) 检查重新运行模型后的结果和标准输出。

        # 执行前读取 submission.csv 和 scores.csv 的内容
        submission_content_before = (
            (implementation.workspace_path / "submission.csv").read_text()
            if (implementation.workspace_path / "submission.csv").exists()
            else NO_SUB
        )
        scores_content_before = (
            (implementation.workspace_path / "scores.csv").read_text()
            if (implementation.workspace_path / "scores.csv").exists()
            else NO_SCORE
        )

        # 删除 submission.csv 和 scores.csv 文件
        implementation.execute(env=env, entry=get_clear_ws_cmd(stage="before_inference"))

        # 执行主脚本的推理模式
        stdout = remove_eda_part(
            implementation.execute(env=env, entry="strace -e trace=file -f -o trace.log python main.py --inference")
        )

        # 遍历模型文件夹并列出文件
        model_folder_files = [
            str(file.relative_to(implementation.workspace_path)) for file in model_folder.iterdir() if file.is_file()
        ]

        # 使用 strace 追踪文件打开操作，检查模型是否从数据源加载数据
        opened_trace_lines = None
        if (implementation.workspace_path / "trace.log").exists():
            input_path = T("scenarios.data_science.share:scen.input_path").r()
            abs_input_path = str(Path(input_path).resolve())
            # 匹配类似 `openat(AT_FDCWD, "/home/user/project/main.py", O_RDONLY) = 5` 的路径字符串
            path_regex = re.compile(r'openat\(.+?,\s*"([^"]+)"')
            log_content = (implementation.workspace_path / "trace.log").read_text()

            opened_files = set()
            for line in log_content.splitlines():
                if "openat" not in line or (abs_input_path not in line and input_path not in line):
                    continue

                match = path_regex.search(line)
                if match:
                    full_path = Path(match.group(1)).resolve()
                    if str(full_path).startswith(abs_input_path):
                        opened_files.add(Path(data_source_path).resolve() / full_path.relative_to(abs_input_path))

            from rdagent.scenarios.data_science.scen.utils import FileTreeGenerator

            tree_gen = FileTreeGenerator(allowed_paths=opened_files)  # 传入打开的文件过滤器
            opened_trace_lines = tree_gen.generate_tree(Path(data_source_path).resolve())
            # 限制：期望训练和测试是不同的文件。

        # 断言必要的文件已生成
        for f in ["submission.csv", "scores.csv"]:
            if not (implementation.workspace_path / f).exists():
                err_msg = f"{f} 不存在。模型未被转储。请确保即使通过直接加载已保存的模型文件绕过模型训练步骤，也创建了所需的文件，如 submission.csv 和 scores.csv。"
                return CoSTEERSingleFeedback(
                    execution=err_msg,
                    return_checking=err_msg,
                    code=err_msg,
                    final_decision=False,
                )

        # 检查分数是否包含 NaN (值)
        score_df = pd.read_csv((implementation.workspace_path / "scores.csv"), index_col=0)
        if score_df.isnull().values.any():
            nan_locations = score_df[score_df.isnull().any(axis=1)]
            err_msg = f"\n[错误] 分数数据帧在以下位置包含 NaN 值：\n{nan_locations}"
            return CoSTEERSingleFeedback(
                execution=err_msg,
                return_checking=err_msg,
                code=err_msg,
                final_decision=False,
            )

        # 执行后再次读取 submission.csv 和 scores.csv 的内容
        submission_content_after = (
            (implementation.workspace_path / "submission.csv").read_text()
            if (implementation.workspace_path / "submission.csv").exists()
            else NO_SUB
        )
        scores_content_after = (
            (implementation.workspace_path / "scores.csv").read_text()
            if (implementation.workspace_path / "scores.csv").exists()
            else NO_SCORE
        )

        # 构建发送给大语言模型的提示
        system_prompt = T(".prompts:dump_model_eval.system").r()
        user_prompt = T(".prompts:dump_model_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.all_codes,
            model_folder_files=model_folder_files,
            scores_content_before=scores_content_before,
            scores_content_after=scores_content_after,
            opened_trace_lines=opened_trace_lines,
        )

        # 使用大语言模型生成反馈
        csfb = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 如果检查级别为 "high"，则进行严格的内容比对
        if DS_RD_SETTING.model_dump_check_level == "high":
            # 检查执行前后文件内容是否发生变化
            if scores_content_before != scores_content_after:
                return_msg = "\n[错误] scores.csv 的内容已更改。请检查代码以确保模型已正确转储，并重新运行代码以直接使用模型而不重新训练它。"
                return_msg += f"\n之前:\n{scores_content_before}\n之后:\n{scores_content_after}"
                if submission_content_before != submission_content_after:
                    return_msg = "[错误] submission.csv 的内容已更改。请检查代码以确保模型已正确转储，并重新运行代码以直接使用模型而不重新训练它。"
                csfb.return_checking = (csfb.return_checking or "") + return_msg
        return csfb
