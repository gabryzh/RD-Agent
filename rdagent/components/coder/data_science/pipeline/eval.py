# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
# 导入必要的库
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 导入 rdagent 内部模块
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.agent.context7 import Agent as DocAgent
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.share.notebook import NotebookConverter
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.test_eval import get_test_eval
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent


@dataclass
class DSCoderFeedback(CoSTEERSingleFeedback):
    """
    用于数据科学 CoSTEER 评估的反馈。
    此反馈用于评估数据科学 CoSTEER 任务的代码和执行情况。
    """

    requires_documentation_search: bool | None = None  # 保持为 None 意味着该功能被禁用
    error_message: str | None = None # 错误信息

    @staticmethod
    def val_and_update_init_dict(data: dict) -> dict:
        """
        验证并更新初始化字典。
        这个静态方法首先调用父类的验证方法来处理基础字段，
        然后验证这个类新增的字段。
        """
        # 首先调用父类的验证方法来处理基础字段
        data = CoSTEERSingleFeedback.val_and_update_init_dict(data)

        # 验证新字段
        if "requires_documentation_search" in data:
            # 处理字符串形式的布尔值
            if isinstance(data["requires_documentation_search"], str):
                if data["requires_documentation_search"].lower() == "false":
                    data["requires_documentation_search"] = False
                elif data["requires_documentation_search"].lower() == "true":
                    data["requires_documentation_search"] = True
                else:
                    raise ValueError(
                        f"'requires_documentation_search' 字符串值必须是 'true', 'True', 'false', 或 'False', 而不是 '{data['requires_documentation_search']}'"
                    )
            # 验证类型是否正确
            elif data["requires_documentation_search"] is not None and not isinstance(
                data["requires_documentation_search"], bool
            ):
                raise ValueError(
                    f"'requires_documentation_search' 必须是布尔值、字符串或 None, 而不是 {type(data['requires_documentation_search'])}"
                )

        if "error_message" in data:
            if data["error_message"] is not None and not isinstance(data["error_message"], str):
                raise ValueError(f"'error_message' 必须是字符串或 None, 而不是 {type(data['error_message'])}")

        return data

    def __str__(self) -> str:
        """
        返回该反馈对象的可读字符串表示。
        """
        base_str = super().__str__()

        # 如果需要文档搜索，则在字符串中添加相关信息
        if self.requires_documentation_search is not None:
            base_str += f"-------------------需要文档搜索------------------\n{self.requires_documentation_search}\n"

        # 如果有错误信息，则在字符串中添加
        if self.error_message is not None:
            # 检查错误信息是否包含 Context7 文档搜索结果
            if "### API Documentation Reference:" in self.error_message:
                base_str += f"-------------------错误分析与文档搜索结果------------------\n{self.error_message}\n"
            else:
                base_str += f"-------------------错误信息------------------\n{self.error_message}\n"

        return base_str

    @classmethod
    def merge(cls, feedback_li: list[CoSTEERSingleFeedback]) -> "DSCoderFeedback":
        """
        将多个反馈对象合并成一个。
        """
        # 调用父类的合并方法来处理基础字段
        merged_fb = super().merge(feedback_li)

        # 如果需要，将合并后的反馈转换为 DSCoderFeedback 类型
        if not isinstance(merged_fb, DSCoderFeedback):
            merged_fb = DSCoderFeedback(
                execution=merged_fb.execution,
                return_checking=merged_fb.return_checking,
                code=merged_fb.code,
                final_decision=merged_fb.final_decision,
            )

        # 合并 error_message 字段
        error_messages = [
            fb.error_message for fb in feedback_li if isinstance(fb, DSCoderFeedback) and fb.error_message is not None
        ]
        if error_messages:
            merged_fb.error_message = "\n\n".join(error_messages)

        # 合并 requires_documentation_search 字段 (任何一个为 True 则结果为 True)
        requires_search = [
            fb.requires_documentation_search
            for fb in feedback_li
            if isinstance(fb, DSCoderFeedback) and fb.requires_documentation_search is not None
        ]
        if requires_search:
            merged_fb.requires_documentation_search = any(requires_search)

        return merged_fb


# PipelineSingleFeedback 是 DSCoderFeedback 的别名，用于保持兼容性
PipelineSingleFeedback = DSCoderFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback


class PipelineCoSTEEREvaluator(CoSTEEREvaluator):
    """
    Pipeline CoSTEER 评估器。
    这个类负责评估数据科学任务的代码实现。
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledgeV2 = None,
        **kwargs,
    ) -> PipelineSingleFeedback:
        """
        评估给定的任务实现。
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
            return PipelineSingleFeedback(
                execution="此任务失败次数过多，跳过实现。",
                return_checking="此任务失败次数过多，跳过实现。",
                code="此任务失败次数过多，跳过实现。",
                error_message="此任务失败次数过多，跳过实现。",
                requires_documentation_search=None,
                final_decision=False,
            )

        # 获取数据科学环境配置
        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        stdout = ""
        # 清理工作空间
        implementation.execute(env=env, entry=get_clear_ws_cmd())
        # 根据设置决定是运行在采样数据上还是完整数据上
        if DS_RD_SETTING.sample_data_by_LLM:
            # 因为coder在完整数据上运行，我们需要预先运行调试模式以节省时间
            result = implementation.run(
                env=env, entry=f"strace -e trace=file -f -o trace.log python -m coverage run main.py --debug"
            )
        else:
            result = implementation.run(
                env=env, entry=f"strace -e trace=file -f -o trace.log python -m coverage run main.py"
            )
        result_stdout = result.get_truncated_stdout()

        nb_conversion_ret_code = 0
        nb_conversion_check_text = ""
        # 如果启用了 notebook 转换
        if DS_RD_SETTING.enable_notebook_conversion:
            notebook_converter = NotebookConverter()
            code = implementation.file_dict["main.py"]
            error_msg = notebook_converter.validate_code_format(code)
            if error_msg is not None:
                nb_conversion_check_text = error_msg
                nb_conversion_ret_code = 1
            else:
                notebook_converter.convert(
                    task=target_task,
                    code=code,
                    stdout=result_stdout,
                    outfile=implementation.workspace_path / "main.ipynb",
                    use_debug_flag=DS_RD_SETTING.sample_data_by_LLM,
                )

        sample_submission_check = True
        test_eval = get_test_eval()
        # 检查代码是否违规打开了样本提交文件
        if (sample_submission_file_name := test_eval.get_sample_submission_name(self.scen.competition)) is not None:
            if (implementation.workspace_path / "trace.log").exists():
                opened_trace_lines = [
                    line
                    for line in (implementation.workspace_path / "trace.log").read_text().splitlines()
                    if "openat" in line and sample_submission_file_name in line
                ]
                if len(opened_trace_lines) > 0:
                    stdout += f"代码在执行期间打开了样本提交文件 '{sample_submission_file_name}'。\n拒绝该实现！\n"
                    sample_submission_check = False

        # 移除 EDA（探索性数据分析）部分的输出
        result_stdout = remove_eda_part(result_stdout)
        if result.exit_code != 0:
            stdout += f"代码运行失败。请检查 stdout：\n以下是调试模式运行的 stdout：\n{result_stdout.strip()}\n"
        else:
            stdout += f"代码运行成功。\n以下是调试模式运行的 stdout：\n{result_stdout.strip()}\n"

        # 如果使用了采样数据，则记录调试时间和预估的完整运行时间
        if DS_RD_SETTING.sample_data_by_LLM:
            debug_time, full_estimated_time = None, None
            if match := re.search(r"debug_time:\s*(\d+(?:.\d+)?)", result_stdout, re.DOTALL):
                debug_time = float(match.group(1))
            if match := re.search(r"estimated_time:\s*(\d+(?:.\d+)?)", result_stdout, re.DOTALL):
                full_estimated_time = float(match.group(1))
            if debug_time is not None and full_estimated_time is not None:
                stdout += f"调试模式运行耗时 {debug_time:.2f} 秒，预计完整运行时间为 {full_estimated_time:.2f} 秒。预计时间是调试时间的 {full_estimated_time / env.conf.running_timeout_period * 100:.2f}%。"
            else:
                stdout += "调试模式未提供 debug_time 或 estimated_time，这是一个有问题的实现。\n"

        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        # 检查生成的 scores.csv 文件
        if not score_fp.exists():
            score_check_text = "[错误] 指标文件 (scores.csv) 未生成！"
            score_ret_code = 1
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)

                # 检查模型名称 (索引)
                if not score_df.index.is_unique:
                    score_check_text += "\n[错误] 文件 'scores.csv' 包含重复的模型名称。"
                    score_ret_code = 1
                if "ensemble" not in model_set_in_scores:
                    score_check_text += "\n[错误] 文件 'scores.csv' 不包含集成模型。"
                    score_ret_code = 1
                if score_ret_code != 0:
                    score_check_text += f"文件 'scores.csv' 中的数据帧是：\n{score_df}"

                # 检查指标名称 (列名) - 不区分大小写
                if [col.lower() for col in score_df.columns.tolist()] != [self.scen.metric_name.lower()]:
                    score_check_text += f"\n[错误] 分数数据帧不包含正确的列名。\n正确的列是：['{self.scen.metric_name}']\n但得到的是：{score_df.columns.tolist()}"
                    score_ret_code = 1

                # 检查分数是否包含 NaN (值)
                if score_df.isnull().values.any():
                    nan_locations = score_df[score_df.isnull().any(axis=1)]
                    score_check_text += f"\n[错误] 分数数据帧在以下位置包含 NaN 值：\n{nan_locations}"
                    score_ret_code = 1

            except Exception as e:
                score_check_text += f"\n[错误] 检查 scores.csv 文件时出错: {e}\nscores.csv 的内容是:\n-----\n{score_fp.read_text()}\n-----"
                score_ret_code = 1

        test_eval = get_test_eval()
        # 验证提交文件的格式
        if DS_RD_SETTING.sample_data_by_LLM and test_eval.enabled(self.scen.competition):
            submission_check_out, submission_ret_code = test_eval.valid(self.scen.competition, implementation)
            stdout += f"\n### 提交检查:\n{submission_check_out}\n如果提交检查返回 'Submission is valid' 或类似消息，即使有一些警告消息，您仍应将提交视为有效并给出积极的最终决定。"
        elif not test_eval.is_sub_enabled(self.scen.competition):
            submission_ret_code = 0
        else:
            # 检查提交文件
            base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
            implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
            submission_result = implementation.run(env=env, entry="python test/submission_format_test.py")
            submission_check_out = submission_result.get_truncated_stdout()
            submission_ret_code = submission_result.exit_code
            stdout += "\n" + submission_check_out

        if not isinstance(implementation, FBWorkspace):
            eda_output = None
        else:
            eda_output = implementation.file_dict.get("EDA.md", None)

        # 从数据科学配置中提取是否启用 mcp 文档搜索
        enable_mcp_documentation_search = DS_RD_SETTING.enable_mcp_documentation_search

        # 获取查询到的相似成功知识
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        # 构建发送给 LLM 的 prompt
        system_prompt = T(".prompts:pipeline_eval.system").r(
            is_sub_enabled=test_eval.is_sub_enabled(self.scen.competition),
            debug_mode=DS_RD_SETTING.sample_data_by_LLM,
            enable_mcp_documentation_search=enable_mcp_documentation_search,
            mle_check=DS_RD_SETTING.sample_data_by_LLM,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:pipeline_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=eda_output),
            task_desc=target_task.get_task_information(),
            stdout=stdout.strip(),
            spec=T("scenarios.data_science.share:component_spec.Pipeline").r(
                metric_name=self.scen.metric_name,
                enable_notebook_conversion=DS_RD_SETTING.enable_notebook_conversion,
            ),
            code=implementation.file_dict["main.py"],
        )
        # 使用 LLM 从 JSON 构建反馈类，并带重试机制
        wfb = build_cls_from_json_with_retry(
            PipelineSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=PipelineSingleFeedback.val_and_update_init_dict,
        )

        # 判断是否应该执行文档搜索
        do_documentation_search = enable_mcp_documentation_search and wfb.requires_documentation_search

        if do_documentation_search:
            # 使用 MCPAgent 提供简洁、用户友好的接口
            try:
                # 创建针对 Context7 服务的 agent - 模型配置来自 mcp_config.json
                doc_agent = DocAgent()

                # 同步查询 - 非常适合评估上下文
                if wfb.error_message:  # 类型安全检查
                    context7_result = doc_agent.query(query=wfb.error_message)

                    if context7_result:
                        logger.info("Context7: 文档搜索成功完成")
                        wfb.error_message += f"\n\n### API 文档参考:\n根据错误检索到以下 API 文档。这仅提供有关 API 更改或参数规格的事实信息：\n\n{context7_result}"
                    else:
                        logger.warning("Context7: 文档搜索失败或未找到结果")
                else:
                    logger.warning("Context7: 没有可供搜索的错误信息")

            # TODO: 确认超时会引发什么异常
            # except concurrent.futures.TimeoutError:
            #     logger.error("Context7: 查询在 180 秒后超时")
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                logger.error(f"Context7: 查询失败 - {error_msg}")

        # 如果硬性检查失败，则将最终决定覆盖为 False
        if score_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\n" + score_check_text
        if submission_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\n提交文件检查失败。"
        if sample_submission_check is False and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += (
                "\n样本提交文件检查失败。代码不应打开样本提交文件。"
            )
        if nb_conversion_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\n" + nb_conversion_check_text
        return wfb
