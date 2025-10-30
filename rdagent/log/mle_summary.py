import pickle
import traceback
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd

from rdagent.core.experiment import FBWorkspace
from rdagent.core.proposal import ExperimentFeedback
from rdagent.log.storage import FileStorage
from rdagent.log.utils import extract_json, extract_loopid_func_name, is_valid_session
from rdagent.log.utils.folder import get_first_session_file_after_duration
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.test_eval import (
    MLETestEval,
    NoTestEvalError,
    get_test_eval,
)

# from rdagent.scenarios.kaggle.kaggle_crawler import score_rank
from rdagent.utils.workflow import LoopBase


def save_grade_info(log_trace_path: Path):
    """
    为单个日志追踪保存评分信息。
    它会评估实验工作空间并记录 MLE 分数。
    """
    test_eval = get_test_eval()

    trace_storage = FileStorage(log_trace_path)
    for msg in trace_storage.iter_msg(tag="competition"):
        competition = msg.content

    for msg in trace_storage.iter_msg(tag="running"):
        if isinstance(msg.content, DSExperiment):
            try:
                mle_score_str = test_eval.eval(competition, msg.content.experiment_workspace)
                trace_storage.log(
                    mle_score_str, tag=f"{msg.tag}.mle_score.pid", save_type="pkl", timestamp=msg.timestamp
                )
            except Exception as e:
                print(f"在 {log_trace_path} 中出错: {e}", traceback.format_exc())


def save_all_grade_info(log_folder: str | Path) -> None:
    """
    为日志文件夹中的所有有效会话保存评分信息。
    """
    for log_trace_path in Path(log_folder).iterdir():
        if is_valid_session(log_trace_path):
            try:
                save_grade_info(log_trace_path)
            except NoTestEvalError as e:
                print(f"在 {log_trace_path} 中出错: {e}", traceback.format_exc())


def _get_loop_and_fn_after_hours(log_folder: Path, hours: int):
    """
    获取指定小时数后的第一个会话的循环ID和函数名。
    """
    stop_session_fp = get_first_session_file_after_duration(log_folder, f"{hours}h")

    with stop_session_fp.open("rb") as f:
        session_obj: LoopBase = pickle.load(f)

    loop_trace = session_obj.loop_trace
    stop_li = max(loop_trace.keys())
    last_loop = loop_trace[stop_li]
    last_step = last_loop[-1]
    stop_fn = session_obj.steps[last_step.step_idx]
    print(f"停止循环: {stop_li=}, {stop_fn=}")
    files = sorted(
        (log_folder / "__session__").glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0]))
    )

    print(f"最大会话: {files[-1:]=}")
    return stop_li, stop_fn


def summarize_folder(log_folder: Path, hours: int | None = None) -> None:
    """
    总结日志文件夹并将摘要保存为 pickle 文件。
    参数:
        log_folder (Path): 日志文件夹的路径（包含许多日志追踪）。
        hours (int | None): 要统计的小时数。如果为 None，则统计全部。
    """
    test_eval = get_test_eval()

    is_mle = isinstance(test_eval, MLETestEval)

    log_folder = Path(log_folder)
    stat = defaultdict(dict)
    for log_trace_path in log_folder.iterdir():  # 一个日志追踪
        if not is_valid_session(log_trace_path):
            continue
        # 初始化统计变量
        loop_num, made_submission_num, valid_submission_num, above_median_num, get_medal_num = 0, 0, 0, 0, 0
        bronze_num, silver_num, gold_num = 0, 0, 0
        test_scores, test_ranks, valid_scores = {}, {}, {}
        bronze_threshold, silver_threshold, gold_threshold, median_threshold = 0.0, 0.0, 0.0, 0.0
        success_loop_num = 0
        sota_exp_stat, sota_exp_score, sota_exp_rank, grade_output = "", None, None, None

        if hours:
            stop_li, stop_fn = _get_loop_and_fn_after_hours(log_trace_path, hours)

        # 读取并排序日志消息
        msgs = [(msg, extract_loopid_func_name(msg.tag)) for msg in FileStorage(log_trace_path).iter_msg()]
        msgs = [(msg, int(loop_id) if loop_id else loop_id, fn) for msg, (loop_id, fn) in msgs]
        msgs.sort(key=lambda m: m[1] if m[1] else -1)  # 按循环ID排序

        for msg, loop_id, fn in msgs:  # 日志追踪中的消息
            if loop_id:
                loop_num = max(loop_id + 1, loop_num)
            if hours and loop_id == stop_li and fn == stop_fn:
                break
            if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
                if "competition" in msg.tag:
                    stat[log_trace_path.name]["competition"] = msg.content
                    # 获取阈值分数
                    if is_mle:
                        workflowexp = FBWorkspace()
                        stdout = workflowexp.execute(
                            env=test_eval.env,
                            entry=f"mlebench grade-sample None {stat[log_trace_path.name]['competition']} --data-dir /mle/data",
                        )
                        grade_output = extract_json(stdout)
                        if grade_output:
                            bronze_threshold = grade_output["bronze_threshold"]
                            silver_threshold = grade_output["silver_threshold"]
                            gold_threshold = grade_output["gold_threshold"]
                            median_threshold = grade_output["median_threshold"]

                elif "running" in msg.tag:
                    if isinstance(msg.content, DSExperiment):
                        if msg.content.result is not None:
                            valid_scores[loop_id] = msg.content.result
                    elif "mle_score" in msg.tag:
                        grade_output = extract_json(msg.content)
                        if grade_output:
                            if grade_output["submission_exists"]: made_submission_num += 1
                            if grade_output["score"] is not None: test_scores[loop_id] = grade_output["score"]
                            if grade_output["valid_submission"]: valid_submission_num += 1
                            if grade_output["above_median"]: above_median_num += 1
                            if grade_output["any_medal"]: get_medal_num += 1
                            if grade_output["bronze_medal"]: bronze_num += 1
                            if grade_output["silver_medal"]: silver_num += 1
                            if grade_output["gold_medal"]: gold_num += 1

                elif "feedback" in msg.tag and "evolving" not in msg.tag:
                    if isinstance(msg.content, ExperimentFeedback) and bool(msg.content):
                        success_loop_num += 1
                        if grade_output:  # SOTA 实验的评分输出
                            if grade_output["gold_medal"]: sota_exp_stat = "gold"
                            elif grade_output["silver_medal"]: sota_exp_stat = "silver"
                            elif grade_output["bronze_medal"]: sota_exp_stat = "bronze"
                            elif grade_output["above_median"]: sota_exp_stat = "above_median"
                            elif grade_output["valid_submission"]: sota_exp_stat = "valid_submission"
                            elif grade_output["submission_exists"]: sota_exp_stat = "made_submission"
                            if grade_output["score"] is not None: sota_exp_score = grade_output["score"]

        # 更新统计字典
        stat[log_trace_path.name].update({
                "loop_num": loop_num, "made_submission_num": made_submission_num, "valid_submission_num": valid_submission_num,
                "above_median_num": above_median_num, "get_medal_num": get_medal_num, "bronze_num": bronze_num,
                "silver_num": silver_num, "gold_num": gold_num, "test_scores": test_scores, "valid_scores": valid_scores,
                "success_loop_num": success_loop_num, "sota_exp_stat": sota_exp_stat, "sota_exp_score": sota_exp_score,
                "bronze_threshold": bronze_threshold, "silver_threshold": silver_threshold, "gold_threshold": gold_threshold,
                "median_threshold": median_threshold,
            })

    # 保存摘要
    save_name = f"summary_{hours}h.pkl" if hours else "summary.pkl"
    save_p = log_folder / save_name
    if save_p.exists():
        save_p.unlink()
        print(f"旧的 {save_name} 已删除。")
    pd.to_pickle(stat, save_p)


def grade_summary(log_folder: str) -> None:
    """
    为日志文件夹中的日志追踪生成测试分数并保存摘要。
    """
    log_folder = Path(log_folder)
    save_all_grade_info(log_folder)
    summarize_folder(log_folder)


if __name__ == "__main__":
    fire.Fire({
            "grade": save_all_grade_info,
            "summary": summarize_folder,
            "grade_summary": grade_summary,
        })
