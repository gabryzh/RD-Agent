"""
该模块提供了一些用于处理日志文件夹的实用函数。
"""

import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

from rdagent.utils.workflow import LoopBase


def get_first_session_file_after_duration(log_folder: str | Path, duration: str | pd.Timedelta) -> Path:
    """
    在指定持续时间后获取第一个会话文件。

    Args:
        log_folder (str | Path): 日志文件夹的路径。
        duration (str | pd.Timedelta): 持续时间。

    Returns:
        Path: 第一个会话文件的路径。
    """
    log_folder = Path(log_folder)
    duration_dt = pd.Timedelta(duration)
    # 按升序迭代转储步骤
    files = sorted(
        (log_folder / "__session__").glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0]))
    )
    fp = None
    for fp in files:
        with fp.open("rb") as f:
            session_obj: LoopBase = pickle.load(f)
        timer = session_obj.timer
        all_duration = timer.all_duration
        remain_time_duration = timer.remain_time()
        if all_duration is None or remain_time_duration is None:
            msg = "计时器未配置"
            raise ValueError(msg)
        time_spent = all_duration - remain_time_duration
        if time_spent >= duration_dt:
            break
    if fp is None:
        msg = f"在持续时间 {duration} 后未找到会话文件"
        raise ValueError(msg)
    return fp


def first_li_si_after_one_time(log_path: Path, hours: int = 12) -> tuple[int, int, str]:
    """
    根据小时数，找到停止的循环 ID 和步骤 ID（在 <hours> 小时后的第一个步骤）。

    Args:
        log_path (Path): 日志文件夹的路径（包含许多日志跟踪）。
        hours (int): 用于统计的小时数。

    Returns:
        tuple[int, int, str]: 循环 ID、步骤 ID 和函数名。
    """
    session_path = log_path / "__session__"
    max_li = max(int(p.name) for p in session_path.iterdir() if p.is_dir() and p.name.isdigit())
    max_step = max(int(p.name.split("_")[0]) for p in (session_path / str(max_li)).iterdir() if p.is_file())
    rdloop_obj_p = next((session_path / str(max_li)).glob(f"{max_step}_*"))

    rdloop_obj = DataScienceRDLoop.load(rdloop_obj_p)
    loop_trace = rdloop_obj.loop_trace
    si2fn = rdloop_obj.steps

    duration = timedelta(seconds=0)
    for li, lts in loop_trace.items():
        for lt in lts:
            si = lt.step_idx
            duration += lt.end - lt.start
            if duration > timedelta(hours=hours):
                return li, si, si2fn[si]


if __name__ == "__main__":
    from rdagent.app.data_science.loop import DataScienceRDLoop

    f = get_first_session_file_after_duration("<path to log aptos2019-blindness-detection>", pd.Timedelta("12h"))

    with f.open("rb") as f:
        session_obj: LoopBase = pickle.load(f)
    loop_trace = session_obj.loop_trace
    last_loop = loop_trace[max(loop_trace.keys())]
    last_step = last_loop[-1]
    session_obj.steps[last_step.step_idx]
