import hashlib
import json
import pickle
import random
import re
from collections import defaultdict
from datetime import time, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from litellm import get_valid_models
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.utils import (
    curve_figure,
    get_sota_exp_stat,
    load_times_info,
    timeline_figure,
    trace_figure,
)
from rdagent.log.utils import (
    LogColors,
    extract_evoid,
    extract_json,
    extract_loopid_func_name,
    is_valid_session,
)
from rdagent.oai.backend.litellm import LITELLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

# 导入必要的类以进行响应格式化
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
    CodingSketch,
    HypothesisList,
    ScenarioChallenges,
    TraceChallenges,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict

# 初始化 session state
if "show_stdout" not in state: state.show_stdout = False
if "show_llm_log" not in state: state.show_llm_log = False
if "data" not in state: state.data = defaultdict(lambda: defaultdict(dict))
if "llm_data" not in state: state.llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
if "log_path" not in state: state.log_path = None
if "log_folder" not in state: state.log_folder = Path("./log")
if "sota_info" not in state: state.sota_info = None

# LiteLLM 相关设置
available_models = get_valid_models()
LITELLM_SETTINGS.dump_chat_cache = False
LITELLM_SETTINGS.dump_embedding_cache = False
LITELLM_SETTINGS.use_chat_cache = False
LITELLM_SETTINGS.use_embedding_cache = False


def convert_defaultdict_to_dict(d):
    """递归地将 defaultdict 转换为 dict。"""
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def load_data(log_path: Path):
    """从日志路径加载数据。"""
    data = defaultdict(lambda: defaultdict(dict))
    llm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    token_costs = defaultdict(list)

    # 从 FileStorage 迭代消息
    for msg in FileStorage(log_path).iter_msg():
        # ... (解析消息标签，分类存储到 data, llm_data, token_costs) ...
        pass

    # 兼容旧版日志格式
    llm_log_p = log_path / "debug_llm.pkl"
    if llm_log_p.exists():
        # ... (读取旧版的 llm 日志) ...
        pass

    return (
        convert_defaultdict_to_dict(data),
        convert_defaultdict_to_dict(llm_data),
        convert_defaultdict_to_dict(token_costs),
    )

# 根据配置决定是否缓存数据加载
if UI_SETTING.enable_cache:
    load_data = st.cache_data(persist=True)(load_data)


def load_stdout(stdout_path: Path):
    """加载标准输出文件。"""
    if stdout_path.exists():
        stdout = stdout_path.read_text()
    else:
        stdout = f"请设置: {stdout_path}"
    return stdout


# UI 窗口函数
def task_win(task):
    """显示任务信息的窗口。"""
    with st.expander(f"**:violet[{task.name}]**", expanded=False):
        # ... (显示任务的详细信息) ...
        pass


def workspace_win(workspace, cmp_workspace=None, cmp_name="last code."):
    """显示工作空间文件和差异的窗口。"""
    # ... (显示文件内容、与上一个版本的差异、运行时间等) ...
    pass


# 帮助函数
def show_text(text, lang=None):
    """显示文本代码块。"""
    # ...


def highlight_prompts_uri(uri):
    """高亮 URI 的格式。"""
    # ...


def llm_log_win(llm_d: list):
    """显示 LLM 交互日志的窗口。"""
    # ... (显示模板渲染、用户/系统提示、响应等) ...
    pass


def hypothesis_win(hypo):
    """显示假设信息的窗口。"""
    # ...


def exp_gen_win(exp_gen_data, llm_data=None):
    """显示实验生成阶段信息的窗口。"""
    st.header("实验生成", divider="blue", anchor="exp-gen")
    # ...


def evolving_win(data, key, llm_data=None, base_workspace=None):
    """显示演进（编码）阶段信息的窗口。"""
    with st.container(border=True):
        # ... (使用滑块选择演进轮次，显示代码、反馈等) ...
        pass


def coding_win(data, base_exp, llm_data: dict | None = None):
    """显示编码阶段（包含多个演进任务）信息的窗口。"""
    st.header("编码", divider="blue", anchor="coding")
    # ...


def running_win(data, base_exp, llm_data=None, last_sota_exp=None):
    """显示运行阶段信息的窗口。"""
    st.header("运行", divider="blue", anchor="running")
    # ... (显示最终的工作空间、结果、MLE 分数等) ...
    pass


def feedback_win(fb_data, llm_data=None):
    """显示反馈阶段信息的窗口。"""
    # ... (显示反馈内容和决策) ...
    pass


def sota_win(sota_exp, trace):
    """显示 SOTA（State-of-the-Art）实验信息的窗口。"""
    st.subheader("SOTA 实验", divider="rainbow", anchor="sota-exp")
    # ...


def main_win(loop_id, llm_data=None):
    """显示单个循环所有阶段信息的主窗口。"""
    loop_data = state.data[loop_id]
    exp_gen_win(loop_data["direct_exp_gen"], llm_data["direct_exp_gen"] if llm_data else None)
    if "coding" in loop_data:
        coding_win(...)
    if "running" in loop_data:
        running_win(...)
    if "feedback" in loop_data:
        feedback_win(...)
    if "record" in loop_data and "SOTA experiment" in loop_data["record"]:
        st.header("记录", divider="violet", anchor="record")
        sota_win(...)


def replace_ep_path(p: Path):
    """替换 workspace 路径以适应不同的机器环境。"""
    # ...


def get_llm_call_stats(llm_data: dict) -> tuple[int, int]:
    """获取 LLM 调用统计信息。"""
    # ...


def get_timeout_stats(llm_data: dict):
    """获取超时统计信息。"""
    # ...


def timedelta_to_str(td: timedelta | None) -> str:
    """将 timedelta 对象转换为 HH:MM:SS 格式的字符串。"""
    # ...


def summarize_win():
    """显示摘要信息的窗口。"""
    st.header("摘要", divider="rainbow")
    with st.container(border=True):
        # ... (显示各种统计信息、图表、表格等) ...
        pass


def stdout_win(loop_id: int):
    """显示指定循环的标准输出。"""
    # ...


def get_folders_sorted(log_path, sort_by_time=False):
    """获取排序后的文件夹列表。"""
    # ...


# UI - 侧边栏
with st.sidebar:
    # ... (放置用于选择日志文件夹、刷新数据、切换显示选项的控件) ...
    pass


def get_state_data_range(state_data):
    """获取 state.data 中有效的循环 ID 范围。"""
    keys = [k for k in state_data.keys() if isinstance(k, int) and "direct_exp_gen" in state_data[k] and "no_tag" in state_data[k]["direct_exp_gen"]]
    return min(keys), max(keys)


# UI - 主界面
if "competition" in state.data:
    st.title(f"{state.data['competition']} ([分享链接](/ds_trace?log_folder={state.log_folder}&selection={state.log_path}))")
    summarize_win()
    min_id, max_id = get_state_data_range(state.data)
    loop_id = st.slider("循环", min_id, max_id, min_id) if max_id > min_id else min_id
    if state.show_stdout:
        stdout_win(loop_id)
    main_win(loop_id, state.llm_data[loop_id] if loop_id in state.llm_data else None)
