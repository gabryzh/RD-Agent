import argparse
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files as rfiles
from pathlib import Path
from typing import Callable, Type

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state
from streamlit_theme import st_theme

from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.evaluators import ModelSingleFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.core.scenario import Scenario
from rdagent.log.base import Message
from rdagent.log.storage import FileStorage
from rdagent.log.ui.qlib_report_figure import report_figure
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.kaggle.experiment.scenario import KGScenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario

# 设置页面配置
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")


# 解析命令行参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="日志目录的路径")
parser.add_argument("--debug", action="store_true", help="启用调试模式")
args = parser.parse_args()
if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"日志目录 `{main_log_path}` 不存在！")
        st.stop()
else:
    main_log_path = None

# Qlib 场景下选定的指标
QLIB_SELECTED_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
]

# 相似场景的元组
SIMILAR_SCENARIOS = (
    QlibModelScenario,
    QlibFactorScenario,
    QlibFactorFromReportScenario,
    QlibQuantScenario,
    KGScenario,
)


def filter_log_folders(main_log_path):
    """
    筛选并返回相对于主日志路径的日志文件夹。
    """
    folders = [folder.relative_to(main_log_path) for folder in main_log_path.iterdir() if folder.is_dir()]
    folders = sorted(folders, key=lambda x: x.name)
    return folders


# 初始化 session state
if "log_path" not in state:
    if main_log_path:
        state.log_path = filter_log_folders(main_log_path)[0]
    else:
        state.log_path = None
        st.toast(":red[**请设置日志路径！**]", icon="⚠️")
# ... 其他 session state 初始化 ...
if "scenario" not in state: state.scenario = None
if "fs" not in state: state.fs = None
if "msgs" not in state: state.msgs = defaultdict(lambda: defaultdict(list))
if "last_msg" not in state: state.last_msg = None
if "current_tags" not in state: state.current_tags = []
if "lround" not in state: state.lround = 0  # RD 循环回合
if "erounds" not in state: state.erounds = defaultdict(int)  # 每个 RD 循环中的演进回合
if "e_decisions" not in state: state.e_decisions = defaultdict(lambda: defaultdict(tuple))
if "hypotheses" not in state: state.hypotheses = defaultdict(None) # 每个 RD 循环中的假设
if "h_decisions" not in state: state.h_decisions = defaultdict(bool)
if "metric_series" not in state: state.metric_series = []
if "all_metric_series" not in state: state.all_metric_series = []
if "alpha_baseline_metrics" not in state: state.alpha_baseline_metrics = None # 因子任务基线


def should_display(msg: Message):
    """
    判断是否应显示某条日志消息。
    """
    for t in state.excluded_tags + ["debug_tpl", "debug_llm"]:
        if t in msg.tag.split("."):
            return False
    if type(msg.content).__name__ in state.excluded_types:
        return False
    return True


def get_msgs_until(end_func: Callable[[Message], bool] = lambda _: True):
    """
    从日志文件中获取消息，直到满足 `end_func` 条件。
    """
    if state.fs:
        while True:
            try:
                msg = next(state.fs)
                if should_display(msg):
                    # ... 日志消息处理和状态更新 ...
                    tags = msg.tag.split(".")
                    if "hypothesis generation" in msg.tag:
                        state.lround += 1
                    msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
                    msg.tag = re.sub(r"Loop_\d+\.[^.]+", "", msg.tag)
                    msg.tag = re.sub(r"\.\.", ".", msg.tag)
                    msg.tag = re.sub(r"init\.", "", msg.tag)
                    msg.tag = re.sub(r"r\.", "", msg.tag)
                    msg.tag = re.sub(r"d\.", "", msg.tag)
                    msg.tag = re.sub(r"ef\.", "", msg.tag)
                    msg.tag = msg.tag.strip(".")
                    if "evolving code" not in state.current_tags and "evolving code" in tags:
                        state.erounds[state.lround] += 1
                    state.current_tags = tags
                    state.last_msg = msg

                    # 更新摘要信息
                    if "runner result" in tags:
                        # ... 处理不同场景下的指标 ...
                        pass
                    elif "hypothesis generation" in tags:
                        state.hypotheses[state.lround] = msg.content
                    elif "evolving code" in tags:
                        msg.content = [i for i in msg.content if i]
                    elif "evolving feedback" in tags:
                        # ... 处理演进反馈 ...
                        pass
                    elif "feedback" in tags and isinstance(msg.content, HypothesisFeedback):
                        state.h_decisions[state.lround] = msg.content.decision

                    state.msgs[state.lround][msg.tag].append(msg)
                    if end_func(msg):
                        break
            except StopIteration:
                st.toast(":red[**没有更多日志可显示！**]", icon="🛑")
                break


def refresh(same_trace: bool = False):
    """
    刷新页面，重新加载日志数据。
    """
    if state.log_path is None:
        st.toast(":red[**请设置日志路径！**]", icon="⚠️")
        return

    # ... 重置 session state ...
    if main_log_path:
        state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    else:
        state.fs = FileStorage(state.log_path).iter_msg()
    if not same_trace:
        get_msgs_until(lambda m: isinstance(m.content, Scenario))
        if state.last_msg is None or not isinstance(state.last_msg.content, Scenario):
            st.write(state.msgs)
            st.toast(":red[**未检测到场景信息**]", icon="❗")
            state.scenario = None
        else:
            state.scenario = state.last_msg.content
            st.toast(f":green[**检测到场景信息**] *{type(state.scenario).__name__}*", icon="✅")

    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.all_metric_series = []
    state.last_msg = None
    state.current_tags = []
    state.alpha_baseline_metrics = None


def evolving_feedback_window(wsf: FactorSingleFeedback | ModelSingleFeedback):
    """
    显示演进反馈的窗口。
    """
    # ... 根据反馈类型显示不同的 Tabs ...


def display_hypotheses(hypotheses: dict[int, Hypothesis], decisions: dict[int, bool], success_only: bool = False):
    """
    显示假设信息。
    """
    # ... 将假设数据转换为 DataFrame 并用样式显示 ...


def metrics_window(df: pd.DataFrame, R: int, C: int, *, height: int = 300, colors: list[str] = None):
    """
    显示指标图表的窗口。
    """
    # ... 使用 Plotly 创建子图并显示指标 ...


def summary_window():
    """
    显示摘要信息的窗口，包括指标和假设。
    """
    # ... 根据场景类型显示不同的摘要信息 ...


def tabs_hint():
    """
    显示一个关于如何导航 Tabs 的提示。
    """
    st.markdown(
        "<p style='font-size: small; color: #888888;'>您可以使用 ⬅️ ➡️ 或按住 Shift 并用鼠标滚轮🖱️来浏览选项卡。</p>",
        unsafe_allow_html=True,
    )


def tasks_window(tasks: list[FactorTask | ModelTask]):
    """
    显示任务信息的窗口。
    """
    # ... 根据任务类型（因子或模型）显示不同的信息 ...


def research_window():
    """
    显示研究阶段信息的窗口。
    """
    # ... 显示 PDF 截图、假设、实验生成等信息 ...


def feedback_window():
    """
    显示反馈阶段信息的窗口。
    """
    # ... 显示回测图表、假设反馈、下载按钮等 ...


def evolving_window():
    """
    显示开发（演进）阶段信息的窗口。
    """
    # ... 显示演进状态、代码、反馈等 ...


# 侧边栏配置
with st.sidebar:
    st.markdown("# RD-Agent🤖  [:grey[@GitHub]](https://github.com/microsoft/RD-Agent)")
    st.subheader(":blue[目录]", divider="blue")
    st.markdown(toc) # 显示目录
    st.subheader(":orange[控制面板]", divider="red")
    # ... 放置控制组件，如日志路径选择、刷新按钮等 ...


# 调试信息窗口
if debug:
    # ... 显示调试信息 ...


if state.log_path and state.fs is None:
    refresh()

# 主窗口
# ... 显示标题、流程图和场景描述 ...


def analyze_task_completion():
    """
    分析并显示任务完成情况。
    """
    # ... 计算并显示每个循环中任务的完成率 ...


# 根据场景显示不同的窗口
if state.scenario is not None:
    summary_window()
    if st.toggle("显示任务完成情况分析"):
        analyze_task_completion()

    # R&D 循环窗口
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        # ... 显示 R&D 循环相关的窗口 ...
        pass
    elif isinstance(state.scenario, GeneralModelScenario):
        # ... 显示通用模型场景相关的窗口 ...
        pass
    else:
        st.error("未知场景！")
        st.stop()

    with rf_c:
        research_window()
        feedback_window()
    with d_c.container(border=True):
        evolving_window()

# 免责声明
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("#### 免责声明")
st.markdown(
    "*此内容由 AI 生成，可能不完全准确或最新；对于关键问题，请咨询专业人士进行核实。*",
    unsafe_allow_html=True,
)
