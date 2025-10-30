from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.log.ui.conf import UI_SETTING


def convert_log_folder_str(lf: str) -> str:
    """
    转换日志文件夹字符串。如果只提供了 amlt 名称，则自动补全为完整路径。
    """
    if "/" not in lf:
        return f"{UI_SETTING.amlt_path}/{lf.strip()}/combined_logs"
    return lf.strip()


def extract_amlt_name(x: str) -> str:
    """
    从完整路径中提取 amlt 名称。
    """
    if "amlt" not in x:
        return x
    return x[x.rfind("amlt") + 5 :].split("/")[0]


# 初始化 session state
if "log_folder" not in state:
    state.log_folder = Path("./log")
if "log_folders" not in state:
    state.log_folders = [convert_log_folder_str(i) for i in UI_SETTING.default_log_folders]

# 定义页面
summary_page = st.Page("ds_summary.py", title="摘要", icon="📊")
trace_page = st.Page("ds_trace.py", title="追踪", icon="📈")
aide_page = st.Page("aide.py", title="Aide", icon="🧑‍🏫")

# 设置页面配置
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")

# 运行导航，这将根据 URL 渲染相应的页面
st.navigation([summary_page, trace_page, aide_page]).run()


# 侧边栏
with st.sidebar:
    st.subheader("页面", divider="rainbow")
    st.page_link(summary_page, icon="📊")
    st.page_link(trace_page, icon="📈")
    st.page_link(aide_page, icon="🧑‍🏫")

    st.subheader("设置", divider="rainbow")
    # 使用表单让用户输入和确认日志文件夹路径
    with st.form("log_folder_form", border=False):
        log_folder_str = st.text_area(
            "**日志文件夹**(用';'分隔)", value=";".join(extract_amlt_name(i) for i in state.log_folders)
        )
        if st.form_submit_button("确认"):
            # 更新 session state 中的日志文件夹列表
            state.log_folders = [
                convert_log_folder_str(folder) for folder in log_folder_str.split(";") if folder.strip()
            ]
            st.rerun()
