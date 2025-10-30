import argparse
import json
import pickle
import re
import time
from pathlib import Path

import streamlit as st
from streamlit import session_state

from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.utils import extract_evoid, extract_loopid_func_name

# 设置页面配置
st.set_page_config(layout="wide", page_title="debug_llm", page_icon="🎓", initial_sidebar_state="expanded")

# 解析命令行参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="日志目录的路径")
args = parser.parse_args()


def get_folders_sorted(log_path):
    """缓存并返回排序后的文件夹列表，并显示加载进度"""
    with st.spinner("正在加载文件夹列表..."):
        folders = sorted(
            (folder for folder in log_path.iterdir() if folder.is_dir() and list(folder.iterdir())),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        st.write(f"找到 {len(folders)} 个文件夹")
    return [folder.name for folder in folders]


if UI_SETTING.enable_cache:
    get_folders_sorted = st.cache_data(get_folders_sorted)


# 设置主日志路径
main_log_path = Path(args.log_dir) if args.log_dir else Path("./log")
if not main_log_path.exists():
    st.error(f"日志目录 {main_log_path} 不存在！")
    st.stop()

# 初始化 session state
if "data" not in session_state:
    session_state.data = []
if "log_path" not in session_state:
    session_state.log_path = None

tlist = []


def load_data():
    """加载数据到 session_state 并显示进度"""
    log_file = main_log_path / session_state.log_path / "debug_llm.pkl"
    try:
        with st.spinner(f"正在加载数据文件 {log_file}..."):
            start_time = time.time()
            with open(log_file, "rb") as f:
                session_state.data = pickle.load(f)
            st.success(f"数据加载完成！耗时 {time.time() - start_time:.2f} 秒")
            st.session_state["current_loop"] = 1
    except Exception as e:
        session_state.data = [{"error": str(e)}]
        st.error(f"加载数据失败: {e}")


# UI - 侧边栏
with st.sidebar:
    st.markdown(":blue[**日志路径**]")
    manually = st.toggle("手动输入")
    if manually:
        st.text_input("日志路径", key="log_path", label_visibility="collapsed")
    else:
        folders = get_folders_sorted(main_log_path)
        st.selectbox(f"**从 {main_log_path.absolute()} 中选择**", folders, key="log_path")

    if st.button("刷新数据"):
        load_data()
        st.rerun()


# 帮助函数
def show_text(text, lang=None):
    """显示文本代码块"""
    # ...


def highlight_prompts_uri(uri):
    """高亮 URI 的格式"""
    parts = uri.split(":")
    return f"**{parts[0]}:**:green[**{parts[1]}**]"


# 显示数据
progress_text = st.empty()
progress_bar = st.progress(0)

# 每页显示一个 Loop
LOOPS_PER_PAGE = 1

# 获取所有的 Loop ID 并分组
loop_groups = {}
for i, d in enumerate(session_state.data):
    tag = d["tag"]
    loop_id, _ = extract_loopid_func_name(tag)
    if loop_id:
        if loop_id not in loop_groups:
            loop_groups[loop_id] = []
        loop_groups[loop_id].append(d)

# 按 Loop ID 排序
sorted_loop_ids = sorted(loop_groups.keys(), key=int)
total_loops = len(sorted_loop_ids)
total_pages = total_loops

if total_pages:
    # 初始化 current_loop
    if "current_loop" not in st.session_state:
        st.session_state["current_loop"] = 1

    # Loop 导航按钮
    col1, col2, col3, col4, col5 = st.sidebar.columns([1.2, 1, 2, 1, 1.2])
    # ... (首页、上一页、下一页、末页按钮和下拉选择框) ...

    # 获取当前 Loop
    current_loop = st.session_state["current_loop"]

    # 渲染当前 Loop 数据
    loop_id = sorted_loop_ids[current_loop - 1]
    progress_text.text(f"正在处理 Loop {loop_id}...")
    progress_bar.progress(current_loop / total_loops, text=f"Loop :green[**{current_loop}**] / {total_loops}")

    # 渲染 Loop 标题
    loop_anchor = f"Loop_{loop_id}"
    if loop_anchor not in tlist:
        tlist.append(loop_anchor)
        st.header(loop_anchor, anchor=loop_anchor, divider="blue")

    # 渲染当前 Loop 的所有数据
    loop_data = loop_groups[loop_id]
    for d in loop_data:
        # ... (解析 tag，获取 func_name, evo_id) ...

        # 渲染函数和演进步骤的标题
        # ...

        # 根据 tag 渲染内容 (模板、LLM 交互等)
        if "debug_tpl" in tag:
            # ... (显示模板渲染信息) ...
            pass
        elif "debug_llm" in tag:
            # ... (显示 LLM 交互信息) ...
            pass

    progress_text.text("当前 Loop 数据处理完成！")

    # 侧边栏目录
    with st.sidebar:
        toc = "\n".join([f"- [{t}](#{t})" if t.startswith("L") else f"  - [{t.split('.')[1]}](#{t})" for t in tlist])
        st.markdown(toc, unsafe_allow_html=True)
