import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from streamlit import session_state as state

from rdagent.app.data_science.conf import DS_RD_SETTING

# 设置页面配置
st.set_page_config(layout="wide", page_title="RD-Agent_user_interact", page_icon="🎓", initial_sidebar_state="expanded")

# 初始化 session state
if "sessions" not in state:
    state.sessions = {}
if "selected_session_name" not in state:
    state.selected_session_name = None


def render_main_content():
    """渲染主内容区域"""
    if state.selected_session_name is not None and state.selected_session_name in state.sessions:
        selected_session_data = state.sessions[state.selected_session_name]
        if selected_session_data is not None:
            # 显示会话和比赛信息
            st.title(
                f"会话: {state.selected_session_name[:4]}，比赛: {selected_session_data['competition']}"
            )
            st.title("上下文信息:")
            st.subheader("比赛场景:", divider=True)
            st.code(selected_session_data["scenario_description"], language="yaml")
            st.subheader("过往尝试总结:", divider=True)
            st.code(selected_session_data["ds_trace_desc"], language="yaml")
            if selected_session_data["current_code"] != "":
                st.subheader("当前 SOTA 代码", divider=True)
                st.code(
                    body=selected_session_data["current_code"],
                    language="python",
                )

            # 显示假设候选
            st.subheader("假设候选:", divider=True)
            hypothesis_candidates = selected_session_data["hypothesis_candidates"]
            tabs = st.tabs(
                [
                    f"{'✅' if i == selected_session_data['target_hypothesis_index'] or selected_session_data['target_hypothesis_index'] == -1 else ''}假设 {i+1}"
                    for i in range(len(hypothesis_candidates))
                ]
            )
            for index, hypothesis in enumerate(hypothesis_candidates):
                with tabs[index]:
                    st.code(str(hypothesis), language="yaml")
            st.text("✅ 表示被选为目标假设")

            st.title("待决策事项:")

            # 用户反馈表单
            with st.form(key="user_form"):
                st.caption("请修改以下字段并提交以提供您的反馈。")
                target_hypothesis = st.text_area(
                    "目标假设: (可以从候选中复制)",
                    value=(original_hypothesis := selected_session_data["target_hypothesis"].hypothesis),
                    height="content",
                )
                target_task = st.text_area(
                    "目标任务描述:",
                    value=(original_task_desc := selected_session_data["task"].description),
                    height="content",
                )
                # 显示和编辑过往的用户指令
                user_instruction_list = []
                if selected_session_data.get("former_user_instructions") is not None:
                    st.caption("过往的用户指令，您可以修改或删除内容以移除特定指令。")
                    for user_instruction in selected_session_data.get("former_user_instructions"):
                        user_instruction_list.append(
                            st.text_area("过往的用户指令", value=user_instruction, height="content")
                        )
                user_instruction_list.append(st.text_area("添加新的用户指令", value="", height="content"))

                submit = st.form_submit_button("提交")
                approve = st.form_submit_button("批准且不作更改")

                if submit or approve:
                    if approve:
                        # 如果用户直接批准
                        submit_dict = {"action": "confirm"}
                    else:
                        # 如果用户提交了修改
                        user_instruction_str_list = [ui for ui in user_instruction_list if ui.strip() != ""]
                        user_instruction_str_list = (
                            None if len(user_instruction_str_list) == 0 else user_instruction_str_list
                        )
                        action = (
                            "confirm"
                            if target_hypothesis == original_hypothesis
                            and target_task == original_task_desc
                            and user_instruction_str_list == selected_session_data.get("user_instruction")
                            else "rewrite"
                        )
                        submit_dict = {
                            "target_hypothesis": target_hypothesis,
                            "task_description": target_task,
                            "user_instruction": user_instruction_str_list,
                            "action": action,
                        }

                    # 将用户反馈写入 JSON 文件
                    json.dump(
                        submit_dict,
                        open(
                            DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}_RET.json", "w"
                        ),
                    )
                    # 删除原始会话文件
                    Path(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl").unlink(
                        missing_ok=True
                    )
                    st.success("您的反馈已提交。谢谢！")
                    time.sleep(5)
                    state.selected_session_name = None

            if st.button("延长60秒"):
                # 延长会话过期时间
                session_data = pickle.load(
                    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl", "rb")
                )
                session_data["expired_datetime"] = session_data["expired_datetime"] + timedelta(seconds=60)
                pickle.dump(
                    session_data,
                    open(DS_RD_SETTING.user_interaction_mid_folder / f"{state.selected_session_name}.pkl", "wb"),
                )
    else:
        st.warning("请从侧边栏选择一个会话。")


# 使用 @st.fragment 每秒更新一次会话
@st.fragment(run_every=1)
def update_sessions():
    """更新并显示活动会话。"""
    log_folder = Path(DS_RD_SETTING.user_interaction_mid_folder)
    state.sessions = {}
    # 遍历 pkl 文件，加载有效的会话
    for session_file in log_folder.glob("*.pkl"):
        try:
            session_data = pickle.load(open(session_file, "rb"))
            if session_data["expired_datetime"] > datetime.now():
                state.sessions[session_file.stem] = session_data
            else:
                # 删除过期的会话文件
                session_file.unlink(missing_ok=True)
                ret_file = log_folder / f"{session_file.stem}_RET.json"
                ret_file.unlink(missing_ok=True)
        except Exception as e:
            continue
    render_main_content()


@st.fragment(run_every=1)
def render_sidebar():
    """渲染侧边栏，显示活动会话列表。"""
    st.title("R&D-Agent 用户交互门户")
    if state.sessions:
        st.header("活动会话")
        st.caption("点击一个会话以查看:")
        session_names = [name for name in state.sessions]
        for session_name in session_names:
            with st.container(border=True):
                remaining = state.sessions[session_name]["expired_datetime"] - datetime.now()
                total_sec = int(remaining.total_seconds())
                label = f"剩余 {total_sec} 秒" if total_sec > 0 else "已过期"
                if st.button(f"会话 ID:{session_name[:4]}", key=f"session_btn_{session_name}"):
                    state.selected_session_name = session_name
                    state.data = state.sessions[session_name]
                st.markdown(f"⏳ {label}")
    else:
        st.warning("没有可用的活动会话。请等待。")


# 启动会话更新
update_sessions()
# 在侧边栏中渲染
with st.sidebar:
    render_sidebar()
