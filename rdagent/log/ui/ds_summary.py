"""
关于指标的更详细文档，请参考 rdagent/log/ui/utils.py:get_summary_df
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state

from rdagent.log.ui.utils import (
    ALL,
    HIGH,
    LITE,
    MEDIUM,
    curve_figure,
    get_statistics_df,
    get_summary_df,
    lite_curve_figure,
    percent_df,
)
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction


def curves_win(summary: dict):
    """
    显示指标曲线的窗口。
    """
    cbwin1, cbwin2 = st.columns(2)
    # 切换显示详细曲线
    if cbwin1.toggle("显示曲线", key="show_curves"):
        for k, v in summary.items():
            with st.container(border=True):
                st.markdown(f"**:blue[{k}] - :violet[{v['competition']}]**")
                try:
                    # ... (处理和绘制详细的验证/测试分数曲线) ...
                except Exception as e:
                    import traceback
                    st.markdown("- 错误: " + str(e))
                    st.code(traceback.format_exc())
                    st.markdown("- 验证分数: ")
                    st.json(v["valid_scores"])
    # 切换显示简版曲线
    if cbwin2.toggle("显示曲线 (简版)", key="show_curves_lite"):
        st.pyplot(lite_curve_figure(summary))


def all_summarize_win():
    """
    显示所有日志文件夹摘要的主窗口。
    """
    def shorten_folder_name(folder: str) -> str:
        """缩短文件夹名称以便显示。"""
        if "amlt" in folder:
            return folder[folder.rfind("amlt") + 5 :].split("/")[0]
        if "ep" in folder:
            return folder[folder.rfind("ep") :]
        return folder

    # 多选框选择要显示的文件夹
    selected_folders = st.multitext("显示这些文件夹", state.log_folders, state.log_folders, format_func=shorten_folder_name)

    # 检查摘要文件是否存在
    for lf in selected_folders:
        if not (Path(lf) / "summary.pkl").exists():
            st.warning(f"在 **{lf}** 中未找到 summary.pkl\n\n运行:`dotenv run -- python rdagent/log/mle_summary.py grade_summary --log_folder={lf} --hours=<>`")

    summary = {}
    dfs = []
    # 加载并合并所选文件夹的摘要数据
    for lf in selected_folders:
        s, df = get_summary_df(lf)
        df.index = [f"{shorten_folder_name(lf)} - {idx}" for idx in df.index]
        dfs.append(df)
        summary.update({f"{shorten_folder_name(lf)} - {k}": v for k, v in s.items()})
    base_df = pd.concat(dfs)

    # ... (计算和显示统计指标) ...

    # 根据比赛级别筛选
    select_lite_level = st.selectbox("选择 MLE-Bench 比赛级别", options=["ALL", "HIGH", "MEDIUM", "LITE"], index=0, key="select_lite_level")
    # ... (根据选择的级别更新 DataFrame 的 "Select" 列) ...

    # 切换只显示最佳结果
    if st.toggle("选择最佳", key="select_best"):
        # ... (根据比赛指标方向，分组找出最佳结果并更新 "Select" 列) ...
        pass

    # 使用 st.data_editor 显示和编辑 DataFrame
    base_df = st.data_editor(base_df, column_config={"Select": st.column_config.CheckboxColumn("选择", help="统计此追踪。", disabled=False)}, disabled=(col for col in base_df.columns if col not in ["Select"]))
    st.markdown("我们 vs 基础: `math.exp(abs(math.log(sota_exp_score / baseline_score)))`")

    # 统计选择的比赛
    base_df = base_df[base_df["Select"]]
    st.markdown(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    stat_win_left, stat_win_right = st.columns(2)
    with stat_win_left:
        # 显示统计数据
        stat_df = get_statistics_df(base_df)
        st.dataframe(stat_df.round(2))
        # ... (显示 Markdown 格式的表格) ...
    with stat_win_right:
        # 显示总循环次数的分布直方图
        Loop_counts = base_df["Total Loops"]
        fig = px.histogram(Loop_counts, nbins=15, title="总循环次数分布", color_discrete_sequence=["#3498db"])
        # ... (添加均值、中位数线和注释) ...
        st.plotly_chart(fig, use_container_width=True)

    # 显示曲线
    st.subheader("曲线", divider="rainbow")
    curves_win(summary)


# 主容器
with st.container(border=True):
    try:
        all_summarize_win()
    except Exception as e:
        import traceback
        st.error(f"显示摘要时发生错误:\n{e}")
        st.code(traceback.format_exc())
