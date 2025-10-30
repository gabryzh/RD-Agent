# %%
import json
from pathlib import Path

import streamlit as st

from rdagent.log.ui.conf import UI_SETTING
from rdagent.utils.repo.diff import generate_diff_from_dict

# 获取 AIDE 日志路径
aide_path = UI_SETTING.aide_path
if not Path(aide_path).exists():
    st.error(f"路径 {aide_path} 不存在，请通过 `UI_AIDE_PATH` 设置")
    st.stop()

# 查找所有 filtered_journal.json 文件
jps = [str(i) for i in Path(aide_path).rglob("**/filtered_journal.json")]
jps = sorted(jps)

# 创建两列布局
left, right = st.columns([1, 4])
with left:
    # 设置默认选中的日志文件
    default = 0
    # 尝试从 URL 查询参数中获取日志路径
    ppp = f"{aide_path}/{st.query_params.get('jnp')}/logs/filtered_journal.json"
    if ppp in jps:
        default = jps.index(ppp)

    # 创建一个单选按钮让用户选择日志文件
    jnp = st.radio("选择日志", options=jps, index=default, format_func=lambda x: str(x).split("/")[-3])
    jnp = Path(jnp)

# 读取选中的 JSON 文件
with jnp.open("r") as f:
    d = json.load(f)

# 创建节点ID到节点数据的映射
nm = {nd["id"]: nd for nd in d["nodes"]}

# %%
with right:
    # 显示 AIDE 追踪信息
    st.header("AIDE 追踪", divider="rainbow")
    st.subheader(jnp)
    # 遍历父子关系
    for c, p in d["node2parent"].items():
        f = nm[p] # 父节点
        t = nm[c] # 子节点
        # 生成父子节点代码之间的差异
        df_lines = generate_diff_from_dict({"aide.py": f["code"]}, {"aide.py": t["code"]})

        # 使用可展开的容器显示每个节点的详细信息
        with st.expander(f"节点 {p} -> {c}"):
            st.markdown(f"## 父节点 ({f['metric']['value']}) 分析")
            st.code(f["analysis"], wrap_lines=True)
            st.markdown(f"## 子节点 ({t['metric']['value']}) 计划")
            st.code(t["plan"], wrap_lines=True)
            st.markdown("## 差异")
            st.code("".join(df_lines), language="diff", wrap_lines=True)

# %%
