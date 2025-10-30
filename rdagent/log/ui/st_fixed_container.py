from typing import Literal

import streamlit as st
from streamlit.components.v1 import html

# 定义固定容器的 CSS 样式
FIXED_CONTAINER_CSS = """
:root {{
    --background-color: #ffffff; /* 默认背景颜色 */
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.fixed-container-{id}):not(:has(div.not-fixed-container)) {{
    position: {mode};
    width: inherit;
    background-color: inherit;
    {position}: {margin};
    z-index: 999;
}}
/* ... (其他 CSS 规则) ... */
""".strip()

# 定义用于更新背景颜色的 JavaScript
FIXED_CONTAINER_JS = """
const root = parent.document.querySelector('.stApp');
let lastBackgroundColor = null;
function updateContainerBackground(currentBackground) {
    parent.document.documentElement.style.setProperty('--background-color', currentBackground);
}
function checkForBackgroundColorChange() {
    // ... (检查并更新背景颜色的逻辑) ...
}
const observerCallback = (mutationsList, observer) => {
    // ... (MutationObserver 的回调) ...
};
const main = () => {
    checkForBackgroundColorChange();
    const observer = new MutationObserver(observerCallback);
    observer.observe(root, { attributes: true, childList: false, subtree: false });
}
document.addEventListener("DOMContentLoaded", main);
""".strip()

# 默认边距
MARGINS = {
    "top": "2.875rem",
    "bottom": "0",
}

# 计数器，用于生成唯一的容器ID
counter = 0


def st_fixed_container(
    *,
    height: int | None = None,
    border: bool | None = None,
    mode: Literal["fixed", "sticky"] = "fixed",
    position: Literal["top", "bottom"] = "top",
    margin: str | None = None,
    transparent: bool = False,
):
    """
    创建一个固定或粘性的容器。

    参数:
        height (int | None): 容器的高度。
        border (bool | None): 是否显示边框。
        mode (Literal["fixed", "sticky"]): 容器的定位模式。
        position (Literal["top", "bottom"]): 容器的位置。
        margin (str | None): 容器的边距。
        transparent (bool): 是否为透明背景。

    返回:
        一个可以放置内容的 Streamlit 容器。
    """
    if margin is None:
        margin = MARGINS[position]
    global counter

    fixed_container = st.container()
    non_fixed_container = st.container()
    css = FIXED_CONTAINER_CSS.format(
        mode=mode,
        position=position,
        margin=margin,
        id=counter,
    )
    with fixed_container:
        # 注入 CSS 和 JS
        html(f"<script>{FIXED_CONTAINER_JS}</script>", scrolling=False, height=0)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='fixed-container-{counter}'></div>",
            unsafe_allow_html=True,
        )
    with non_fixed_container:
        st.markdown(
            f"<div class='not-fixed-container'></div>",
            unsafe_allow_html=True,
        )
    counter += 1

    parent_container = fixed_container if transparent else fixed_container.container()
    return parent_container.container(height=height, border=border)


if __name__ == "__main__":
    # 示例用法
    for i in range(30):
        st.write(f"Line {i}")

    with st_fixed_container(mode="fixed", position="bottom", border=True):
        st.write("这是一个固定容器。")
        st.write("这是一个固定容器。")
        st.write("这是一个固定容器。")

    st.container(border=True).write("这是一个普通容器。")
    for i in range(30):
        st.write(f"Line {i}")
