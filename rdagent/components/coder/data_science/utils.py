import re


def remove_eda_part(stdout: str) -> str:
    """
    数据科学场景具有基于LLM的EDA（探索性数据分析）功能。
    当当前任务不涉及EDA时，我们可以移除这部分输出。

    参数:
        stdout (str): 包含EDA部分的原始标准输出字符串。

    返回:
        str: 移除了EDA部分的标准输出字符串。
    """
    return re.sub(r"=== Start of EDA part ===(.*)=== End of EDA part ===", "", stdout, flags=re.DOTALL)
