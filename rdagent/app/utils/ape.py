"""
这是APE（自动提示工程）的初步版本。
"""

import pickle
from pathlib import Path

from rdagent.log.conf import LOG_SETTINGS


def get_llm_qa(file_path):
    """
    从pickle文件中加载并过滤LLM的问答数据。
    """
    data_flt = []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"加载了 {len(data)} 条数据")
        for item in data:
            if "debug_llm" in item["tag"]:
                data_flt.append(item)
    return data_flt


# 示例用法
# 从日志中加载LLM的问答数据
file_path = Path(LOG_SETTINGS.trace_path) / "debug_llm.pkl"
llm_qa = get_llm_qa(file_path)
print(f"过滤后剩下 {len(llm_qa)} 条数据")

if llm_qa:
    print("第一条数据:", llm_qa[0])

# 初始化APE后端
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

api = APIBackend()

# 分析测试数据并生成改进的提示
for qa in llm_qa:
    # 为APE生成系统提示
    system_prompt = T(".prompts:ape.system").r()

    # 根据LLM的问答上下文生成用户提示
    user_prompt = T(".prompts:ape.user").r(
        system=qa["obj"].get("system", ""), user=qa["obj"]["user"], answer=qa["obj"]["resp"]
    )

    # 调用API进行分析
    analysis_result = api.build_messages_and_create_chat_completion(
        system_prompt=system_prompt, user_prompt=user_prompt
    )

    # 打印分隔符并等待用户输入
    print(f"█" * 60)
    yes = input("是否继续？ (y/n)")
    if yes.lower() != 'y':
        break
