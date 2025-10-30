import re
from pathlib import Path

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.core.developer import Developer
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

# 获取当前文件所在的目录路径
DIRNAME = Path(__file__).absolute().resolve().parent


class ModelCodeWriter(Developer[ModelExperiment]):
    """
    模型代码编写器，采用“一次生成”（One-shot）策略。

    该类负责接收一个模型实验（包含多个任务），并为每个任务调用 LLM 生成代码，
    然后将代码注入到相应的工作空间中。
    """
    def develop(self, exp: ModelExperiment) -> ModelExperiment:
        """
        为实验中的每个任务生成代码。

        Args:
            exp (ModelExperiment): 包含一个或多个模型任务的实验对象。

        Returns:
            ModelExperiment: 更新了工作空间列表（包含生成的代码）的实验对象。
        """
        mti_l = []  # 用于存储每个任务的工作空间
        for t in exp.sub_tasks:
            # 1. 为每个任务创建并准备一个工作空间
            mti = ModelFBWorkspace(t)
            mti.prepare()

            # 2. 构建 user 和 system prompt
            user_prompt = T(".prompts:code_implement_user").r(
                name=t.name,
                description=t.description,
                formulation=t.formulation,
                variables=t.variables,
            )
            system_prompt = T(".prompts:code_implement_sys").r()

            # 3. 调用 LLM API 获取响应
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt)

            # 4. 从响应中提取 Python 代码块
            #    - 使用正则表达式匹配 ```python ... ``` 格式的代码
            match = re.search(r".*```[Pp]ython\n(.*)\n```.*", resp, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                # 如果没有匹配到代码块，则将整个响应作为代码（一种备用策略）
                code = resp

            # 5. 将提取的代码注入到工作空间的 model.py 文件中
            mti.inject_files(**{"model.py": code})
            mti_l.append(mti)

        # 6. 更新实验对象的工作空间列表并返回
        exp.sub_workspace_list = mti_l
        return exp
