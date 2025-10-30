"""
专注于为工作空间编写文档的开发者
"""

# 导入 rdagent 内部模块
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import MarkdownAgentOut
from rdagent.utils.agent.tpl import T


class DocDev(Developer[Experiment]):
    """
    该开发者负责为工作空间编写文档。
    """

    def develop(self, exp: Experiment) -> None:
        """
        为工作空间编写文档。

        此方法会分析工作空间中的文件，特别是关键文件（如 main.py, scores.csv），
        然后利用大语言模型生成一个README.md文件并注入到工作空间中。
        """
        ws: FBWorkspace = exp.experiment_workspace

        # 获取工作空间中所有文件的相对路径列表
        file_li = [str(file.relative_to(ws.workspace_path)) for file in ws.workspace_path.rglob("*") if file.is_file()]

        # 定义需要重点关注的关键文件列表
        key_file_list = ["main.py", "scores.csv"]

        # 构建发送给大语言模型的提示
        system_prompt = T(".prompts:docdev.system").r()
        user_prompt = T(".prompts:docdev.user").r(
            file_li=file_li,
            # 读取关键文件的内容
            key_files={f: (ws.workspace_path / f).read_text() for f in key_file_list},
        )

        # 调用大语言模型生成文档内容
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        # 从模型的响应中提取 Markdown 格式的输出
        markdown = MarkdownAgentOut.extract_output(resp)
        # 将生成的 Markdown 内容作为 README.md 文件注入到工作空间中
        ws.inject_files(**{"README.md": markdown})
