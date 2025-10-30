"""
处理从 Python 文件到 Jupyter notebook 的转换。
"""

import argparse
from typing import Optional

import nbformat

from rdagent.components.coder.data_science.share.util import (
    extract_first_section_name_from_code,
    extract_function_body,
    split_code_and_output_into_sections,
)
from rdagent.core.experiment import Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import MarkdownAgentOut
from rdagent.utils.agent.tpl import T


class NotebookConverter:
    """
    负责为工作空间编写 Jupyter notebook 的构建器。
    """

    def validate_code_format(self, code: str) -> str | None:
        """
        如果代码格式有效，则返回 None，否则返回错误消息。
        """
        main_function_body = extract_function_body(code, "main")
        if not main_function_body:
            return "[错误] 代码中未找到 main 函数。请确保定义了 main 函数，并包含必要的使用 print 语句划分的部分。"

        found_section_name = extract_first_section_name_from_code(main_function_body)
        if not found_section_name:
            return "[错误] 代码中未找到任何部分。期望看到 'print(\"Section: <section name>\")' 作为部分分隔符。并确保它们实际被运行，而不仅仅是注释。"

        return None

    def convert(
        self,
        task: Optional[Task],
        code: str,
        stdout: str,
        outfile: Optional[str] = None,
        use_debug_flag: bool = False,
    ) -> str:
        """
        根据当前进度构建一个 notebook。
        """
        # 处理代码中的 argparse，以确保其在 notebook 环境中正常工作
        should_handle_argparse = "argparse" in code
        sections = split_code_and_output_into_sections(code=code, stdout=stdout)
        notebook = nbformat.v4.new_notebook()

        # 使用 LLM 为 notebook 生成一个介绍性单元格
        if task:
            system_prompt = T(".prompts:notebookconverter.system").r()
            user_prompt = T(".prompts:notebookconverter.user").r(
                plan=task.get_task_information(),
                code=code,
            )
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt
            )
            intro_content = MarkdownAgentOut.extract_output(resp)
            notebook.cells.append(nbformat.v4.new_markdown_cell(intro_content))

        if should_handle_argparse:
            # 删除多余的 `import sys`，因为在处理 argparse 时会添加它
            if "import sys\n" in sections[0]["code"]:
                sections[0]["code"] = sections[0]["code"].replace("import sys\n", "")

            # 添加 sys.argv 修改以处理 argparse
            sections[0]["code"] = (
                "\n".join(
                    [
                        "import sys",
                        "# hack 以允许 argparse 在 notebook 中工作",
                        ('sys.argv = ["main.py", "--debug"]' if use_debug_flag else 'sys.argv = ["main.py"]'),
                    ]
                )
                + "\n\n"
                + sections[0]["code"].lstrip()
            )

        for section in sections:
            # 为部分名称和注释创建一个 markdown 单元格
            markdown_content = ""
            if section["name"]:
                markdown_content += f"## {section['name']}\n"
            if section["comments"]:
                markdown_content += f"{section['comments']}\n"
            if markdown_content:
                notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_content))

            # 为部分代码和输出创建一个代码单元格
            if section["code"]:
                cell = nbformat.v4.new_code_cell(section["code"])
                if section["output"]:
                    # 为简单起见，将所有输出都视为来自 stdout
                    # TODO: 支持 Jupyter 内核执行并在此处适当处理输出
                    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=section["output"])]
                notebook.cells.append(cell)

        # 保存 notebook 或将其作为字符串返回
        if outfile:
            with open((outfile), "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
                logger.info(f"Notebook 已写入 {outfile}")

        return nbformat.writes(notebook)


if __name__ == "__main__":
    converter = NotebookConverter()
    parser = argparse.ArgumentParser(description="将 Python 代码转换为 Jupyter notebook。")
    parser.add_argument("inputfile", type=str, help="输入 Python 文件的路径。")
    parser.add_argument("outfile", type=str, help="输出 Notebook 文件的路径。")
    parser.add_argument(
        "--stdout",
        type=str,
        default="",
        help="代码执行的标准输出。",
    )
    parser.add_argument("--debug", action="store_true", help="使用调试标志来修改 sys.argv。")
    args = parser.parse_args()
    converter.convert(
        task=None,
        code=open(args.inputfile, "r").read(),
        stdout=args.stdout,
        outfile=args.outfile,
        use_debug_flag=False,
    )
