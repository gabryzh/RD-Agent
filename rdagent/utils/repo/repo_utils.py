import ast
import inspect
import os
from pathlib import Path
from typing import Dict, List, Union


class RepoAnalyzer:
    """
    一个用于分析 Python 代码仓库的类。
    它可以生成仓库结构的概览、文件内容的摘要等。
    """
    def __init__(self, repo_path: str):
        """
        初始化 RepoAnalyzer。

        :param repo_path: 要分析的代码仓库的根目录路径。
        """
        self.repo_path = Path(repo_path)
        self.summaries = {}  # 用于缓存摘要信息（当前未使用）

    def summarize_repo(self, verbose_level: int = 1, doc_str_level: int = 1, sign_level: int = 1) -> str:
        """
        为整个代码仓库工作区生成一个自然语言摘要。

        :param verbose_level: 摘要的详细程度 (0-2)。
                              0: 最精简 (仅文件名)
                              1: 默认 (文件信息、类名、函数名)
                              2+: 详细 (包含类内部的方法细节)
        :param doc_str_level: 文档字符串的详细程度 (0-2)。控制是否包含以及包含多少文档字符串信息。
        :param sign_level: 函数签名的详细程度 (0-2)。控制是否包含函数签名信息。
        :return: 一个包含工作区摘要的字符串。
        """
        file_summaries = []
        # 首先生成仓库的目录树结构图
        tree_structure = self._generate_tree_structure()

        # 遍历仓库目录下的所有 .py 文件
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    # 为每个文件生成摘要
                    file_summaries.append(self._summarize_file(file_path, verbose_level, doc_str_level, sign_level))

        # 组装最终的完整摘要报告
        total_files = len(file_summaries)
        workspace_summary = f"工作区摘要: {self.repo_path.name}\n"
        workspace_summary += f"{'=' * 40}\n\n"
        workspace_summary += "工作区结构:\n"
        workspace_summary += tree_structure
        workspace_summary += (
            f"\n此工作区包含 {total_files} 个 Python 文件。\n\n"
        )

        for i, summary in enumerate(file_summaries, 1):
            workspace_summary += f"文件 {i} / {total_files}:\n{summary}\n"

        workspace_summary += f"\n工作区摘要结束: {self.repo_path.name}"
        return workspace_summary

    def _generate_tree_structure(self) -> str:
        """
        生成仓库的树状目录结构图。
        """
        tree = []
        # 遍历目录，忽略非 .py 文件
        for root, dirs, files in os.walk(self.repo_path):
            # 计算当前目录的深度，用于生成缩进
            level = root.replace(str(self.repo_path), "").count(os.sep)
            indent = "│   " * (level - 1) + "├── " if level > 0 else ""
            # 添加目录名
            tree.append(f"{indent}{os.path.basename(root)}/")

            subindent = "│   " * level + "├── "
            # 添加该目录下的 .py 文件名
            for file in files:
                if file.endswith(".py"):
                    tree.append(f"{subindent}{file}")

        return "\n".join(tree)

    def _summarize_file(self, file_path: Path, verbose_level: int, doc_str_level: int, sign_level: int) -> str:
        """
        为单个 Python 文件生成摘要。
        """
        with open(file_path, "r") as f:
            content = f.read()

        # 使用 ast (抽象语法树) 解析文件内容
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return f"文件: {file_path.relative_to(self.repo_path)}\n{'-' * 40}\n无法解析文件: 语法错误 {e}\n"

        summary = f"文件: {file_path.relative_to(self.repo_path)}\n"
        summary += f"{'-' * 40}\n"

        # 提取文件顶层的类和函数定义
        classes = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)]

        if classes:
            summary += f"此文件包含 {len(classes)} 个类。\n"
        if functions:
            summary += f"此文件包含 {len(functions)} 个顶层函数。\n"

        # 分别为每个类和函数生成摘要
        for node in classes + functions:
            if isinstance(node, ast.ClassDef):
                summary += self._summarize_class(node, verbose_level, doc_str_level, sign_level)
            elif isinstance(node, ast.FunctionDef):
                summary += self._summarize_function(node, verbose_level, doc_str_level, sign_level)

        return summary

    def _summarize_class(self, node: ast.ClassDef, verbose_level: int, doc_str_level: int, sign_level: int) -> str:
        """
        为单个类定义生成摘要。
        """
        summary = f"\n类: {node.name}\n"
        # 提取文档字符串的第一句作为描述
        if doc_str_level > 0 and ast.get_docstring(node):
            summary += f"  描述: {ast.get_docstring(node).split('.')[0]}.\n"

        # 查找类中的方法
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if methods:
            summary += f"  此类有 {len(methods)} 个方法。\n"

        # 如果详细程度足够高，则递归地为每个方法生成摘要
        if verbose_level > 1:
            for method in methods:
                summary += self._summarize_function(method, verbose_level, doc_str_level, sign_level, indent="  ")
        return summary

    def _summarize_function(
        self, node: ast.FunctionDef, verbose_level: int, doc_str_level: int, sign_level: int, indent: str = ""
    ) -> str:
        """
        为单个函数或方法定义生成摘要。
        """
        summary = f"{indent}函数: {node.name}\n"
        # 如果需要，生成函数签名
        if sign_level > 0:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"{node.name}({', '.join(args)}){returns}"
            summary += f"{indent}  签名: {signature}\n"

        # 如果需要，提取文档字符串的第一句作为用途说明
        if doc_str_level > 0 and ast.get_docstring(node):
            doc = ast.get_docstring(node)
            summary += f"{indent}  用途: {doc.split('.')[0]}.\n"
        return summary

    def highlight(self, file_names: Union[str, List[str]]) -> Dict[str, str]:
        """
        从仓库中提取指定文件或文件列表的完整内容。

        :param file_names: 一个文件名或一个文件名列表。
        :return: 一个包含文件名和其对应内容的字典。
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        highlighted_content = {}
        for file_name in file_names:
            file_path = self.repo_path / file_name
            if file_path.exists() and file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    highlighted_content[file_name] = f.read()
            else:
                highlighted_content[file_name] = f"文件未找到: {file_name}"

        return highlighted_content


if __name__ == "__main__":
    # 这是一个当此脚本被直接执行时的示例
    # 注意： "features" 目录和其中的文件需要在本地存在才能成功运行
    try:
        analyzer = RepoAnalyzer(repo_path="features")
        summary = analyzer.summarize_repo(verbose_level=2, doc_str_level=2, sign_level=2)
        print(summary)
        highlighted_files = analyzer.highlight(
            file_names=["utils/repo/repo_utils.py", "components/benchmark/eval_method.py"]
        )
        print("\n高亮文件内容:")
        for file_name, content in highlighted_files.items():
            print(f"\n{file_name}\n{'=' * len(file_name)}\n{content}")
    except FileNotFoundError as e:
        print(f"示例代码运行失败，请确保 'features' 目录存在: {e}")
