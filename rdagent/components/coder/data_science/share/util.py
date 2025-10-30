# 导入必要的库和模块
import ast
import io
import re
import tokenize
from itertools import zip_longest
from typing import List, Optional, Set, Tuple, TypedDict


class CodeSection(TypedDict):
    """
    表示原始 Python 源代码的一个部分，将被转换为 notebook 单元格。
    """
    name: Optional[str]
    code: Optional[str]
    comments: Optional[str]
    output: Optional[str]


def extract_function_body(source_code: str, function_name: str) -> Optional[str]:
    """
    从源代码中提取函数体。
    如果未找到函数，则返回 None。

    假设：该函数是多行的，并且在顶层定义。
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = source_code.splitlines()
            start = node.body[0].lineno
            end = node.body[-1].end_lineno
            body_lines = lines[start - 1 : end]
            indent_level = len(body_lines[0]) - len(body_lines[0].lstrip())
            return "\n".join(line[indent_level:] for line in body_lines)
    return None


def split_sections(
    text: str, section_header_regex: str, known_sections: Optional[list[str]] = None
) -> tuple[Optional[str], list[str], list[str]]:
    """
    根据部分标题将文本拆分为多个部分。
    """
    sections = []
    section_names = []
    current_section = []
    next_section_name_index = 0
    for line in text.splitlines():
        match = re.match(section_header_regex, line)
        extracted_section_name = match.group(1).strip() if match else None
        if extracted_section_name and (
            not known_sections
            or (
                next_section_name_index < len(known_sections)
                and extracted_section_name == known_sections[next_section_name_index]
            )
        ):
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
            current_section.append(line)
            section_names.append(extracted_section_name)
            next_section_name_index += 1
        else:
            current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))

    # 如果第一部分与标题正则表达式不匹配，则将其视为主标题部分。
    header_section = None
    if sections and not re.search(section_header_regex, sections[0]):
        header_section = sections[0]
        sections = sections[1:]

    return header_section, sections, section_names


def split_code_sections(source_code: str) -> tuple[Optional[str], list[str]]:
    """
    根据部分标题将代码拆分为多个部分。
    """
    return split_sections(source_code, r'^print\(["\']Section: (.+)["\']\)')


def split_output_sections(stdout: str, known_sections: list[str]) -> tuple[Optional[str], list[str]]:
    """
    根据部分标题将输出拆分为多个部分。
    """
    header_section, sections, _ = split_sections(stdout, r"^Section: (.+)", known_sections=known_sections)
    return header_section, sections


def extract_comment_under_first_print(source_code) -> tuple[Optional[str], str]:
    """
    从第一个 print 语句后的源代码中提取注释。
    """
    lines = source_code.splitlines()
    lines_to_remove = set()
    all_comments = []

    parsed = ast.parse(source_code)
    # 仅查找第一个 print 语句
    first_print_lineno = None
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if getattr(node.value.func, "id", None) == "print":
                first_print_lineno = node.lineno
                break

    if first_print_lineno is None:
        # 未找到 print 语句，返回空注释和原始代码
        return None, source_code

    for i in range(first_print_lineno, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            comment_text = stripped.lstrip("# ").strip()
            all_comments.append(comment_text)
            lines_to_remove.add(i)
        elif stripped == "":
            continue
        elif i > first_print_lineno:
            break  # 遇到实际代码行后停止

    cleaned_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]
    cleaned_code = "\n".join(cleaned_lines)
    comments_str = "\n".join(all_comments) if all_comments else None

    return comments_str, cleaned_code


def extract_first_section_name_from_code(source_code):
    """
    从源代码中提取第一个部分名称。
    """
    parsed = ast.parse(source_code)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if getattr(call.func, "id", None) == "print" and call.args:
                arg0 = call.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    # 匹配 "Section: ..." 模式
                    m = re.match(r"Section:\s*(.+)", arg0.value)
                    if m:
                        return m.group(1).strip()
    return None


def extract_first_section_name_from_output(stdout: str) -> Optional[str]:
    """
    从输出字符串中提取第一个部分名称。
    """
    match = re.search(r"Section:\s*(.+)", stdout)
    if match:
        return match.group(1).strip()
    return None


def is_function_called(source_code: str, func_name: str) -> bool:
    """
    如果名为 `func_name` 的函数在 `source_code` 中被调用，则返回 True。
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 对于简单的函数调用，如 func()
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                return True

            # 对于调用，如 module.func()
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                return True
    return False


def remove_function(source_code: str, function_name: str) -> str:
    """
    从源代码中删除函数定义。
    """
    tree = ast.parse(source_code)
    lines = source_code.splitlines()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start_lineno = node.lineno - 1
            end_lineno = node.end_lineno
            return "\n".join(lines[:start_lineno] + lines[end_lineno:])

    return source_code


def remove_main_block(source_code: str) -> str:
    """
    从源代码中删除 if __name__ == "__main__": 块。
    """
    tree = ast.parse(source_code)
    lines = source_code.splitlines()

    # 找到主块并记下其行号
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):

                # 删除与此块对应的行
                start_lineno = node.lineno - 1
                end_lineno = node.end_lineno
                return "\n".join(lines[:start_lineno] + lines[end_lineno:])

    return source_code


def extract_top_level_functions_with_decorators_and_comments(
    code: str,
) -> List[Tuple[str, str]]:
    """
    返回顶层函数（不包括 "main"）的 (function_name, source_segment) 列表，
    包括装饰器和连续的前置注释。
    """
    # 解析 AST 以获取函数节点
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)

    # 预计算哪些行号有注释标记
    comment_lines: Set[int] = set()
    lines = code.splitlines(keepends=True)  # 保留确切的行内容以进行前缀检查

    tokgen = tokenize.generate_tokens(io.StringIO(code).readline)  # 生成 (type, string, start, end, line)
    for tok_type, _, (srow, scol), _, _ in tokgen:
        if tok_type == tokenize.COMMENT:
            # 该行注释之前的所有内容必须是空白
            prefix = lines[srow - 1][:scol]
            if prefix.strip() == "":
                comment_lines.add(srow)

    functions = []

    for node in tree.body:  # 仅顶层
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "main":
            continue

        # 确定起始行：如果存在，则为最早的装饰器，否则为 def/async 行
        if node.decorator_list:
            start_lineno = min(d.lineno for d in node.decorator_list)
        else:
            start_lineno = node.lineno

        # 向上扩展以包括连续的注释行（没有 intervening non-blank/non-comment）
        span_start = start_lineno
        curr = span_start - 1  # 检查上一行；行是 1-based 的
        while curr > 0:
            line_text = lines[curr - 1]
            if curr in comment_lines:
                span_start = curr
                curr -= 1
                continue
            if line_text.strip() == "":
                # 空行：包括它并继续向上扫描
                span_start = curr
                curr -= 1
                continue
            break  # 遇到代码或其他东西；停止

        # 确定函数定义的结束行，包括其主体
        # 如果可用，首选 end_lineno（Python 3.8+）
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            span_end = node.end_lineno
        else:
            # 后备：从主体中最深层子节点获取最后一行号
            def _max_lineno(n):
                max_ln = getattr(n, "lineno", 0)
                for child in ast.iter_child_nodes(n):
                    ln = _max_lineno(child)
                    if ln > max_ln:
                        max_ln = ln
                return max_ln

            span_end = _max_lineno(node)

        # 对原始源代码行进行切片
        segment = "".join(lines[span_start - 1 : span_end])
        functions.append((node.name, segment))

    return functions


def split_code_and_output_into_sections(code: str, stdout: str) -> list[CodeSection]:
    """
    将 Python 脚本及其输出转换为 CodeSection 列表。
    前提条件：main() 函数中的代码包含指示部分名称的 print 语句，例如 `print("Section: <section name>")`。
    """
    # 这将保存所有顶层代码，并且默认情况下所有函数定义。
    # 如果需要，稍后将函数移动到更相关的部分。
    # 第一步是删除 if __name__ == "__main__": 块和 main 函数
    top_level_code = remove_main_block(remove_function(code, "main"))

    main_function_body = extract_function_body(code, "main")
    functions = extract_top_level_functions_with_decorators_and_comments(top_level_code)

    # 根据 print("Section: <section name>") 代码将主函数体拆分为多个部分
    main_fn_top_level_section, main_fn_sections, known_section_names = (
        split_code_sections(main_function_body) if main_function_body else (None, [], [])
    )

    # 根据 "Section: " 标题将输出拆分为多个部分
    output_top_level_section, output_sections = split_output_sections(stdout, known_section_names)

    # 将代码和输出合并到代码部分
    result_sections: list[CodeSection] = []
    for output_section, code_section in zip_longest(output_sections, main_fn_sections):
        name = None
        if code_section is not None:
            # 如果代码部分可用，则从中提取部分名称
            name = extract_first_section_name_from_code(code_section)
        elif output_section:
            # 如果只有输出部分可用，则从中提取部分名称
            name = extract_first_section_name_from_output(output_section)
        comments, cleaned_code = (
            extract_comment_under_first_print(code_section) if code_section is not None else (None, None)
        )
        # 去除单元格的空白
        if cleaned_code is not None:
            cleaned_code = cleaned_code.strip()
        result_sections.append(CodeSection(name=name, code=cleaned_code, comments=comments, output=output_section))

    # 小优化：将函数定义移动到它们首次被调用的部分
    # TODO: 这不处理嵌套的函数引用，例如，fn A 调用 fn B，后者调用 fn C
    # 目前不会将 C 移动到调用 A 的部分
    for name, segment in functions:
        for section in result_sections:
            if section["code"] and is_function_called(section["code"], name):
                section["code"] = segment.strip() + "\n\n" + section["code"].lstrip()
                top_level_code = top_level_code.replace(segment, "")
                break

    # 在部分的开头注入顶层代码
    top_level_code = (
        top_level_code.rstrip() + "\n\n" + main_fn_top_level_section.lstrip()
        if main_fn_top_level_section
        else top_level_code
    )
    result_sections.insert(
        0,
        CodeSection(
            name=None,
            code=top_level_code,
            comments=None,
            output=output_top_level_section,
        ),
    )

    return result_sections
