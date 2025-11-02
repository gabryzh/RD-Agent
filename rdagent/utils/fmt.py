"""
支持生成更好格式的工具。
"""


def shrink_text(
    text: str, context_lines: int = 200, line_len: int = 5000, *, row_shrink: bool = True, col_shrink: bool = True
) -> str:
    """
    当文本内容过长时，隐藏中间部分。
    可以按行数和单行字符数进行缩减。

    行缩减示例:
    >>> shrink_text("line1\\nline2\\nline3", context_lines=2, line_len=5)
    'line1\\n... (1 lines are hidden) ...\\nline3'

    禁用行缩减示例:
    >>> shrink_text("line1\\nline2\\nline3", context_lines=2, line_len=5, row_shrink=False)
    'line1\\nline2\\nline3'

    列（单行字符）缩减示例:
    >>> shrink_text("short line", context_lines=2, line_len=5)
    'sh... (5 chars are hidden) ...ine'

    长列缩减示例:
    >>> shrink_text("a" * 5010, context_lines=2, line_len=10)
    'aaaaa... (5000 chars are hidden) ...aaaaa'

    参数:
    ----------
    text : str
        输入的文本。
    context_lines : int
        希望保留的上下文总行数（头部和尾部）。
    line_len : int
        单行的最大字符长度，超过则会被缩减。
    row_shrink : bool
        是否启用行缩减。
    col_shrink : bool
        是否启用列（单行字符）缩减。

    返回:
    -------
    str
        缩减后的文本。
    """

    lines = text.splitlines()
    total_lines = len(lines)

    # 首先处理列（单行字符）的缩减
    new_lines = []
    for line in lines:
        if col_shrink and len(line) > line_len:
            # 如果单行长度超过 line_len，则截取头部和尾部，中间用省略号代替
            half_len = line_len // 2
            line = f"{line[:half_len]}... ({len(line) - line_len} chars are hidden) ...{line[-half_len:]}"
        new_lines.append(line)
    lines = new_lines

    # 如果不启用行缩减，或者总行数未超过限制，则直接返回
    if not row_shrink or total_lines <= context_lines:
        return "\n".join(lines)

    # 仅当启用行缩减且总行数大于 context_lines 时才进行行缩减
    # 计算从开头和结尾各显示多少行
    half_lines = context_lines // 2
    start = "\n".join(lines[:half_lines])
    end = "\n".join(lines[-half_lines:])

    # 计算我们隐藏了多少行
    hidden_lines = total_lines - (half_lines * 2)

    return f"{start}\n... ({hidden_lines} lines are hidden) ...\n{end}"
