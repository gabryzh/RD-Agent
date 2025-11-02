import difflib
import fnmatch
from pathlib import Path


def generate_diff(dir1: str, dir2: str, file_pattern: str = "*.py") -> list[str]:
    """
    比较两个目录中符合指定文件模式的文件，并生成它们之间的差异（diff）。
    这个函数的行为类似于 Linux 命令 `diff -durN dir1 dir2`。

    Args:
        dir1 (str): 第一个目录的路径。
        dir2 (str): 第二个目录的路径。
        file_pattern (str, optional): 用于过滤文件的 glob 模式。默认为 "*.py"。

    Returns:
        list[str]: 一个包含两个目录间不同文件的 diff 信息的列表，
                   每项是一个字符串行。
    """

    # 递归查找两个目录中所有符合模式的文件，并获取它们的相对路径集合
    dir1_files = {f.relative_to(dir1) for f in Path(dir1).rglob(file_pattern) if f.is_file()}
    dir2_files = {f.relative_to(dir2) for f in Path(dir2).rglob(file_pattern) if f.is_file()}

    # 合并两个文件集合，以确保处理所有在一个或两个目录中存在的文件
    all_files = dir1_files.union(dir2_files)
    file_dict1 = {}
    file_dict2 = {}

    # 读取所有文件的内容到字典中，以便后续比较
    for file in all_files:
        file1 = Path(dir1) / file
        file2 = Path(dir2) / file
        if file1.exists():
            with file1.open() as f1:
                file_dict1[str(file)] = f1.read()
        else:
            # 如果文件在 dir1 中不存在，视为空内容
            file_dict1[str(file)] = ""
        if file2.exists():
            with file2.open() as f2:
                file_dict2[str(file)] = f2.read()
        else:
            # 如果文件在 dir2 中不存在，视为空内容
            file_dict2[str(file)] = ""

    # 调用 generate_diff_from_dict 函数来实际生成 diff
    # 这里的 file_pattern="*" 是因为已经在上面按需过滤了文件
    return generate_diff_from_dict(file_dict1, file_dict2, file_pattern="*")


def generate_diff_from_dict(file_dict1: dict, file_dict2: dict, file_pattern: str = "*.py") -> list[str]:
    """
    比较两个文件内容字典之间的差异。
    字典的格式应为 {文件路径: 文件内容}。

    Args:
        file_dict1 (dict): 第一个文件字典。
        file_dict2 (dict): 第二个文件字典。
        file_pattern (str, optional): 用于过滤文件的 glob 模式。默认为 "*.py"。

    Returns:
        List[str]: 一个包含两个字典间不同文件的 diff 信息的列表。
    """
    diff_files = []
    # 合并两个字典的所有键，以获取需要比较的完整文件列表
    all_files = set(file_dict1.keys()).union(set(file_dict2.keys()))

    # 遍历排序后的文件列表，以确保输出顺序一致
    for file in sorted(all_files):
        # 如果文件名不匹配指定的模式，则跳过
        if not fnmatch.fnmatch(file, file_pattern):
            continue

        # 获取文件在两个字典中的内容，如果不存在则视为空字符串
        content1 = file_dict1.get(file, "")
        content2 = file_dict2.get(file, "")

        # 使用 difflib.unified_diff 生成统一格式的 diff
        diff = list(
            difflib.unified_diff(
                content1.splitlines(keepends=True),  # 按行分割并保留换行符
                content2.splitlines(keepends=True),
                # 设置 diff 头部的文件名信息
                fromfile=file if file in file_dict1 else file + " (empty file)",
                tofile=file if file in file_dict2 else file + " (empty file)",
            )
        )

        # 如果生成了 diff 内容（即文件有差异），则将其添加到结果列表中
        if diff:
            diff_files.extend(diff)

    return diff_files
