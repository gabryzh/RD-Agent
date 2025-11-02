#!/usr/bin/env python3
# 以下代码修改自 https://cookbook.openai.com/examples/gpt4-1_prompting_guide

"""
一个独立的、纯 Python 3.9+ 的工具，用于将人类可读的
“伪 diff” 补丁文件应用到一组文本文件中。
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# --------------------------------------------------------------------------- #
#  领域对象 (Domain objects)
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    """定义了文件变更的三种类型：添加、删除、更新。"""
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    """表示单个文件的变更。"""
    type: ActionType  # 变更类型
    old_content: str | None = None  # 旧文件内容（用于 DELETE 和 UPDATE）
    new_content: str | None = None  # 新文件内容（用于 ADD 和 UPDATE）
    move_path: str | None = None  # 文件移动的目标路径（用于 UPDATE）


@dataclass
class Commit:
    """表示一次提交，包含多个文件的变更。"""
    changes: dict[str, FileChange] = field(default_factory=dict)  # 文件路径到文件变更的映射


# --------------------------------------------------------------------------- #
#  异常 (Exceptions)
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """在解析或应用补丁时检测到的任何问题。"""


# --------------------------------------------------------------------------- #
#  解析补丁时使用的辅助数据类
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """表示文件中的一个变更块（hunk）。"""
    orig_index: int = -1  # 在原始文件中的起始行号
    del_lines: list[str] = field(default_factory=list)  # 被删除的行
    ins_lines: list[str] = field(default_factory=list)  # 被插入的行


@dataclass
class PatchAction:
    """表示对单个文件的补丁操作。"""
    type: ActionType  # 操作类型
    new_file: str | None = None  # 新文件的完整内容（仅用于 ADD）
    chunks: list[Chunk] = field(default_factory=list)  # 变更块列表（仅用于 UPDATE）
    move_path: str | None = None  # 文件移动的目标路径（仅用于 UPDATE）


@dataclass
class Patch:
    """表示一个完整的补丁，包含对多个文件的操作。"""
    actions: dict[str, PatchAction] = field(default_factory=dict)  # 文件路径到补丁操作的映射


# --------------------------------------------------------------------------- #
#  补丁文本解析器 (Patch text parser)
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    """负责解析补丁文本并生成 Patch 对象。"""
    current_files: dict[str, str]  # 当前文件系统中的文件内容
    lines: list[str]  # 补丁文本的行列表
    index: int = 0  # 当前解析到的行号
    patch: Patch = field(default_factory=Patch)  # 解析结果
    fuzz: int = 0  # 模糊匹配的程度计数
    prefix: Path | None = None  # 应用于所有文件路径的前缀

    # ------------- 底层辅助方法 -------------------------------------- #
    def _cur_line(self) -> str:
        """获取当前行，如果超出范围则抛出异常。"""
        if self.index >= len(self.lines):
            raise DiffError("解析补丁时意外遇到输入结尾")
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """规范化行，去除行尾的 CR（回车符），以便在 LF 和 CRLF 换行符下都能正常比较。"""
        return line.rstrip("\r")

    # ------------- 扫描便利方法 ----------------------------------- #
    def is_done(self, prefixes: tuple[str, ...] | None = None) -> bool:
        """检查是否已解析完毕，或者是否遇到了指定的停止前缀。"""
        if self.index >= len(self.lines):
            return True
        if prefixes and len(prefixes) > 0 and self._norm(self._cur_line()).startswith(prefixes):
            return True
        return False

    def startswith(self, prefix: str | tuple[str, ...]) -> bool:
        """检查当前行是否以指定的前缀开始。"""
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        如果当前行以前缀开始，则消费该行并返回前缀之后的部分。
        如果前缀为空则会引发异常。
        """
        if prefix == "":
            raise ValueError("read_str() 需要一个非空的前缀")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """返回当前原始行并前进到下一行。"""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- 公共入口点 -------------------------------------- #
    def parse(self) -> None:
        """解析整个补丁文本。"""
        while not self.is_done(("*** End Patch",)):
            # ---------- 更新文件 (UPDATE) ---------- #
            path = self.read_str("*** Update File: ")
            if self.prefix:
                path = str(self.prefix / path)
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"重复的文件更新: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"更新文件错误 - 找不到文件: {path}")
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- 删除文件 (DELETE) ---------- #
            path = self.read_str("*** Delete File: ")
            if self.prefix:
                path = str(self.prefix / path)
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"重复的文件删除: {path}")
                if path not in self.current_files:
                    raise DiffError(f"删除文件错误 - 找不到文件: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- 添加文件 (ADD) ---------- #
            path = self.read_str("*** Add File: ")
            if self.prefix:
                path = str(self.prefix / path)
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"重复的文件添加: {path}")
                if path in self.current_files:
                    raise DiffError(f"添加文件错误 - 文件已存在: {path}")
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise DiffError(f"解析时遇到未知行: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise DiffError("缺少结束标记 *** End Patch")
        self.index += 1  # 消费结束标记

    # ------------- 片段解析器 ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        """解析文件更新部分的具体内容。"""
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            ),
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise DiffError(f"更新区域的无效行:\n{self._cur_line()}")

            # 模糊查找上下文行
            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [s.strip() for s in lines[:index]]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            # 预读下一个变更片段
            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)
            new_index, fuzz = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise DiffError(
                    f"在 {index} 处找到无效的 {'EOF ' if eof else ''}上下文:\n{ctx_txt}",
                )
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_ctx)
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        """解析添加文件部分的具体内容。"""
        lines: list[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:"),
        ):
            s = self.read_line()
            if not s.startswith("+"):
                raise DiffError(f"添加文件部分的无效行 (缺少 '+'): {s}")
            lines.append(s[1:])  # 去掉前导 '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


# --------------------------------------------------------------------------- #
#  辅助函数
# --------------------------------------------------------------------------- #
def find_context_core(
    lines: list[str],
    context: list[str],
    start: int,
) -> tuple[int, int]:
    """核心的上下文查找逻辑，支持不同程度的模糊匹配。"""
    if not context:
        return start, 0

    # 精确匹配
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0
    # 忽略行尾空白的匹配
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [s.rstrip() for s in context]:
            return i, 1
    # 忽略两端空白的匹配
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [s.strip() for s in context]:
            return i, 100
    return -1, 0


def find_context(
    lines: list[str],
    context: list[str],
    start: int,
    eof: bool,
) -> tuple[int, int]:
    """在文件行中查找上下文片段。"""
    if eof:
        # 如果是文件末尾，优先从文件末尾开始匹配
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(
    lines: list[str],
    index: int,
) -> tuple[list[str], list[Chunk], int, bool]:
    """预读并解析下一个变更片段（+/-/  行）。"""
    old: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            ),
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"无效行: {s}")
        index += 1

        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(f"无效行: {s}")
        s = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    ),
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            ),
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise DiffError("此片段中无内容")
    return old, chunks, index, False


# --------------------------------------------------------------------------- #
#  Patch → Commit 转换和 Commit 应用
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    """根据 PatchAction 中的变更块，应用更新到文件内容。"""
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file 被非更新操作调用")
    orig_lines = text.split("\n")
    dest_lines: list[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: 变更块的起始索引 {chunk.orig_index} 超出文件长度",
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: 在 {orig_index} > {chunk.orig_index} 处存在重叠的变更块",
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: dict[str, str]) -> Commit:
    """将解析后的 Patch 对象转换为 Commit 对象。"""
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE,
                old_content=orig[path],
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD 操作缺少文件内容")
            commit.changes[path] = FileChange(
                type=ActionType.ADD,
                new_content=action.new_file,
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  面向用户的辅助函数
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: dict[str, str], prefix: Path | None = None) -> tuple[Patch, int]:
    """将补丁文本转换为 Patch 对象。"""
    lines = text.splitlines()  # 保留空行
    if (
        len(lines) < 2
        or not Parser._norm(lines[0]).startswith("*** Begin Patch")
        or Parser._norm(lines[-1]) != "*** End Patch"
    ):
        raise DiffError("无效的补丁文本 - 缺少开始/结束标记")

    parser = Parser(current_files=orig, lines=lines, index=1, prefix=prefix)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str, prefix: Path | None = None) -> list[str]:
    """从补丁文本中识别出需要被更新或删除的文件列表。"""
    lines = text.splitlines()
    update_files = [line[len("*** Update File: ") :] for line in lines if line.startswith("*** Update File: ")]
    delete_files = [line[len("*** Delete File: ") :] for line in lines if line.startswith("*** Delete File: ")]
    all_files = update_files + delete_files

    if prefix is None:
        return all_files
    else:
        return [str(prefix / file) for file in all_files]


def identify_files_added(text: str, prefix: Path | None = None) -> list[str]:
    """从补丁文本中识别出需要被添加的文件列表。"""
    lines = text.splitlines()
    added_files = [line[len("*** Add File: ") :] for line in lines if line.startswith("*** Add File: ")]

    if prefix is None:
        return added_files
    else:
        return [str(prefix / file) for file in added_files]


# --------------------------------------------------------------------------- #
#  文件系统辅助函数
# --------------------------------------------------------------------------- #
def load_files(paths: list[str], open_fn: Callable[[str], str]) -> dict[str, str]:
    """加载多个文件的内容。"""
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
    inplace: bool = False,
) -> None | dict:
    """将 Commit 对象应用到文件系统。"""
    batch_edit = {}
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            remove_fn(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"文件 {path} 的 ADD 变更缺少内容")
            write_fn(path, change.new_content)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"文件 {path} 的 UPDATE 变更缺少新内容")
            if inplace:
                target = change.move_path or path
                write_fn(target, change.new_content)
                if change.move_path:
                    remove_fn(path)
            batch_edit[path] = change.new_content
    return batch_edit


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
    inplace: bool = False,
    prefix: Path | None = None,
) -> str:
    """处理整个补丁应用流程。"""
    if not text.startswith("*** Begin Patch"):
        raise DiffError("补丁文本必须以 *** Begin Patch 开始")
    paths = identify_files_needed(text, prefix)
    orig = load_files(paths, open_fn)
    patch, _fuzz = text_to_patch(text, orig, prefix)
    commit = patch_to_commit(patch, orig)
    batch_edit = apply_commit(commit, write_fn, remove_fn, inplace)
    return batch_edit


# --------------------------------------------------------------------------- #
#  默认的文件系统操作函数
# --------------------------------------------------------------------------- #
def open_file(path: str) -> str:
    """默认的打开文件函数。"""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str, content: str) -> None:
    """默认的写入文件函数。"""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wt", encoding="utf-8") as fh:
        fh.write(content)


def remove_file(path: str) -> None:
    """默认的删除文件函数。"""
    pathlib.Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  命令行入口点
# --------------------------------------------------------------------------- #
def apply_patch_from_text(patch_text: str, inplace: bool = False, prefix: Path | None = None) -> str:
    """
    从文本应用补丁到文件系统，与 main() 函数功能相同，但通过参数接收输入。
    """
    if not patch_text:
        raise DiffError("补丁文本不能为空")

    try:
        result = process_patch(patch_text, open_file, write_file, remove_file, inplace, prefix)
        return result
    except DiffError as exc:
        raise exc


def main() -> None:
    """命令行主函数，从标准输入读取补丁并应用。"""
    import sys

    patch_text = sys.stdin.read()
    if not patch_text:
        print("请通过标准输入传递补丁文本", file=sys.stderr)
        return
    try:
        result = process_patch(patch_text, open_file, write_file, remove_file)
    except DiffError as exc:
        print(exc, file=sys.stderr)
        return
    print(result)


if __name__ == "__main__":
    main()
