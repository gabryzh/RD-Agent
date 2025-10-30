# 导入Path类，用于处理文件系统路径
from pathlib import Path

# 导入dill作为pickle，dill可以序列化更多Python对象
import dill as pickle  # type: ignore[import-untyped]

# 从rdagent.log模块导入rdagent_logger
from rdagent.log import rdagent_logger as logger


class KnowledgeBase:
    """知识库基类，用于加载和转储知识。"""
    def __init__(self, path: str | Path | None = None) -> None:
        """
        初始化知识库。
        :param path: 知识库文件的路径。
        """
        self.path = Path(path) if path else None
        self.load()

    def load(self) -> None:
        """从文件加载知识库。"""
        if self.path is not None and self.path.exists():
            with self.path.open("rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    # 如果加载的是字典，则更新当前对象的属性
                    self.__dict__.update({k: v for k, v in loaded.items() if k != "path"})
                else:
                    # 如果加载的是对象，则更新当前对象的属性
                    self.__dict__.update({k: v for k, v in loaded.__dict__.items() if k != "path"})

    def dump(self) -> None:
        """将知识库转储到文件。"""
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.__dict__, self.path.open("wb"))
        else:
            logger.warning("知识库路径未设置，转储失败。")
