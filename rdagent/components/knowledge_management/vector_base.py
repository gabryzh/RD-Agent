import uuid
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from scipy.spatial.distance import cosine

from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend


class KnowledgeMetaData:
    """知识元数据类，用于封装知识的基本信息。"""
    def __init__(self, content: str = "", label: str = None, embedding=None, identity=None):
        """
        初始化知识元数据。

        Args:
            content (str): 内容文本。
            label (str, optional): 内容的标签。
            embedding (optional): 内容的嵌入向量。
            identity (optional): 自定义ID。如果为None，则根据内容生成UUID。
        """
        self.label = label
        self.content = content
        # 基于内容生成唯一的ID，除非提供了自定义ID
        self.id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(self.content))) if identity is None else identity
        self.embedding = embedding
        self.trunks = []  # 内容分块
        self.trunks_embedding = []  # 内容分块的嵌入

    def split_into_trunk(self, size: int = 1000, overlap: int = 0):
        """
        将内容分割成多个块（trunk），并为每个块创建嵌入。

        Args:
            size (int): 每个块的大小。
            overlap (int): 块之间的重叠大小（当前未实现）。
        """

        def split_string_into_chunks(string: str, chunk_size: int) -> list[str]:
            """辅助函数：将字符串按指定大小切块。"""
            chunks = []
            for i in range(0, len(string), chunk_size):
                chunk = string[i : i + chunk_size]
                chunks.append(chunk)
            return chunks

        self.trunks = split_string_into_chunks(self.content, chunk_size=size)
        self.trunks_embedding = APIBackend().create_embedding(input_content=self.trunks)

    def create_embedding(self):
        """为完整内容创建嵌入向量。"""
        if self.embedding is None:
            self.embedding = APIBackend().create_embedding(input_content=self.content)

    def from_dict(self, data: dict):
        """从字典加载属性。"""
        for key, value in data.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return f"Document(id={self.id}, label={self.label}, data={self.content})"


# 类型别名，Document 与 KnowledgeMetaData 等价
Document = KnowledgeMetaData


def contents_to_documents(contents: List[str], label: str = None) -> List[Document]:
    """
    将内容字符串列表批量转换为 Document 对象列表，并生成嵌入。
    """
    # OpenAI 创建嵌入 API 的输入最大长度为 16
    size = 16
    embedding = []
    # 分批调用嵌入API
    for i in range(0, len(contents), size):
        embedding.extend(APIBackend().create_embedding(input_content=contents[i : i + size]))
    # 创建 Document 对象
    docs = [Document(content=c, label=label, embedding=e) for c, e in zip(contents, embedding)]
    return docs


class VectorBase(KnowledgeBase):
    """
    用于处理向量存储和查询的基类。
    """

    def add(self, document: Union[Document, List[Document]]):
        """
        向向量数据库中添加新文档。
        """
        pass

    def search(self, content: str, topk_k: int | None = None, similarity_threshold: float = 0) -> List[Document]:
        """
        通过内容在向量数据库中进行搜索。
        """
        pass


class PDVectorBase(VectorBase):
    """
    使用 Pandas 实现的向量数据库。
    """

    def __init__(self, path: Union[str, Path] = None):
        # 初始化一个空的DataFrame用于存储向量数据
        self.vector_df = pd.DataFrame(columns=["id", "label", "content", "embedding"])
        super().__init__(path)

    def shape(self):
        """返回DataFrame的形状。"""
        return self.vector_df.shape

    def add(self, document: Union[Document, List[Document]]):
        """
        向 DataFrame 中添加新的文档。
        如果文档有分块，会将每个分块作为单独的行添加。
        """
        if isinstance(document, Document):
            if document.embedding is None:
                document.create_embedding()
            # 创建包含主文档和其所有分块的记录列表
            docs = [
                {
                    "id": document.id,
                    "label": document.label,
                    "content": document.content,
                    "trunk": document.content,
                    "embedding": document.embedding,
                }
            ]
            docs.extend(
                [
                    {
                        "id": document.id,
                        "label": document.label,
                        "content": document.content,
                        "trunk": trunk,
                        "embedding": embedding,
                    }
                    for trunk, embedding in zip(document.trunks, document.trunks_embedding)
                ]
            )
            self.vector_df = pd.concat([self.vector_df, pd.DataFrame(docs)], ignore_index=True)
        else:
            # 如果输入是文档列表，则递归添加
            for doc in document:
                self.add(document=doc)

    def search(
        self,
        content: str,
        topk_k: int | None = None,
        similarity_threshold: float = 0,
        constraint_labels: list[str] | None = None,
    ) -> Tuple[List[Document], List]:
        """
        通过内容的嵌入向量在 DataFrame 中进行搜索。

        Args:
            content (str): 要搜索的内容。
            topk_k (int, optional): 返回最相似的 top-k 个结果。
            similarity_threshold (float, optional): 相似度阈值。
            constraint_labels (list[str], optional): 标签约束。

        Returns:
            Tuple[List[Document], List]: 匹配的文档对象列表和对应的相似度分数列表。
        """
        # 如果DataFrame为空，直接返回空结果
        if not self.vector_df.shape[0]:
            return [], []

        # 为查询内容创建嵌入
        document = Document(content=content)
        document.create_embedding()

        # 根据标签进行过滤
        filtered_df = self.vector_df
        if constraint_labels is not None:
            filtered_df = self.vector_df[self.vector_df["label"].isin(constraint_labels)]

        # 计算余弦相似度
        # `cosine` 函数计算的是余弦距离，所以用 1 减去它得到相似度
        similarities = filtered_df["embedding"].apply(
            lambda x: 1 - cosine(x, document.embedding)
        )

        # 应用相似度阈值和 top-k 限制
        searched_similarities = similarities[similarities > similarity_threshold]
        if topk_k is not None:
            searched_similarities = searched_similarities.nlargest(topk_k)

        # 获取最相似的文档
        most_similar_docs = filtered_df.loc[searched_similarities.index]

        # 将结果转换为 Document 对象列表
        docs = []
        for _, similar_docs in most_similar_docs.iterrows():
            docs.append(Document().from_dict(similar_docs.to_dict()))

        return docs, searched_similarities.to_list()
