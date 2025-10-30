from __future__ import annotations

import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, NoReturn

from rdagent.components.knowledge_management.vector_base import (
    KnowledgeMetaData,
    PDVectorBase,
    VectorBase,
    cosine,
)
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend

# 类型别名，Node 代表知识元数据
Node = KnowledgeMetaData


class UndirectedNode(Node):
    """无向图节点类，继承自 Node。"""
    def __init__(self, content: str = "", label: str = "", embedding: Any = None, appendix: Any = None) -> None:
        """
        初始化一个无向图节点。

        Args:
            content (str): 节点内容。
            label (str): 节点标签。
            embedding (Any): 节点的嵌入向量。
            appendix (Any): 附加信息。
        """
        super().__init__(content, label, embedding)
        self.neighbors: set[UndirectedNode] = set()  # 邻居节点集合
        self.appendix = appendix  # 存储任何附加信息
        assert isinstance(content, str), "节点内容必须是字符串"

    def add_neighbor(self, node: UndirectedNode) -> None:
        """添加一个邻居节点（无向关系）。"""
        self.neighbors.add(node)
        node.neighbors.add(self)

    def remove_neighbor(self, node: UndirectedNode) -> None:
        """移除一个邻居节点（无向关系）。"""
        if node in self.neighbors:
            self.neighbors.remove(node)
            node.neighbors.remove(self)

    def get_neighbors(self) -> set[UndirectedNode]:
        """获取所有邻居节点。"""
        return self.neighbors

    def __str__(self) -> str:
        return (
            f"UndirectedNode(id={self.id}, label={self.label}, content={self.content[:100]}, "
            f"neighbors={self.neighbors})"
        )

    def __repr__(self) -> str:
        return (
            f"UndirectedNode(id={self.id}, label={self.label}, content={self.content[:100]}, "
            f"neighbors={self.neighbors})"
        )


class Graph(KnowledgeBase):
    """
    用于知识图谱搜索的基础图类。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.nodes = {}  # 存储图中所有节点的字典
        super().__init__(path=path)

    def size(self) -> int:
        """返回图的大小（节点数量）。"""
        return len(self.nodes)

    def get_node(self, node_id: str) -> Node | None:
        """根据节点ID获取节点。"""
        return self.nodes.get(node_id)

    def add_node(self, **kwargs: Any) -> NoReturn:
        """添加节点的抽象方法，需要在子类中实现。"""
        raise NotImplementedError

    def get_all_nodes(self) -> list[Node]:
        """获取图中所有的节点列表。"""
        return list(self.nodes.values())

    def get_all_nodes_by_label_list(self, label_list: list[str]) -> list[Node]:
        """根据标签列表获取所有匹配的节点。"""
        return [node for node in self.nodes.values() if node.label in label_list]

    def find_node(self, content: str, label: str) -> Node | None:
        """根据内容和标签查找节点。"""
        for node in self.nodes.values():
            if node.content == content and node.label == label:
                return node
        return None

    @staticmethod
    def batch_embedding(nodes: list[Node]) -> list[Node]:
        """
        批量为节点生成嵌入向量。
        """
        contents = [node.content for node in nodes]
        # OpenAI 创建嵌入 API 的输入最大长度为 16
        size = 16
        embeddings = []
        for i in range(0, len(contents), size):
            logger.info(
                f"为索引 {i} 到 {i + size} 的 {len(contents)} 个内容创建嵌入",
                tag="batch embedding",
            )
            embeddings.extend(
                APIBackend().create_embedding(input_content=contents[i : i + size]),
            )

        assert len(nodes) == len(embeddings), "节点列表长度必须等于嵌入列表长度"
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

    def __str__(self) -> str:
        return f"Graph(nodes={self.nodes})"


class UndirectedGraph(Graph):
    """
    无向图类，边没有方向性。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.vector_base: VectorBase = PDVectorBase()  # 向量数据库
        super().__init__(path=path)

    def __str__(self) -> str:
        return f"UndirectedGraph(nodes={self.nodes})"

    def add_node(
        self,
        node: UndirectedNode,
        neighbor: UndirectedNode = None,
        same_node_threshold: float = 0.95,  # noqa: ARG002
    ) -> None:
        """
        向图中添加节点和邻居。
        如果节点已存在（通过ID或内容+标签判断），则使用现有节点。
        否则，创建嵌入并将其添加到图和向量数据库中。

        Args:
            same_node_threshold (float): 相似度阈值，用于判断是否为同一节点（经验值）。
            node (UndirectedNode): 要添加的节点。
            neighbor (UndirectedNode, optional): 要连接的邻居节点。
        """
        if tmp_node := self.get_node(node.id):
            node = tmp_node
        elif tmp_node := self.find_node(content=node.content, label=node.label):
            node = tmp_node
        else:
            node.create_embedding()
            self.vector_base.add(document=node)
            self.nodes.update({node.id: node})

        if neighbor is not None:
            if tmp_neighbor := self.get_node(neighbor.id):
                neighbor = tmp_neighbor
            elif tmp_neighbor := self.find_node(content=neighbor.content, label=node.label):
                neighbor = tmp_neighbor
            else:
                neighbor.create_embedding()
                self.vector_base.add(document=neighbor)
                self.nodes.update({neighbor.id: neighbor})

            node.add_neighbor(neighbor)

    def add_nodes(self, node: UndirectedNode, neighbors: list[UndirectedNode]) -> None:
        """批量添加邻居节点。"""
        if not neighbors:
            self.add_node(node)
        else:
            for neighbor in neighbors:
                self.add_node(node, neighbor=neighbor)

    def get_node(self, node_id: str) -> UndirectedNode:
        """根据ID获取节点。"""
        return self.nodes.get(node_id)

    def get_node_by_content(self, content: str) -> UndirectedNode | None:
        """
        通过语义距离获取节点。
        """
        match = self.semantic_search(node=content, similarity_threshold=0.999)
        if match:
            return match[0]
        return None

    def get_nodes_within_steps(
        self,
        start_node: UndirectedNode,
        steps: int = 1,
        constraint_labels: list[str] | None = None,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        返回图中与起始节点距离小于等于指定步数的节点（BFS算法）。
        """
        visited = set()
        queue = deque([(start_node, 0)])
        result = []

        while queue:
            node, current_steps = queue.popleft()

            if current_steps > steps:
                break

            if node not in visited:
                visited.add(node)
                result.append(node)

                # 对邻居进行排序以确保结果的确定性
                for neighbor in sorted(self.get_node(node.id).neighbors, key=lambda x: x.content):
                    if neighbor not in visited and not (block and neighbor.label not in constraint_labels):
                        queue.append((neighbor, current_steps + 1))

        if constraint_labels:
            result = [node for node in result if node.label in constraint_labels]
        if start_node in result:
            result.remove(start_node)
        return result

    def get_nodes_intersection(
        self,
        nodes: list[UndirectedNode],
        steps: int = 1,
        constraint_labels: list[str] | None = None,
    ) -> list[UndirectedNode]:
        """
        获取多个节点在n步内连接的节点的交集。
        """
        min_nodes_count = 2
        assert len(nodes) >= min_nodes_count, "节点列表长度必须大于等于2"
        intersection = None

        for node in nodes:
            connected_nodes = self.get_nodes_within_steps(node, steps=steps, constraint_labels=constraint_labels)
            if intersection is None:
                intersection = connected_nodes
            else:
                intersection = self.intersection(nodes1=intersection, nodes2=connected_nodes)
        return intersection

    def semantic_search(
        self,
        node: UndirectedNode | str,
        similarity_threshold: float = 0.0,
        topk_k: int = None,
        constraint_labels: list[str] | None = None,
    ) -> list[UndirectedNode]:
        """
        通过节点的嵌入向量进行语义搜索。

        Args:
            node: 用于搜索的节点或内容字符串。
            similarity_threshold: 相似度阈值，低于此值的将被排除。
            topk_k: 返回的最相似节点的最大数量。
            constraint_labels: 标签约束，只在匹配标签的节点中搜索。

        Returns:
            按相似度排序的节点列表。
        """
        if isinstance(node, str):
            node = UndirectedNode(content=node)
        docs, scores = self.vector_base.search(
            content=node.content,
            topk_k=topk_k,
            similarity_threshold=similarity_threshold,
            constraint_labels=constraint_labels,
        )
        return [self.get_node(doc.id) for doc in docs]

    def clear(self) -> None:
        """清空图和向量数据库。"""
        self.nodes.clear()
        self.vector_base = PDVectorBase()

    def query_by_node(
        self,
        node: UndirectedNode,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        constraint_distance: float = 0,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        通过连接关系进行图搜索。如果结果链中没有靠近约束节点的节点，则返回空列表。

        Args:
            block: 除起始节点外，搜索只能流经 constraint_label 类型的节点。
        """
        nodes = self.get_nodes_within_steps(
            start_node=node,
            steps=step,
            constraint_labels=constraint_labels,
            block=block,
        )
        if constraint_node is not None:
            for n in nodes:
                if self.cal_distance(n, constraint_node) > constraint_distance:
                    return nodes
            return []
        return nodes

    def query_by_content(
        self,
        content: str | list[str],
        topk_k: int = 5,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        similarity_threshold: float = 0.0,
        constraint_distance: float = 0,
        *,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        通过内容相似度和连接关系搜索图。
        """
        if isinstance(content, str):
            content = [content]

        res_list = []
        for query in content:
            similar_nodes = self.semantic_search(
                content=query,
                topk_k=topk_k,
                similarity_threshold=similarity_threshold,
            )

            connected_nodes = []
            for node in similar_nodes:
                graph_query_node_res = self.query_by_node(
                    node,
                    step=step,
                    constraint_labels=constraint_labels,
                    constraint_node=constraint_node,
                    constraint_distance=constraint_distance,
                    block=block,
                )
                connected_nodes.extend(
                    [n for n in graph_query_node_res if n not in connected_nodes],
                )
                if len(connected_nodes) >= topk_k:
                    break

            res_list.extend(
                [n for n in connected_nodes[:topk_k] if n not in res_list],
            )
        return res_list

    @staticmethod
    def intersection(nodes1: list[UndirectedNode], nodes2: list[UndirectedNode]) -> list[UndirectedNode]:
        """返回两个节点列表的交集。"""
        return [node for node in nodes1 if node in nodes2]

    @staticmethod
    def different(nodes1: list[UndirectedNode], nodes2: list[UndirectedNode]) -> list[UndirectedNode]:
        """返回两个节点列表的对称差集。"""
        return list(set(nodes1).symmetric_difference(set(nodes2)))

    @staticmethod
    def cal_distance(node1: UndirectedNode, node2: UndirectedNode) -> float:
        """计算两个节点嵌入之间的余弦距离。"""
        return cosine(node1.embedding, node2.embedding)

    @staticmethod
    def filter_label(nodes: list[UndirectedNode], labels: list[str]) -> list[UndirectedNode]:
        """根据标签过滤节点列表。"""
        return [node for node in nodes if node.label in labels]


# --- 辅助函数 ---

def graph_to_edges(graph: dict[str, list[str]]) -> list[tuple[str, str]]:
    """将邻接表表示的图转换为边列表。"""
    edges = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            # 避免重复添加无向边
            if (node, neighbor) not in edges and (neighbor, node) not in edges:
                edges.append((node, neighbor))
    return edges


def assign_random_coordinate_to_node(
    nodes: list[str],
    scope: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict[str, tuple[float, float]]:
    """为节点分配随机坐标。"""
    coordinates = {}
    for node in nodes:
        x = random.SystemRandom().uniform(0, scope) + origin[0]
        y = random.SystemRandom().uniform(0, scope) + origin[1]
        coordinates[node] = (x, y)
    return coordinates


def assign_isometric_coordinate_to_node(
    nodes: list,
    x_step: float = 1.0,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> dict:
    """为节点分配等距坐标。"""
    coordinates = {}
    for i, node in enumerate(nodes):
        x = x_origin + i * x_step
        y = y_origin
        coordinates[node] = (x, y)
    return coordinates


def curly_node_coordinate(
    coordinates: dict,
    center_y: float = 1.0,
    r: float = 1.0,
) -> dict:
    """
    将节点的坐标弯曲成圆形路径。
    注意：此方法只能弯曲小于90度的角度，并且弯曲的线是圆的一部分。
    原始函数为：x**2 + (y-m)**2 = r**2
    """
    for node, coordinate in coordinates.items():
        coordinates[node] = (coordinate[0], center_y + (r**2 - coordinate[0] ** 2) ** 0.5)
    return coordinates
