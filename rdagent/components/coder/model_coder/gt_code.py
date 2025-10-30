"""
这只是一个示例。
它将被一系列真实的基准任务所取代。
"""

import math
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj


class AntiSymmetricConv(torch.nn.Module):
    r"""来自论文《Anti-Symmetric DGN: a stable architecture for Deep Graph Networks》
    <https://openreview.net/forum?id=J3Y7cgZOOS> 的反对称图卷积算子。

    数学公式:
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \epsilon \cdot \sigma \left(
            (\mathbf{W}-\mathbf{W}^T-\gamma \mathbf{I}) \mathbf{x}_i +
            \Phi(\mathbf{X}, \mathcal{N}_i) + \mathbf{b}\right),

    其中 :math:`\Phi(\mathbf{X}, \mathcal{N}_i)` 表示一个
    :class:`~torch.nn.conv.MessagePassing` 层。

    Args:
        in_channels (int): 每个输入样本的大小。
        phi (MessagePassing, optional): 消息传递模块 :math:`\Phi`。
            如果设置为 :obj:`None`，将默认使用 :class:`~torch_geometric.nn.conv.GCNConv` 层。
            (默认: :obj:`None`)
        num_iters (int, optional): 调用反对称深度图网络算子的次数。
            (默认: :obj:`1`)
        epsilon (float, optional): 离散化步长 :math:`\epsilon`。
            (默认: :obj:`0.1`)
        gamma (float, optional): 扩散强度 :math:`\gamma`。
            它调节方法的稳定性。(默认: :obj:`0.1`)
        act (str, optional): 非线性激活函数 :math:`\sigma`，
            例如 :obj:`"tanh"` 或 :obj:`"relu"`。(默认: :class:`"tanh"`)
        act_kwargs (Dict[str, Any], optional): 传递给由 :obj:`act` 定义的
            相应激活函数的参数。(默认: :obj:`None`)
        bias (bool, optional): 如果设置为 :obj:`False`，该层将不学习
            加性偏置。(默认: :obj:`True`)

    形状:
        - **输入:**
          节点特征 :math:`(|\mathcal{V}|, F_{in})`,
          边索引 :math:`(2, |\mathcal{E}|)`,
          边权重 :math:`(|\mathcal{E}|)` *(可选)*
        - **输出:** 节点特征 :math:`(|\mathcal{V}|, F_{in})`
    """

    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = "tanh",
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activation_resolver(act, **(act_kwargs or {}))

        if phi is None:
            # 如果未提供消息传递模块，则默认为GCNConv
            phi = GCNConv(in_channels, in_channels, bias=False)

        self.W = Parameter(torch.empty(in_channels, in_channels))
        # 将单位矩阵注册为缓冲区，它不是模型参数
        self.register_buffer("eye", torch.eye(in_channels))
        self.phi = phi

        if bias:
            self.bias = Parameter(torch.empty(in_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""重置模块的所有可学习参数。"""
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.phi.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        r"""执行模块的前向传播。"""
        # 构建反对称权重矩阵
        antisymmetric_W = self.W - self.W.t() - self.gamma * self.eye

        # 迭代更新节点特征
        for _ in range(self.num_iters):
            # 消息传递
            h = self.phi(x, edge_index, *args, **kwargs)
            # 线性变换
            h = x @ antisymmetric_W.t() + h

            if self.bias is not None:
                h += self.bias

            if self.act is not None:
                h = self.act(h)

            # 更新节点表示
            x = x + self.epsilon * h

        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, "
            f"phi={self.phi}, "
            f"num_iters={self.num_iters}, "
            f"epsilon={self.epsilon}, "
            f"gamma={self.gamma})"
        )


if __name__ == "__main__":
    # 加载示例数据
    node_features = torch.load("node_features.pt")
    edge_index = torch.load("edge_index.pt")

    # 模型实例化和前向传播
    model = AntiSymmetricConv(in_channels=node_features.size(-1))
    output = model(node_features, edge_index)

    # 将输出保存到文件
    torch.save(output, "gt_output.pt")
