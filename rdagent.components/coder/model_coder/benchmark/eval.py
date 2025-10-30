# TODO: 应继承自基准测试的基类
import torch

from rdagent.components.coder.model_coder.model import ModelFBWorkspace


def get_data_conf(init_val):
    """
    获取数据配置。
    TODO: 在工作流中设计这一步。
    """
    in_dim = 1000
    in_channels = 128
    exec_config = {"model_eval_param_init": init_val}
    node_feature = torch.randn(in_dim, in_channels)
    edge_index = torch.randint(0, in_dim, (2, 2000))
    return (node_feature, edge_index), exec_config


class ModelImpValEval:
    """
    通过改变输入并观察输出来评估模型结构的相似性。

    假设:
    - 如果模型结构相似，当我们改变输入时，输出会以类似的方式改变。

    挑战:
    - 与实现普通模型不同，这里的层中包含参数（模型算子通常没有参数或参数是给定的）。
    - 我们尝试以相似的值初始化模型参数，因此只有模型结构是不同的。

    比较以下序列的相关性：
    - modelA[init1](input1).hidden_out1, modelA[init1](input2).hidden_out1, ...
    - modelB[init1](input1).hidden_out1, modelB[init1](input2).hidden_out1, ...

    对于每个隐藏层输出，我们可以计算一个相关性。平均相关性将作为评估指标。
    """

    def evaluate(self, gt: ModelFBWorkspace, gen: ModelFBWorkspace):
        """
        评估生成模型 (gen) 与真实模型 (gt) 的相似性。

        Args:
            gt (ModelFBWorkspace): 真实模型的工作空间。
            gen (ModelFBWorkspace): 生成模型的工作空间。

        Returns:
            float: 两个模型输出之间的平均皮尔逊相关系数。
        """
        round_n = 10  # 评估轮次

        eval_pairs: list[tuple] = []

        # 1. 生成评估数据对
        # 遍历不同的输入值
        for _ in range(round_n):
            # 遍历不同的模型初始化参数
            for init_val in [-0.2, -0.1, 0.1, 0.2]:
                _, gt_res = gt.execute(input_value=init_val, param_init_value=init_val)
                _, res = gen.execute(input_value=init_val, param_init_value=init_val)
                # 只有在两个模型都成功执行时才添加评估对
                if gt_res is not None and res is not None:
                    eval_pairs.append((res, gt_res))

        # 如果没有成功的评估对，返回0.0
        if not eval_pairs:
            return 0.0

        # 2. 扁平化和拼接输出
        res_batch, gt_res_batch = [], []
        for res, gt_res in eval_pairs:
            res_batch.append(res.reshape(-1))
            gt_res_batch.append(gt_res.reshape(-1))
        res_batch = torch.stack(res_batch)
        gt_res_batch = torch.stack(gt_res_batch)

        res_batch = res_batch.detach().numpy()
        gt_res_batch = gt_res_batch.detach().numpy()

        # 3. 计算每个隐藏输出维度的皮尔逊相关性
        def norm(x):
            """对数据进行标准化 (z-score)"""
            return (x - x.mean(axis=0)) / x.std(axis=0)

        # 逐个维度计算相关性
        dim_corr = (norm(res_batch) * norm(gt_res_batch)).mean(axis=0)

        # 4. 聚合所有相关性
        avr_corr = dim_corr.mean()

        # FIXME:
        # 相关性值过高（例如 0.944）。
        # 需要检查这是否是一个好的评估方法！！
        # 可能是因为相同的初始参数导致了极高的相关性，而与模型结构无关。
        return avr_corr
