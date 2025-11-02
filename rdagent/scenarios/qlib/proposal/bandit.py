# 导入 json 模块
import json
# 导入 math 模块
import math
# 从 dataclasses 模块导入 dataclass，用于创建数据类
from dataclasses import dataclass
# 从 pathlib 模块导入 Path 类
from pathlib import Path
# 从 typing 模块导入 List, Literal, Tuple
from typing import List, Literal, Tuple

# 导入 numpy 用于数值计算
import numpy as np


@dataclass
class Metrics:
    """
    一个数据类，用于存储从实验结果中提取的关键性能指标。
    """
    ic: float = 0.0  # 信息系数
    icir: float = 0.0  # 信息系数的信息比率
    rank_ic: float = 0.0  # 排名信息系数
    rank_icir: float = 0.0  # 排名信息系数的信息比率
    arr: float = 0.0  # 年化回报率
    ir: float = 0.0  # 信息比率
    mdd: float = 0.0  # 最大回撤
    sharpe: float = 0.0  # 夏普比率

    def as_vector(self) -> np.ndarray:
        """将指标转换为一个 numpy 向量，用于 Bandit 算法。"""
        # 注意：最大回撤 mdd 是负向指标，因此取负值
        return np.array(
            [
                self.ic,
                self.icir,
                self.rank_ic,
                self.rank_icir,
                self.arr,
                self.ir,
                -self.mdd,
                self.sharpe,
            ]
        )


def extract_metrics_from_experiment(experiment) -> Metrics:
    """
    从一个实验对象中提取性能指标并填充到 Metrics 对象中。
    """
    try:
        result = experiment.result
        ic = result.get("IC", 0.0)
        icir = result.get("ICIR", 0.0)
        rank_ic = result.get("Rank IC", 0.0)
        rank_icir = result.get("Rank ICIR", 0.0)
        arr = result.get("1day.excess_return_with_cost.annualized_return ", 0.0)
        ir = result.get("1day.excess_return_with_cost.information_ratio", 0.0)
        mdd = result.get("1day.excess_return_with_cost.max_drawdown", 1.0)  # 避免除以零
        sharpe = arr / -mdd if mdd != 0 else 0.0

        return Metrics(ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir, arr=arr, ir=ir, mdd=mdd, sharpe=sharpe)
    except Exception as e:
        print(f"提取指标时出错: {e}")
        return Metrics()


class LinearThompsonTwoArm:
    """
    实现一个双臂（"factor" 和 "model"）线性汤普森采样 Bandit 算法。
    该算法用于在给定上下文（即当前性能指标）的情况下，探索性地选择下一个最优动作。
    """
    def __init__(self, dim: int, prior_var: float = 1.0, noise_var: float = 1.0):
        self.dim = dim  # 上下文向量的维度
        self.noise_var = noise_var  # 噪声方差
        # 每个臂都有自己的后验分布参数：均值向量和精度矩阵（协方差矩阵的逆）
        self.mean = {
            "factor": np.zeros(dim),
            "model": np.zeros(dim),
        }
        self.precision = {
            "factor": np.eye(dim) / prior_var,
            "model": np.eye(dim) / prior_var,
        }

    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        """
        从给定臂的后验分布中采样一个权重向量，并计算预期的奖励。
        """
        P = self.precision[arm]
        P = 0.5 * (P + P.T)  # 确保精度矩阵对称

        eps = 1e-6
        try:
            # 通过 Cholesky 分解从多维高斯分布中采样
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，则直接使用均值作为采样结果
            w_sample = self.mean[arm]

        return float(np.dot(w_sample, x))

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        """
        根据观察到的奖励 `r` 和上下文 `x`，更新选定臂的后验分布。
        """
        P = self.precision[arm]
        P += np.outer(x, x) / self.noise_var
        self.precision[arm] = P
        self.mean[arm] = np.linalg.solve(P, P @ self.mean[arm] + (r / self.noise_var) * x)

    def next_arm(self, x: np.ndarray) -> str:
        """
        根据当前上下文 `x`，为每个臂采样一个预期奖励，并选择奖励最高的臂。
        """
        scores = {arm: self.sample_reward(arm, x) for arm in ("factor", "model")}
        return max(scores, key=scores.get)


class EnvController:
    """
    环境控制器，封装了 Bandit 算法的决策、记录和奖励计算逻辑。
    """
    def __init__(self, weights: Tuple[float, ...] = None) -> None:
        # 定义不同性能指标在计算总奖励时的权重
        self.weights = np.asarray(weights or (0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2))
        # 初始化 Bandit 算法实例
        self.bandit = LinearThompsonTwoArm(dim=8, prior_var=10.0, noise_var=0.5)

    def reward(self, m: Metrics) -> float:
        """根据加权平均计算总奖励。"""
        return float(np.dot(self.weights, m.as_vector()))

    def decide(self, m: Metrics) -> str:
        """根据当前指标 `m` 决定下一个动作（"factor" 或 "model"）。"""
        x = m.as_vector()
        return self.bandit.next_arm(x)

    def record(self, m: Metrics, arm: str) -> None:
        """记录一个动作的结果，并更新 Bandit 算法。"""
        r = self.reward(m)
        self.bandit.update(arm, m.as_vector(), r)
