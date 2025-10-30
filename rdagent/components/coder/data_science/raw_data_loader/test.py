"""
用于测试 raw_data_loader 编码器（基于CoSTEER）组件的辅助函数。
- 开发者循环是否正常工作

它不是：
- 它不是接口单元测试（即CoSTEER循环中的工作空间评估器）
"""

# 导入必要的库和模块
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


def develop_one_competition(competition: str):  # -> experiment
    """
    针对单个竞赛项目进行数据加载器的开发。

    这个函数封装了为一个指定的Kaggle竞赛项目初始化场景、创建数据加载任务、
    并运行CoSTEER开发流程的完整过程。

    参数:
        competition (str): 竞赛项目的名称。
    """
    # 初始化竞赛场景
    scen = KaggleScen(competition=competition)
    # 初始化数据加载器编码器
    data_loader_coder = DataLoaderCoSTEER(scen)

    # 创建实验
    # 定义一个数据加载任务
    dlt = DataLoaderTask(name="DataLoaderTask", description="")
    # 创建一个包含该任务的数据科学实验
    exp = DSExperiment(
        sub_tasks=[dlt],
    )

    # 开发实验
    # 调用数据加载器编码器的 develop 方法来执行开发流程
    exp = data_loader_coder.develop(exp)


if __name__ == "__main__":
    # 运行针对 "aerial-cactus-identification" 竞赛的开发流程
    develop_one_competition("aerial-cactus-identification")
