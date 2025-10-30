"""
用于测试 feature coder（基于 CoSTEER）组件的辅助函数。
- 主要测试开发循环（developer loop）是否正常工作。

它不是：
- 接口单元测试（例如，CoSTEER 循环中的工作空间评估器）。
"""

from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


def develop_one_competition(competition: str):
    """
    针对单个Kaggle竞赛，开发并测试特征工程部分的代码。

    Args:
        competition (str): 竞赛名称。
    """
    # 1. 初始化场景和特征代码生成器
    scen = KaggleScen(competition=competition)
    feature_coder = FeatureCoSTEER(scen)

    # 2. 加载特征工程的规范文档
    # 注意：这里的路径是硬编码的，仅用于测试
    with open("./rdagent/scenarios/kaggle/tpl_ex/aerial-cactus-identification/spec/feature.md", "r") as file:
        feat_spec = file.read()

    # 3. 创建特征工程任务和实验
    ft = FeatureTask(name="FeatureTask", description=scen.get_competition_full_desc())
    exp = DSExperiment(
        sub_tasks=[ft],
    )

    # 4. 将前置步骤的代码（数据加载）和规范文档注入到工作空间
    with open("./rdagent/scenarios/kaggle/tpl_ex/aerial-cactus-identification/load_data.py", "r") as file:
        load_data_code = file.read()
    exp.experiment_workspace.inject_files(**{"load_data.py": load_data_code, "spec/feature.md": feat_spec})

    # 5. 调用 develop 方法生成代码
    exp = feature_coder.develop(exp)


if __name__ == "__main__":
    # 针对 "aerial-cactus-identification" 竞赛运行开发流程
    develop_one_competition("aerial-cactus-identification")
