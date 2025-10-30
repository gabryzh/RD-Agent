"""
用于测试 ensemble coder（基于 CoSTEER）组件的辅助函数。
"""

import sys
from pathlib import Path

from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen

# 将特定竞赛的模板文件夹添加到 Python 路径中，以便导入相关模块
COMPETITION_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "scenarios"
    / "kaggle"
    / "tpl_ex"
    / "aerial-cactus-identification"
)
sys.path.append(str(COMPETITION_PATH))

# 类型别名
EnsembleExperiment = DSExperiment


def load_ensemble_spec() -> str:
    """加载集成（ensemble）阶段的规范文档。"""
    spec_path = COMPETITION_PATH / "spec" / "ensemble.md"
    with open(spec_path, "r") as f:
        return f.read()


def develop_one_competition(competition: str) -> EnsembleExperiment:
    """
    针对单个Kaggle竞赛，开发并测试集成（ensemble）部分的代码。

    Args:
        competition (str): 竞赛名称。

    Returns:
        EnsembleExperiment: 开发完成的实验对象。
    """
    # 1. 初始化场景和集成代码生成器
    scen = KaggleScen(competition=competition)
    ensemble_coder = EnsembleCoSTEER(scen)

    # 2. 加载集成规范
    ensemble_spec = load_ensemble_spec()

    # 3. 创建集成任务
    task = EnsembleTask(
        name="EnsembleTask",
        description="""
        实现模型预测的集成和决策。
        """,
    )

    # 4. 创建实验对象
    exp = EnsembleExperiment(sub_tasks=[task])

    # 5. 将规范文件注入到实验工作空间
    exp.experiment_workspace.inject_files(**{"spec/ensemble.md": ensemble_spec})

    # 6. 调用 develop 方法生成代码
    exp = ensemble_coder.develop(exp)
    return exp


if __name__ == "__main__":
    # 针对 "aerial-cactus-identification" 竞赛运行开发流程
    develop_one_competition("aerial-cactus-identification")
