"""
生成用于测试模型工作流输出的数据集
"""

# 导入必要的库和模块
from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.eval import (
    ModelGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


# 以任务、spec.md 和特征作为输入，生成反馈作为输出
def develop_one_competition(competition: str):
    """
    针对单个竞赛项目进行模型开发。

    这个函数封装了为一个指定的Kaggle竞赛项目初始化场景、创建模型任务、
    准备工作空间、运行CoSTEER开发流程的完整过程。

    参数:
        competition (str): 竞赛项目的名称。
    """
    # 初始化竞赛场景
    scen = KaggleScen(competition=competition)
    # 初始化模型编码器
    model_coder = ModelCoSTEER(scen)

    # 创建模型任务
    mt = ModelTask(
        name="ModelTask",
        description="一个CNN模型",
        model_type="CNN",
        architecture="\\hat{y}_u = CNN(X_u)",
        hyperparameters="...",
        base_code="",
    )

    # 定义模板示例文件的路径
    tpl_ex_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "scenarios" / "kaggle" / "tpl_ex" / competition
    # 定义需要注入工作空间的文件列表
    injected_file_names = ["spec/model.md", "load_data.py", "feature.py", "model01.py"]

    # 创建模型实验的工作空间
    modelexp = FBWorkspace()
    # 将文件注入工作空间
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        modelexp.inject_files(**{file_name: file_path.read_text()})

    # 将初始模型代码添加到任务中
    mt.base_code += modelexp.file_dict["model01.py"]
    # 创建数据科学实验
    exp = DSExperiment(
        sub_tasks=[mt],
    )

    # --- 以下是被注释掉的测试代码 ---
    # 测试评估器:
    """eva = ModelGeneralCaseSpecEvaluator(scen=scen)
    exp.feedback = eva.evaluate(target_task=mt, queried_knowledge=None, implementation=modelexp, gt_implementation=None)
    print(exp.feedback)"""

    # 测试演进策略:
    """es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
    new_code = es.implement_one_task(target_task=mt, queried_knowledge=None, workspace=modelexp)
    print(new_code)"""

    # 运行实验
    # 将文件注入到主实验的工作空间
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        exp.experiment_workspace.inject_files(**{file_name: file_path.read_text()})

    # 使用模型编码器进行开发
    exp = model_coder.develop(exp)


if __name__ == "__main__":
    # 运行针对 "aerial-cactus-identification" 竞赛的开发流程
    develop_one_competition("aerial-cactus-identification")
    # 命令行运行指令示例:
    # dotenv run -- python rdagent/components/coder/data_science/model/test.py
