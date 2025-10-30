"""
生成用于测试工作流输出的数据集。
该脚本主要用于测试和调试数据科学工作流的CoSTEER代码生成器。
"""

from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.eval import (
    WorkflowGeneralCaseSpecEvaluator,
)
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import FBWorkspace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import KaggleScen


def develop_one_competition(competition: str):
    """
    针对单个Kaggle竞赛，开发并测试完整的工作流代码。

    Args:
        competition (str): 竞赛的名称。
    """
    # 1. 初始化场景和工作流代码生成器
    scen = KaggleScen(competition=competition)
    workflow_coder = WorkflowCoSTEER(scen)

    # 2. 创建一个工作流任务
    wt = WorkflowTask(
        name="WorkflowTask",
        description="将现有的 load_data, feature, model 和 ensemble 过程整合到一个完整的工作流中。",
        base_code="",
    )

    # 3. 准备注入的模板文件
    #    这些文件代表了数据科学流程中各个阶段的已有代码实现。
    tpl_ex_path = Path(__file__).resolve().parent.parent.parent.parent.parent / f"scenarios/kaggle/tpl_ex/{competition}"
    injected_file_names = ["spec/workflow.md", "load_data.py", "feature.py", "model01.py", "ensemble.py", "main.py"]

    # 4. 创建一个工作空间并注入文件
    #    这部分代码被注释掉了，但其目的是模拟一个已包含各部分代码的工作空间。
    # workflowexp = FBWorkspace()
    # for file_name in injected_file_names:
    #     file_path = tpl_ex_path / file_name
    #     workflowexp.inject_files(**{file_name: file_path.read_text()})
    # wt.base_code += workflowexp.file_dict["main.py"]

    # 5. 创建实验对象
    exp = DSExperiment(
        sub_tasks=[wt],
    )

    """ 以下是被注释掉的测试代码块 """
    # 测试演进策略（implement_one_task）
    # es = WorkflowMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)
    # new_code = es.implement_one_task(target_task=wt, queried_knowledge=None, workspace = workflowexp)
    # print(new_code)

    # 测试评估器（evaluate）
    # eva = WorkflowGeneralCaseSpecEvaluator(scen=scen)
    # exp.feedback = eva.evaluate(target_task=wt, queried_knowledge=None, implementation=workflowexp, gt_implementation=None)
    # print(exp.feedback)
    """ """

    # 6. 将模板文件注入到主实验工作空间
    for file_name in injected_file_names:
        file_path = tpl_ex_path / file_name
        if file_path.exists():
            exp.experiment_workspace.inject_files(**{file_name: file_path.read_text()})

    # 7. 运行CoSTEER的develop方法，生成最终的工作流代码
    exp = workflow_coder.develop(exp)


if __name__ == "__main__":
    # 针对 "aerial-cactus-identification" 竞赛运行开发流程
    develop_one_competition("aerial-cactus-identification")
    # 运行命令示例:
    # dotenv run -- python rdagent/components/coder/data_science/workflow/test.py
