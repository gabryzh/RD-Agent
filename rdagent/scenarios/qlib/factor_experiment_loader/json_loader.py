# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 导入测试用例相关的类
from rdagent.components.benchmark.eval_method import TestCase, TestCases
# 导入因子实验相关的类
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
# 导入因子实验加载器基类
from rdagent.components.loader.experiment_loader import FactorExperimentLoader
# 导入核心实验类和加载器基类
from rdagent.core.experiment import Experiment, Loader
# 导入 Qlib 因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class FactorExperimentLoaderFromDict(FactorExperimentLoader):
    """
    从字典加载因子实验的加载器。
    """
    def load(self, factor_dict: dict) -> QlibFactorExperiment:
        """从字典加载数据。"""
        task_l = []
        # 遍历字典中的每个因子
        for factor_name, factor_data in factor_dict.items():
            # 创建一个 FactorTask 实例
            task = FactorTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            task_l.append(task)
        # 使用任务列表创建一个 QlibFactorExperiment 实例
        exp = QlibFactorExperiment(sub_tasks=task_l)
        return exp


class FactorExperimentLoaderFromJsonFile(FactorExperimentLoader):
    """
    从 JSON 文件加载因子实验的加载器。
    """
    def load(self, json_file_path: Path) -> list:
        """从 JSON 文件加载数据。"""
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        # 复用 FactorExperimentLoaderFromDict 的逻辑
        return FactorExperimentLoaderFromDict().load(factor_dict)


class FactorExperimentLoaderFromJsonString(FactorExperimentLoader):
    """
    从 JSON 字符串加载因子实验的加载器。
    """
    def load(self, json_string: str) -> list:
        """从 JSON 字符串加载数据。"""
        factor_dict = json.loads(json_string)
        # 复用 FactorExperimentLoaderFromDict 的逻辑
        return FactorExperimentLoaderFromDict().load(factor_dict)


# TODO: 加载器目前只支持任务或实验的泛型，测试用例可能会导致 CI 错误
# class FactorTestCaseLoaderFromJsonFile(Loader[TestCases]):
class FactorTestCaseLoaderFromJsonFile:
    """
    从 JSON 文件加载因子测试用例的加载器。
    """
    def load(self, json_file_path: Path) -> TestCases:
        """从 JSON 文件加载数据。"""
        with open(json_file_path, "r") as file:
            factor_dict = json.load(file)
        test_cases = TestCases()
        # 遍历字典中的每个因子
        for factor_name, factor_data in factor_dict.items():
            # 创建一个 FactorTask 实例
            task = FactorTask(
                factor_name=factor_name,
                factor_description=factor_data["description"],
                factor_formulation=factor_data["formulation"],
                variables=factor_data["variables"],
            )
            # 创建一个工作区用于存放基准代码 (ground truth)
            gt = FactorFBWorkspace(task, raise_exception=False)
            code = {"factor.py": factor_data["gt_code"]}
            gt.inject_files(**code)
            # 创建一个 TestCase 实例并添加到测试用例列表中
            test_cases.test_case_l.append(TestCase(task, gt))

        return test_cases
