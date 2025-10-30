from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash


class FactorTask(CoSTEERTask):
    """
    因子任务类，定义了生成一个因子所需的所有信息。
    """
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = factor_name  # TODO: 为了 pickle 版本兼容性而保留，后续版本可移除
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """为保持兼容性而设置的属性。"""
        return self.description

    def get_task_information(self) -> str:
        """获取任务的详细信息字符串。"""
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_brief_information(self) -> str:
        """获取任务的简要信息字符串。"""
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self) -> dict:
        """获取任务信息和实现结果的字典。"""
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(d: dict) -> 'FactorTask':
        """从字典创建 FactorTask 实例。"""
        return FactorTask(**d)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    因子反馈工作空间（Factor Feedback Workspace）。
    用于通过将代码写入文件来实现因子。输入数据和输出的因子值也通过文件进行读写。
    """
    FB_EXEC_SUCCESS = "执行成功，无错误。"
    FB_CODE_NOT_SET = "代码未设置。"
    FB_OUTPUT_FILE_NOT_FOUND = "\n未找到预期的输出文件。"
    FB_OUTPUT_FILE_FOUND = "\n找到预期的输出文件。"

    def __init__(self, *args, raise_exception: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str | None:
        """为缓存生成哈希值。"""
        if "factor.py" in self.file_dict and not self.raise_exception:
            return md5_hash(data_type + self.file_dict["factor.py"])
        return None

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame | None]:
        """
        执行因子代码并获取因子值。步骤如下：
        1. 在工作空间路径下创建目录。
        2. 将代码写入文件。
        3. 将所有源数据链接到工作空间目录。
        4. 根据版本执行代码或生成执行脚本。
        5. 从输出文件中读取因子值。
        返回执行反馈（字符串）和因子值（DataFrame）。
        """
        self.before_execute()
        if "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            return self.FB_CODE_NOT_SET, None

        with FileLock(self.workspace_path / "execution.lock"):
            # 根据任务版本和数据类型确定源数据路径
            if self.target_task.version == 1:
                source_data_path = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug if data_type == "Debug" else FACTOR_COSTEER_SETTINGS.data_folder)
            elif self.target_task.version == 2:
                source_data_path = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition

            source_data_path.mkdir(exist_ok=True, parents=True)
            self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)

            execution_feedback = self.FB_EXEC_SUCCESS
            execution_success = False

            # 根据版本确定执行脚本路径
            if self.target_task.version == 1:
                execution_code_path = self.workspace_path / "factor.py"
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            # 执行代码
            try:
                subprocess.check_output(
                    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                )
                execution_success = True
            except subprocess.CalledProcessError as e:
                import site
                execution_feedback = e.output.decode().replace(str(self.workspace_path.absolute()), r"/path/to").replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                if len(execution_feedback) > 2000:
                    execution_feedback = execution_feedback[:1000] + "....隐藏长错误信息...." + execution_feedback[-1000:]
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback) from e
            except subprocess.TimeoutExpired as e:
                execution_feedback += f"执行超时（超时设置为 {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} 秒）。"
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback) from e

            # 读取输出结果
            workspace_output_file_path = self.workspace_path / "result.h5"
            executed_factor_value_dataframe = None
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"读取HDF文件时出错: {e}"[:1000]
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs) -> 'FactorFBWorkspace':
        """从文件夹加载工作空间。"""
        path = Path(path)
        code_dict = {file_path.name: file_path.read_text() for file_path in path.iterdir() if file_path.suffix == ".py"}
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


# 类型别名
FactorExperiment = Experiment
FeatureExperiment = Experiment
