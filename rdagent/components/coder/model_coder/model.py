import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.env import KGDockerEnv, QlibCondaConf, QlibCondaEnv, QTDockerEnv


class ModelTask(CoSTEERTask):
    """
    模型任务类，定义了生成一个模型所需的所有信息。
    """
    def __init__(
        self,
        name: str,
        description: str,
        architecture: str,
        *args,
        hyperparameters: Dict[str, str],
        training_hyperparameters: Dict[str, str],
        formulation: str = None,
        variables: Dict[str, str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        初始化模型任务。

        Args:
            name (str): 模型名称。
            description (str): 模型描述。
            architecture (str): 模型架构描述。
            hyperparameters (Dict[str, str]): 模型超参数。
            training_hyperparameters (Dict[str, str]): 训练超参数。
            formulation (str, optional): 模型的数学公式。
            variables (Dict[str, str], optional): 公式中的变量解释。
            model_type (Optional[str], optional): 模型类型（例如 'Graph', 'Tabular'）。
        """
        self.formulation: str = formulation
        self.architecture: str = architecture
        self.variables: str = variables
        self.hyperparameters: str = hyperparameters
        self.training_hyperparameters: str = training_hyperparameters
        self.model_type: str = (
            model_type  # 表格模型、时间序列模型、图模型、XGBoost模型等
        )
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self) -> str:
        """获取任务的详细信息字符串。"""
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        task_desc += f"formulation: {self.formulation}\n" if self.formulation else ""
        task_desc += f"architecture: {self.architecture}\n"
        task_desc += f"variables: {self.variables}\n" if self.variables else ""
        task_desc += f"hyperparameters: {self.hyperparameters}\n"
        task_desc += f"training_hyperparameters: {self.training_hyperparameters}\n"
        task_desc += f"model_type: {self.model_type}\n"
        return task_desc

    def get_task_brief_information(self) -> str:
        """获取任务的简要信息字符串。"""
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        task_desc += f"architecture: {self.architecture}\n"
        task_desc += f"hyperparameters: {self.hyperparameters}\n"
        task_desc += f"training_hyperparameters: {self.training_hyperparameters}\n"
        task_desc += f"model_type: {self.model_type}\n"
        return task_desc

    @staticmethod
    def from_dict(d: dict) -> 'ModelTask':
        """从字典创建 ModelTask 实例。"""
        return ModelTask(**d)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class ModelFBWorkspace(FBWorkspace):
    """
    模型反馈工作空间（Model Feedback Workspace）。
    这是一个用于 PyTorch 模型实现任务的环境，所有相关文件都放在一个文件夹中。

    文件夹结构:
    - `prepare` 方法准备的数据源和文档。
      - 注意：新的数据可能会在 `execute` 方法中动态传入。
    - `inject_code` 方法注入的代码（文件 `model.py`）。
      - `model.py` 文件包含一个名为 `model_cls` 的变量，它是一个 `torch.nn.Module` 的实例，代表了模型的实现。

    支持两种接口:
    - (版本1) qlib: 创建一个脚本，在将当前工作目录设置到该文件夹后，导入 `model.py` 中的模型。
    - (版本2) kaggle: 创建一个脚本，调用 `model.py` 中的 fit 和 predict 函数。
    """

    def hash_func(
        self,
        batch_size: int = 8,
        num_features: int = 10,
        num_timesteps: int = 4,
        num_edges: int = 20,
        input_value: float = 1.0,
        param_init_value: float = 1.0,
    ) -> str:
        """
        为执行结果的缓存生成哈希值。
        哈希值基于执行参数和工作空间中的所有代码文件内容。
        """
        target_file_name = f"{batch_size}_{num_features}_{num_timesteps}_{input_value}_{param_init_value}"
        for code_file_name in sorted(list(self.file_dict.keys())):
            target_file_name = f"{target_file_name}_{self.file_dict[code_file_name]}"
        return md5_hash(target_file_name)

    @cache_with_pickle(hash_func)
    def execute(
        self,
        batch_size: int = 8,
        num_features: int = 10,
        num_timesteps: int = 4,
        num_edges: int = 20,
        input_value: float = 1.0,
        param_init_value: float = 1.0,
    ):
        """
        在隔离的环境中执行模型代码，并返回结果。
        使用 `@cache_with_pickle` 装饰器缓存执行结果。
        """
        self.before_execute()
        try:
            # 根据任务版本和配置选择合适的执行环境
            if self.target_task.version == 1:
                if MODEL_COSTEER_SETTINGS.env_type == "docker":
                    qtde = QTDockerEnv()
                elif MODEL_COSTEER_SETTINGS.env_type == "conda":
                    qtde = QlibCondaEnv(conf=QlibCondaConf())
                else:
                    raise ValueError(f"未知的 env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            else:
                qtde = KGDockerEnv()
            qtde.prepare()

            # 根据任务版本加载不同的执行模板
            if self.target_task.version == 1:
                dump_code = f"""
MODEL_TYPE = "{self.target_task.model_type}"
BATCH_SIZE = {batch_size}
NUM_FEATURES = {num_features}
NUM_TIMESTEPS = {num_timesteps}
NUM_EDGES = {num_edges}
INPUT_VALUE = {input_value}
PARAM_INIT_VALUE = {param_init_value}
{(Path(__file__).parent / 'model_execute_template_v1.txt').read_text()}
"""
            elif self.target_task.version == 2:
                dump_code = (Path(__file__).parent / "model_execute_template_v2.txt").read_text()

            # 在环境中转储并运行代码，然后获取结果
            log, results = qtde.dump_python_code_run_and_get_results(
                code=dump_code,
                dump_file_names=["execution_feedback_str.pkl", "execution_model_output.pkl"],
                local_path=str(self.workspace_path),
                env={},
                code_dump_file_py_name="model_test",
            )
            if len(results) == 0:
                raise RuntimeError(f"运行模型代码时出错: {log}")
            [execution_feedback_str, execution_model_output] = results

        except Exception as e:
            # 捕获异常并格式化错误信息
            execution_feedback_str = f"执行错误: {e}\nTraceback: {traceback.format_exc()}"
            execution_model_output = None

        # 如果错误信息过长，进行截断
        if len(execution_feedback_str) > 2000:
            execution_feedback_str = (
                execution_feedback_str[:1000] + "....隐藏了长错误信息...." + execution_feedback_str[-1000:]
            )
        return execution_feedback_str, execution_model_output


# 类型别名
ModelExperiment = Experiment
