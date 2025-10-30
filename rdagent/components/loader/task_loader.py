import json
from pathlib import Path
from typing import Sequence

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.experiment import Loader, WsLoader


class FactorTaskLoader(Loader[FactorTask]):
    """
    因子任务加载器。
    继承自通用的 Loader 类，专门用于加载 `FactorTask` 类型的任务。
    """
    pass


class ModelTaskLoader(Loader[ModelTask]):
    """
    模型任务加载器。
    继承自通用的 Loader 类，专门用于加载 `ModelTask` 类型的任务。
    """
    pass


class ModelTaskLoaderJson(ModelTaskLoader):
    """
    从 JSON 文件加载模型任务的具体实现。
    """
    def __init__(self, json_uri: str) -> None:
        """
        初始化加载器。

        Args:
            json_uri (str): 包含模型任务定义的 JSON 文件的路径。
        """
        super().__init__()
        self.json_uri = json_uri

    def load(self, *argT, **kwargs) -> Sequence[ModelTask]:
        """
        从 JSON 文件加载一个或多个模型任务。

        JSON 文件格式应为 {model_name: {model_data}}。

        Returns:
            Sequence[ModelTask]: 加载的模型任务对象列表。
        """
        # 从 JSON 文件加载模型字典
        with open(self.json_uri, "r") as f:
            model_dict = json.load(f)

        # FIXME: 由于提取错误，json 文件中的模型信息可能不正确。
        #        未来需要逐个修复这些问题。

        model_impl_task_list = []
        # 遍历字典中的每个模型，创建 ModelTask 对象
        for model_name, model_data in model_dict.items():
            model_impl_task = ModelTask(
                name=model_name,
                description=model_data["description"],
                formulation=model_data["formulation"],
                variables=model_data["variables"],
                model_type=model_data["model_type"],
                architecture="",  # 默认为空字符串
                hyperparameters="",  # 默认为空字符串
            )
            model_impl_task_list.append(model_impl_task)
        return model_impl_task_list


class ModelWsLoader(WsLoader[ModelTask, ModelFBWorkspace]):
    """
    模型工作空间加载器。
    用于从给定路径加载与特定模型任务相关的代码，并注入到工作空间中。
    """
    def __init__(self, path: Path) -> None:
        """
        初始化加载器。

        Args:
            path (Path): 存放模型代码文件的目录路径。
        """
        self.path = Path(path)

    def load(self, task: ModelTask) -> ModelFBWorkspace:
        """
        为给定的模型任务加载代码并准备工作空间。

        Args:
            task (ModelTask): 目标模型任务。

        Returns:
            ModelFBWorkspace: 准备好并注入了代码的工作空间。
        """
        assert task.name is not None, "任务名称不能为空"
        # 创建与任务关联的工作空间
        mti = ModelFBWorkspace(task)
        # 准备工作空间（例如，创建目录结构）
        mti.prepare()
        # 读取与任务同名的 .py 文件
        code_file = self.path / f"{task.name}.py"
        with open(code_file, "r") as f:
            code = f.read()
        # 将读取的代码注入到工作空间的 'model.py' 文件中
        mti.inject_files(**{"model.py": code})
        return mti
