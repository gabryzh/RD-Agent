# 导入Path类，用于处理文件系统路径
from pathlib import Path

# 导入yaml库，用于解析YAML文件
import yaml

# 从rdagent.core.utils模块导入SingletonBaseClass
from rdagent.core.utils import SingletonBaseClass


class Prompts(SingletonBaseClass, dict[str, str]):
    """
    一个单例类，用于加载和存储提示。
    继承自dict，使其可以像字典一样使用。
    """
    def __init__(self, file_path: Path) -> None:
        """
        初始化Prompts类。
        :param file_path: 包含提示的YAML文件的路径。
        """
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            # 从YAML文件加载提示
            prompt_yaml_dict = yaml.safe_load(file)

        if prompt_yaml_dict is None:
            # 如果加载失败，则引发ValueError
            error_message = f"从 {file_path} 加载提示失败"
            raise ValueError(error_message)

        # 将加载的提示存储到类实例中
        for key, value in prompt_yaml_dict.items():
            self[key] = value
