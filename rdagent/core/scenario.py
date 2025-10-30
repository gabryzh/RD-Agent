# 导入ABC（抽象基类）和abstractmethod（抽象方法）
from abc import ABC, abstractmethod

# 从rdagent.core.experiment模块导入Task
from rdagent.core.experiment import Task


class Scenario(ABC):
    """
    场景抽象基类。
    这里应该包含场景信息。不应包含以下信息：
    - 与方法相关的（例如，具体模块的rag配置）
    """

    @property
    @abstractmethod
    def background(self) -> str:
        """背景信息"""

    # TODO: 我们需要修改所有子类以覆盖get_source_data_desc而不是`source_data`
    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        """
        源数据描述。
        数据的选择可能因具体任务而异。
        """
        return ""

    @property
    def source_data(self) -> str:
        """
        描述源数据的便捷方式。
        """
        return self.get_source_data_desc()

    # 注意：我们应该保持接口更简单。因此删除了一些以前的接口。
    # 如果我们需要一些仅在子类中使用的特定函数（没有外部使用），
    # 我们不应该在基类中设置它们。

    @property
    @abstractmethod
    def rich_style_description(self) -> str:
        """用于呈现的富文本样式描述"""

    @abstractmethod
    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """
        将所有描述组合在一起。
        场景描述因执行的任务而异。
        """

    @abstractmethod
    def get_runtime_environment(self) -> str:
        """
        获取运行时环境信息。
        """

    @property
    def experiment_setting(self) -> str | None:
        """获取实验设置并以富文本字符串形式返回"""
        return None
