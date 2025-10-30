from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
from rdagent.components.coder.model_coder.evolving_strategy import (
    ModelMultiProcessEvolvingStrategy,
)
from rdagent.core.scenario import Scenario


class ModelCoSTEER(CoSTEER):
    """
    专门用于模型生成的 CoSTEER (Code evolution with STrEER) 框架实现。

    该类通过组合特定于模型的评估器和演进策略来定制通用的 CoSTEER 框架。
    """
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        """
        初始化 ModelCoSTEER。

        Args:
            scen (Scenario): 当前实验的场景对象，提供了上下文信息。
            *args: 传递给父类构造函数的额外位置参数。
            **kwargs: 传递给父类构造函数的额外关键字参数。
        """
        # 1. 初始化模型专用的评估器
        #    - `ModelCoSTEEREvaluator` 是执行具体评估逻辑的单元。
        #    - `CoSTEERMultiEvaluator` 是一个包装器，用于处理多个任务的并行评估。
        eva = CoSTEERMultiEvaluator(ModelCoSTEEREvaluator(scen=scen), scen=scen)

        # 2. 初始化模型专用的演进策略
        #    - `ModelMultiProcessEvolvingStrategy` 定义了如何根据反馈迭代和优化代码，
        #      并使用多进程来加速。
        es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        # 3. 调用父类 `CoSTEER` 的构造函数
        #    - 传入创建好的评估器 (eva) 和演进策略 (es)。
        #    - `evolving_version=2` 可能表示使用了特定版本的演进框架。
        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
