from rdagent.components.coder.factor_coder.factor import FactorExperiment
from rdagent.core.experiment import Loader


class FactorExperimentLoader(Loader[FactorExperiment]):
    """
    因子实验加载器。

    该类继承自通用的 Loader 类，并专门用于加载 `FactorExperiment` 类型的实验。
    具体的加载逻辑由父类 `Loader` 提供，这里仅作为类型提示和特定加载器的占位符。
    """
    pass


class ModelExperimentLoader(Loader[FactorExperiment]):
    """
    模型实验加载器。

    该类继承自通用的 Loader 类，目前也用于加载 `FactorExperiment` 类型的实验。
    未来可能会根据模型实验的具体结构进行调整。
    """
    pass
