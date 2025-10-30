from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario


class FactorCoSTEER(CoSTEER):
    """
    专门用于因子生成的 CoSTEER (Code evolution with STrEER) 框架实现。
    """
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        """
        初始化 FactorCoSTEER。

        Args:
            scen (Scenario): 当前实验的场景对象。
        """
        # 加载因子专用的配置
        setting = FACTOR_COSTEER_SETTINGS

        # 1. 初始化因子专用的评估器
        #    - `FactorEvaluatorForCoder` 是执行具体评估逻辑的单元。
        #    - `CoSTEERMultiEvaluator` 是处理并行评估的包装器。
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)

        # 2. 初始化因子专用的演进策略
        #    - `FactorMultiProcessEvolvingStrategy` 定义了代码如何迭代优化。
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        # 3. 调用父类 `CoSTEER` 的构造函数
        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)

    def develop(self, exp: Experiment) -> Experiment:
        """
        重写 develop 方法，以在开发过程结束后附加最终的反馈信息。

        Args:
            exp (Experiment): 要进行开发的实验对象。

        Returns:
            Experiment: 开发完成并附带反馈的实验对象。
        """
        try:
            # 调用父类的 develop 方法执行核心的开发流程
            exp = super().develop(exp)
        finally:
            # 无论开发成功与否，都尝试记录最后一次演进的反馈
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                # 获取最后一次演进步骤的记录
                es = self.evolve_agent.evolving_trace[-1]
                # 将反馈附加到实验对象上
                exp.prop_dev_feedback = es.feedback
        return exp
