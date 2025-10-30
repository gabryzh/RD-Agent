from rdagent.core.experiment import ASpecificExp
from rdagent.core.interactor import Interactor
from rdagent.core.proposal import Trace


class SkipInteractor(Interactor[ASpecificExp]):
    """
    一个跳过实际交互的交互器实现。
    它直接返回传入的实验，不进行任何修改。
    """

    def interact(self, exp: ASpecificExp, trace: Trace) -> ASpecificExp:
        """
        与用户交互以获取反馈或确认。

        职责:
        - 向用户呈现实验的当前状态。
        - 收集用户输入以指导实验的下一步。
        - 根据用户反馈重写实验。

        在此实现中，该方法仅直接返回实验对象，模拟了跳过用户交互的场景。

        Args:
            exp (ASpecificExp): 当前的实验对象。
            trace (Trace): 包含历史信息的追踪对象。

        Returns:
            ASpecificExp: 未经修改的原始实验对象。
        """
        return exp
