from pathlib import Path

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry


class PrevModelLoadEvaluator(CoSTEEREvaluator):
    """此评估器检查代码是否实际从`prev_model`加载模型。"""

    def __init__(self, scen: Scenario):
        super().__init__(scen)

    def evaluate(
        self, target_task: Task, implementation: FBWorkspace, gt_implementation: FBWorkspace, *args, **kwargs
    ) -> CoSTEERSingleFeedback:
        """评估代码是否加载了先前训练的模型。"""
        data_source_path = T("scenarios.data_science.share:scen.input_path").r()
        prev_model_dir = Path(data_source_path) / "prev_model"

        # 1) 检查代码本身是否引用了prev_model的加载
        code_str = implementation.file_dict["main.py"]
        code_contain_prev = "prev_model" in code_str
        print(f"代码引用了prev_model: {code_contain_prev}")
        if not code_contain_prev:
            err = (
                "未找到代码从`prev_model`加载模型的证据。"
                "请检查您是否调用了正确的加载函数，"
                f"并将其指向`{prev_model_dir}`目录。"
            )
            return CoSTEERSingleFeedback(
                execution=err,
                return_checking=err,
                code=err,
                final_decision=False,
            )

        system_prompt = T(".prompts:prev_model_eval.system").r()
        user_prompt = T(".prompts:prev_model_eval.user").r(
            code=implementation.all_codes,
        )

        csfb = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return csfb
