from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCorrelationEvaluator,
    FactorEqualValueRatioEvaluator,
    FactorEvaluator,
    FactorIndexEvaluator,
    FactorRowCountEvaluator,
    FactorSingleColumnEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError
from rdagent.core.experiment import Experiment, Task, Workspace
from rdagent.core.scenario import Scenario
from rdagent.core.utils import multiprocessing_wrapper

# 定义评估结果的类型别名
EVAL_RES = Dict[
    str,
    List[Tuple[FactorEvaluator, Union[object, CoderError]]],
]


class TestCase:
    """测试用例类"""
    def __init__(
        self,
        target_task: Task,
        ground_truth: Workspace,
    ):
        self.target_task = target_task
        self.ground_truth = ground_truth


class TestCases:
    """测试用例集合类"""
    def __init__(self, test_case_l: list[TestCase] = []):
        self.test_case_l = test_case_l

    def __getitem__(self, item):
        return self.test_case_l[item]

    def __len__(self):
        return len(self.test_case_l)

    def get_exp(self):
        """获取实验对象"""
        return Experiment([case.target_task for case in self.test_case_l])

    @property
    def target_task(self):
        """获取所有目标任务"""
        return [case.target_task for case in self.test_case_l]

    @property
    def ground_truth(self):
        """获取所有基准真相"""
        return [case.ground_truth for case in self.test_case_l]


class BaseEval:
    """
    基准测试评估的基类。
    """

    def __init__(
        self,
        evaluator_l: List[FactorEvaluator],
        test_cases: TestCases,
        generate_method: Developer,
        catch_eval_except: bool = True,
    ):
        """参数
        ----------
        test_cases : TestCases
            待评估的用例，基准真相包含在测试用例中。
        evaluator_l : List[FactorEvaluator]
            用于评估生成代码的评估器列表。
        catch_eval_except : bool
            如果想调试评估器，建议将此参数设置为 True。
        """
        self.evaluator_l = evaluator_l
        self.test_cases = test_cases
        self.generate_method = generate_method
        self.catch_eval_except = catch_eval_except

    def load_cases_to_eval(
        self,
        path: Union[Path, str],
        **kwargs,
    ) -> List[Workspace]:
        """加载待评估的用例"""
        path = Path(path)
        fi_l = []
        for tc in self.test_cases:
            try:
                fi = FactorFBWorkspace.from_folder(tc.task, path, **kwargs)
                fi_l.append(fi)
            except FileNotFoundError:
                print("加载因子测试用例失败: ", tc.task.factor_name)
        return fi_l

    def eval_case(
        self,
        case_gt: Workspace,
        case_gen: Workspace,
    ) -> List[Union[Tuple[FactorEvaluator, object], Exception]]:
        """参数
        ----------
        case_gt : FactorImplementation
            基准真信用卡实现
        case_gen : FactorImplementation
            生成的实现

        返回
        -------
        List[Union[Tuple[FactorEvaluator, object],Exception]]
            对于每个项目
                如果评估成功运行，则返回评估结果。否则，返回异常。
        """
        eval_res = []
        for ev in self.evaluator_l:
            try:
                case_gen.raise_exception = True
                eval_res.append((ev, ev.evaluate(implementation=case_gen, gt_implementation=case_gt)))
            except CoderError as e:
                return e
            except Exception as e:
                # 评估时发生异常
                if self.catch_eval_except:
                    eval_res.append((ev, e))
                else:
                    raise e
        return eval_res


class FactorImplementEval(BaseEval):
    """因子实现评估类"""
    def __init__(
        self,
        test_cases: TestCases,
        method: Developer,
        *args,
        scen: Scenario,
        test_round: int = 10,
        **kwargs,
    ):
        online_evaluator_l = [
            FactorSingleColumnEvaluator(scen),
            FactorRowCountEvaluator(scen),
            FactorIndexEvaluator(scen),
            FactorEqualValueRatioEvaluator(scen),
            FactorCorrelationEvaluator(hard_check=False, scen=scen),
        ]
        super().__init__(online_evaluator_l, test_cases, method, *args, **kwargs)
        self.test_round = test_round

    def develop(self):
        """开发因子实现"""
        gen_factor_l_all_rounds = []
        for _ in tqdm(range(self.test_round), desc="评估回合"):
            print("\n========================================================")
            print(f"第 {_} 次评估...")
            print("========================================================\n")
            try:
                gen_factor_l = self.generate_method.develop(self.test_cases.get_exp())
            except KeyboardInterrupt:
                print("手动中断评估。保存现有结果")
                break

            if len(gen_factor_l.sub_workspace_list) != len(self.test_cases.ground_truth):
                raise ValueError(
                    "待评估的用例数应等于测试用例数。",
                )
            gen_factor_l_all_rounds.extend(gen_factor_l.sub_workspace_list)

        return gen_factor_l_all_rounds

    def eval(self, gen_factor_l_all_rounds):
        """评估生成的因子实现"""
        test_cases_all_rounds = []
        res = defaultdict(list)
        for _ in range(self.test_round):
            test_cases_all_rounds.extend(self.test_cases.ground_truth)
        eval_res_list = multiprocessing_wrapper(
            [
                (self.eval_case, (gt_case, gen_factor))
                for gt_case, gen_factor in zip(test_cases_all_rounds, gen_factor_l_all_rounds)
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for gt_case, eval_res, gen_factor in tqdm(zip(test_cases_all_rounds, eval_res_list, gen_factor_l_all_rounds)):
            res[gt_case.target_task.factor_name].append((gen_factor, eval_res))

        return res

    @staticmethod
    def summarize_res(res: EVAL_RES) -> pd.DataFrame:
        """总结评估结果"""
        # None: 表示引发异常并且没有得到结果
        sum_res = {}
        for factor_name, runs in res.items():
            for fi, err_or_res_l in runs:
                uniq_key = f"{str(fi)},{id(fi)}"

                key = (factor_name, uniq_key)
                val = {}
                if isinstance(err_or_res_l, Exception):
                    val["运行因子错误"] = str(err_or_res_l.__class__)
                else:
                    val["运行因子错误"] = None
                    for ev_obj, err_or_res in err_or_res_l:
                        if isinstance(err_or_res, Exception):
                            val[str(ev_obj)] = None
                        else:
                            feedback, metric = err_or_res
                            val[str(ev_obj)] = metric
                sum_res[key] = val

        return pd.DataFrame(sum_res)
