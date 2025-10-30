# 导入所需的模块和类
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.benchmark.eval_method import FactorImplementEval
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorTestCaseLoaderFromJsonFile,
)

if __name__ == "__main__":
    # 步骤 1: 读取基准测试设置
    # 从配置文件中加载基准测试的相关参数，例如测试轮次、要测试的方法类等。
    bs = BenchmarkSettings()

    # 步骤 2: 读取并准备评估数据
    # 从指定的JSON文件中加载因子测试用例。
    test_cases = FactorTestCaseLoaderFromJsonFile().load(bs.bench_data_path)

    # 步骤 3: 动态导入并实例化要测试的方法类
    # 根据配置文件中的场景类路径，动态导入场景类。
    scen: Scenario = import_class(FACTOR_PROP_SETTING.scen)()
    # 根据配置文件中的方法类路径和额外参数，动态实例化要进行基准测试的方法。
    generate_method = import_class(bs.bench_method_cls)(scen=scen, **bs.bench_method_extra_kwargs)

    # 步骤 4: 声明评估方法并配置参数
    # 实例化因子实现评估类，并传入以下参数：
    # - method: 要评估的因子生成方法。
    # - test_cases: 用于评估的测试用例。
    # - scen: 当前的实验场景。
    # - catch_eval_except: 是否捕获评估过程中的异常，以防止单个测试用例失败导致整个评估中断。
    # - test_round: 每个测试用例的测试轮次。
    eval_method = FactorImplementEval(
        method=generate_method,
        test_cases=test_cases,
        scen=scen,
        catch_eval_except=True,
        test_round=bs.bench_test_round,
    )

    # 步骤 5: 运行评估流程
    # 首先调用 develop 方法来生成代码实现，然后调用 eval 方法对生成的实现进行评估。
    res = eval_method.eval(eval_method.develop())

    # 步骤 6: 保存评估结果
    # 使用日志记录器将评估结果保存到文件中，便于后续分析。
    logger.log_object(res)
