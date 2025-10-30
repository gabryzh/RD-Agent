from dataclasses import field
from pathlib import Path
from typing import Optional

from rdagent.core.conf import ExtendedBaseSettings

# 定义当前目录
DIRNAME = Path("./")


class BenchmarkSettings(ExtendedBaseSettings):
    """基准测试设置类"""
    class Config:
        env_prefix = "BENCHMARK_"
        """使用 `BENCHMARK_` 作为环境变量的前缀"""

    bench_data_path: Path = DIRNAME / "example.json"
    """基准测试数据路径"""

    bench_test_round: int = 10
    """要运行的回合数，每回合可能花费10分钟"""

    bench_test_case_n: Optional[int] = None
    """要运行的测试用例数；如果未给出，将运行所有测试用例"""

    bench_method_cls: str = "rdagent.components.coder.factor_coder.FactorCoSTEER"
    """用于测试用例的方法"""

    bench_method_extra_kwargs: dict = field(
        default_factory=dict,
    )
    """要测试的方法的额外关键字参数（任务列表除外）"""

    bench_result_path: Path = DIRNAME / "result"
    """结果保存路径"""
