# 导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 导入 pandas 用于数据处理
import pandas as pd
# 导入 pandarallel 用于并行处理 pandas 操作
from pandarallel import pandarallel

# 导入 RD_AGENT 的配置
from rdagent.core.conf import RD_AGENT_SETTINGS
# 导入缓存工具
from rdagent.core.utils import cache_with_pickle

# 初始化 pandarallel，verbose=1 表示显示进度条
pandarallel.initialize(verbose=1)

# 导入 CachedRunner 基类，用于缓存实验结果
from rdagent.components.runner import CachedRunner
# 导入自定义异常类
from rdagent.core.exception import FactorEmptyError
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入 qlib 场景下的工具函数
from rdagent.scenarios.qlib.developer.utils import process_factor_data
# 导入因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
# 导入模型实验类
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

# 获取当前文件所在的目录
DIRNAME = Path(__file__).absolute().resolve().parent
# 获取当前工作目录
DIRNAME_local = Path.cwd()

# class QlibFactorExpWorkspace:
#
#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf_baseline.yaml")

# TODO: 支持多进程并保留先前结果


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    因子运行器，负责在 Docker 环境中执行因子实验。
    实验的所有内容（配置、数据、代码）都组织在一个文件夹中，包括：
    - config.yaml: 实验配置文件
    - price-volume data dumper: 价格和交易量数据导出器
    - `data.py` + Adaptor to Factor implementation: 因子实现的数据适配代码
    - results in `mlflow`: 实验结果记录在 mlflow 中
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        """
        计算两组因子（SOTA 因子和新因子）之间的信息系数（IC），即相关系数。
        :param concat_feature: 包含了 SOTA 因子和新因子的 DataFrame。
        :param SOTA_feature_column_size: SOTA 因子的数量。
        :param new_feature_columns_size: 新因子的数量。
        :return: 包含所有 IC 计算结果的 pd.Series。
        """
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        # 遍历 SOTA 因子和新因子，计算它们两两之间的相关性
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        """
        对新因子进行去重，移除与现有 SOTA（State-of-the-art）因子高度相关的因子。
        :param SOTA_feature: SOTA 因子的 DataFrame。
        :param new_feature: 新因子的 DataFrame。
        :return: 去重后的新因子 DataFrame。
        """
        # 计算每对 SOTA 因子和新因子之间的 IC
        # 如果 IC 大于某个阈值，则移除该新因子列
        # 返回去重后的新因子

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        # 按 'datetime' 分组，并行计算 IC 的均值
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        # 将索引设置为 SOTA 因子和新因子的多重索引
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        # 找到每个新因子与所有 SOTA 因子 IC 的最大值
        IC_max = IC_max.unstack().max(axis=0)
        # 返回 IC 最大值小于 0.99 的新因子
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        执行因子开发实验。
        该方法会处理并合并因子数据，然后将合并后的数据传递给 Docker 运行回测。
        使用 cache_with_pickle 装饰器缓存实验结果，避免重复计算。
        :param exp: Qlib 因子实验对象。
        :return: 完成的 Qlib 因子实验对象。
        """
        # 如果存在基线实验且未执行，则先递归执行基线实验
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info(f"开始执行基线实验...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        if exp.based_experiments:
            SOTA_factor = None
            # 过滤出 QlibFactorExperiment 实例作为 SOTA 因子来源
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"开始处理 SOTA 因子...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            logger.info(f"开始处理新因子...")
            # 处理新的因子数据
            new_factors = process_factor_data(exp)

            if new_factors.empty:
                raise FactorEmptyError("因子在全样本上运行失败，本轮实验失败。")

            # 如果 SOTA 因子存在，则合并 SOTA 因子和新因子
            if SOTA_factor is not None and not SOTA_factor.empty:
                # 对新因子进行去重
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "本轮生成的因子与之前的因子高度相似。请更换生成新因子的方向。"
                    )
                # 合并去重后的新因子和 SOTA 因子
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # 对合并后的因子进行排序和格式化
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            num_features = RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns)
            logger.info(f"因子数据处理完成。")

            # 由于 rdagent 和 qlib docker 镜像中 numpy 版本存在差异，
            # `combined_factors_df.pkl` 文件无法在 qlib docker 中正确加载，
            # 因此我们将 `combined_factors_df` 的文件类型从 pkl 更改为 parquet。
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # 将合并后的因子保存到工作区
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # 检查之前的实验中是否存在模型实验
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info(f"开始执行实验...")
            if exist_sota_model_exp:
                # 如果存在 SOTA 模型实验，则使用该模型进行回测
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]}
                )
                env_to_use = {"PYTHONPATH": "./"}
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                if sota_training_hyperparameters:
                    # 设置模型训练的超参数
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        }
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                if sota_model_type == "TimeSeries":
                    # 设置时间序列模型相关参数
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                    )
                elif sota_model_type == "Tabular":
                    # 设置表格模型相关参数
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # 执行 SOTA 模型 + 合并因子的实验
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use
                )
            else:
                # 如果不存在 SOTA 模型，则使用默认的 LGBM 模型
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name=(
                        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                    )
                )
        else:
            # 如果没有基线实验，直接执行
            logger.info(f"开始执行实验...")
            result, stdout = exp.experiment_workspace.execute(
                qlib_config_name=(
                    f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                )
            )

        if result is None:
            # 如果实验失败，记录错误并抛出异常
            logger.error(f"实验运行失败，原因: {stdout}")
            raise FactorEmptyError(f"实验运行失败，原因: {stdout}")

        # 保存实验结果和标准输出
        exp.result = result
        exp.stdout = stdout

        return exp
