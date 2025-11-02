# 导入 pandas 用于数据处理
import pandas as pd

# 导入 CachedRunner 基类，用于缓存实验结果
from rdagent.components.runner import CachedRunner
# 导入 RD_AGENT 的配置
from rdagent.core.conf import RD_AGENT_SETTINGS
# 导入自定义异常类
from rdagent.core.exception import ModelEmptyError
# 导入缓存工具
from rdagent.core.utils import cache_with_pickle
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入 qlib 场景下的工具函数
from rdagent.scenarios.qlib.developer.utils import process_factor_data
# 导入因子实验类
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
# 导入模型实验类
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


class QlibModelRunner(CachedRunner[QlibModelExperiment]):
    """
    模型运行器，负责在 Docker 环境中执行模型实验。
    实验的所有内容（配置、代码、结果）都组织在一个文件夹中，包括：
    - config.yaml: 实验配置文件
    - Pytorch `model.py`: PyTorch 模型定义文件
    - results in `mlflow`: 实验结果记录在 mlflow 中

    参考 Qlib 的模型实现: https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri: 在配置文件中硬编码为 `model.py:Net`
    - 允许大语言模型修改 `model.py` 文件
    """

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        """
        执行模型开发实验。
        该方法会处理因子数据，配置模型参数，并在 Docker 中运行模型训练和回测。
        使用 cache_with_pickle 装饰器缓存实验结果，避免重复计算。
        :param exp: Qlib 模型实验对象。
        :return: 完成的 Qlib 模型实验对象。
        """
        # 如果存在基线实验且未执行，则先递归执行基线实验
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            # 过滤出 QlibFactorExperiment 实例作为 SOTA 因子来源
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"开始处理 SOTA 因子...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                # 对合并后的因子进行排序和格式化
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns))

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # 将合并后的因子保存到工作区
                combined_factors.to_parquet(target_path, engine="pyarrow")

        # 检查模型文件是否存在
        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py 为空")
        # 注入模型代码到实验工作区
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        # 获取并设置训练超参数
        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        if training_hyperparameters:
            env_to_use.update(
                {
                    "n_epochs": str(training_hyperparameters.get("n_epochs", "100")),
                    "lr": str(training_hyperparameters.get("lr", "2e-4")),
                    "early_stop": str(training_hyperparameters.get("early_stop", 10)),
                    "batch_size": str(training_hyperparameters.get("batch_size", 256)),
                    "weight_decay": str(training_hyperparameters.get("weight_decay", 0.0001)),
                }
            )

        logger.info(f"开始运行 {exp.sub_tasks[0].name} 模型")
        # 根据模型类型（时间序列或表格）和是否存在 SOTA 因子，选择不同的配置和参数执行实验
        if exp.sub_tasks[0].model_type == "TimeSeries":
            if exist_sota_factor_exp:
                env_to_use.update(
                    {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                )
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )
        elif exp.sub_tasks[0].model_type == "Tabular":
            if exist_sota_factor_exp:
                env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "DatasetH"})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )

        # 保存实验结果和标准输出
        exp.result = result
        exp.stdout = stdout

        # 如果实验失败，记录错误并抛出异常
        if result is None:
            logger.error(f"运行 {exp.sub_tasks[0].name} 失败，原因: {stdout}")
            raise ModelEmptyError(f"运行 {exp.sub_tasks[0].name} 模型失败，原因: {stdout}")

        return exp
