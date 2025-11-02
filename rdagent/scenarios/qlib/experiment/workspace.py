# 导入 re 模块，用于正则表达式操作
import re
# 导入 Path 类，用于处理文件系统路径
from pathlib import Path
# 导入 Any 类型提示
from typing import Any

# 导入 pandas 用于数据处理
import pandas as pd

# 导入模型编码器的配置
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
# 导入实验工作区基类
from rdagent.core.experiment import FBWorkspace
# 导入日志记录器
from rdagent.log import rdagent_logger as logger
# 导入 Qlib 运行环境相关的类
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    """
    Qlib 实验工作区类。
    负责管理实验所需的文件、环境，并执行实验。
    """
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 从指定的模板文件夹中注入代码和配置文件
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        """
        执行 Qlib 实验。

        :param qlib_config_name: 要使用的 Qlib 配置文件名。
        :param run_env: 实验运行时需要的环境变量。
        :return: 一个元组，包含实验结果 (pd.Series) 和标准输出 (str)。如果失败，则返回 (None, error_message)。
        """
        # 根据配置选择使用 Docker 还是 Conda 环境
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"未知的环境类型: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "未知的环境类型"
        # 准备环境 (例如，构建 Docker 镜像)
        qtde.prepare()

        # 运行 Qlib 回测
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",  # qrun 是 qlib 提供的命令行工具
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        # 运行脚本以读取和解析实验结果
        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        # 检查并记录回测图表结果
        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="量化回测图表")
        else:
            logger.error("未找到结果文件。")
            return None, execute_qlib_log

        # 检查并解析包含主要指标的结果文件
        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # 在这里，我们确保 qlib 实验已成功运行，然后使用正则表达式从 execute_qlib_log 中提取信息；
            # 否则，我们保留原始的实验标准输出。
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"文件 {qlib_res_path} 不存在。")
            return None, execute_qlib_log
