# 导入 pickle 用于序列化和反序列化 Python 对象
import pickle
# 导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 导入 pandas 用于数据处理
import pandas as pd
# 导入 qlib 库
import qlib
# 从 mlflow.entities 导入 ViewType，用于指定查询类型
from mlflow.entities import ViewType
# 从 mlflow.tracking 导入 MlflowClient，用于与 MLflow Tracking Server 交互
from mlflow.tracking import MlflowClient

# 初始化 qlib
qlib.init()

# 从 qlib.workflow 导入 R，R 是 Qlib Recorder 的高级 API
from qlib.workflow import R

# Qlib Recorder 文档: https://qlib.readthedocs.io/en/latest/component/recorder.html

# TODO: 列出所有的 recorder 和 metrics

# 假设已经列出了所有实验
experiments = R.list_experiments()

# 遍历每个实验以找到最新的 recorder
experiment_name = None
latest_recorder = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is not None:
            experiment_name = experiment
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            end_time = recorder.info["end_time"]
            try:
                # 检查 recorder 是否有有效的结束时间
                if end_time is not None:
                    # 如果是第一个 recorder 或者当前 recorder 的结束时间比已记录的最新时间还要晚
                    if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                        latest_recorder = recorder
                else:
                    print(f"警告: Recorder {recorder_id} 没有有效的结束时间")
            except Exception as e:
                print(f"错误: {e}")

# 检查是否找到了最新的 recorder
if latest_recorder is None:
    print("没有找到 recorders")
else:
    print(f"最新的 recorder: {latest_recorder}")

    # 从最新的 recorder 加载指定的 metrics 文件
    metrics = pd.Series(latest_recorder.list_metrics())

    # 定义输出路径
    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    # 将 metrics 保存到 CSV 文件
    metrics.to_csv(output_path)

    print(f"输出已保存到 {output_path}")

    # 加载投资组合分析报告
    ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    # 将报告保存为 pickle 文件
    ret_data_frame.to_pickle("ret.pkl")
