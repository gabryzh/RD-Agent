import json
import pickle
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.benchmark.eval_method import FactorImplementEval


class BenchmarkAnalyzer:
    """基准分析器，用于分析评估结果。"""
    def __init__(self, settings, only_correct_format=False):
        """
        初始化 BenchmarkAnalyzer。

        Args:
            settings: BenchmarkSettings 对象，包含基准测试的设置。
            only_correct_format (bool): 是否只分析格式正确的结果。
        """
        self.settings = settings
        self.index_map = self.load_index_map()
        self.only_correct_format = only_correct_format

    def load_index_map(self):
        """
        从基准数据文件加载索引映射。
        映射关系为：因子名称 -> (因子名称, 类别, 难度)。
        """
        index_map = {}
        # 打开并读取基准数据路径下的JSON文件
        with open(self.settings.bench_data_path, "r") as file:
            factor_dict = json.load(file)
        # 遍历因子字典，构建索引映射
        for factor_name, data in factor_dict.items():
            index_map[factor_name] = (factor_name, data["Category"], data["Difficulty"])
        return index_map

    def load_data(self, file_path):
        """
        从给定的 pickle 文件加载数据。

        Args:
            file_path (str or Path): pickle 文件的路径。

        Returns:
            any: 从 pickle 文件中加载的数据。
        """
        file_path = Path(file_path)
        # 检查路径是否为有效的 pickle 文件
        if not (file_path.is_file() and file_path.suffix == ".pkl"):
            raise ValueError("无效的文件路径")

        # 以二进制读取模式打开文件并加载数据
        with file_path.open("rb") as f:
            res = pickle.load(f)

        return res

    def process_results(self, results):
        """
        处理多个实验的结果。

        Args:
            results (dict): 包含实验名称和结果文件路径的字典。

        Returns:
            dict: 每个实验的最终分析结果。
        """
        final_res = {}
        # 遍历所有实验结果
        for experiment, path in results.items():
            # 加载原始数据
            data = self.load_data(path)
            # 汇总评估结果
            summarized_data = FactorImplementEval.summarize_res(data)
            # 分析汇总后的数据
            processed_data = self.analyze_data(summarized_data)
            # 存储每个实验的最终结果（通常是平均值行）
            final_res[experiment] = processed_data.iloc[-1, :]
        return final_res

    def reformat_index(self, display_df):
        """
        重构 DataFrame 的索引格式，以支持多级索引（因子, 类别, 难度）。
        """
        new_idx = []
        # 筛选出索引在 index_map 中的行
        display_df = display_df[display_df.index.isin(self.index_map.keys())]
        # 根据 index_map 创建新的多级索引
        for idx in display_df.index:
            new_idx.append(self.index_map[idx])

        display_df.index = pd.MultiIndex.from_tuples(
            new_idx,
            names=["Factor", "Category", "Difficulty"],
        )
        # 调整索引层级顺序并排序
        display_df = display_df.swaplevel(0, 2).swaplevel(0, 1).sort_index(axis=0)

        # 根据难度级别进行排序
        return display_df.sort_index(
            key=lambda x: [{"Easy": 0, "Medium": 1, "Hard": 2, "New Discovery": 3}.get(i, i) for i in x]
        )

    def result_all_key_order(self, x):
        """
        为结果的列定义一个自定义的排序顺序。
        """
        order_v = []
        # 定义排序映射
        order_map = {
            "Avg Run SR": 0,
            "Avg Format SR": 1,
            "Avg Correlation": 2,
            "Max Correlation": 3,
            "Max Accuracy": 4,
            "Avg Accuracy": 5,
        }
        # 根据映射转换列名
        for i in x:
            order_v.append(order_map.get(i, i))
        return order_v

    def analyze_data(self, sum_df):
        """
        对汇总的评估数据进行详细分析。
        """
        # 定义需要分析的评估器索引
        index = [
            "FactorSingleColumnEvaluator",
            "FactorRowCountEvaluator",
            "FactorIndexEvaluator",
            "FactorEqualValueRatioEvaluator",
            "FactorCorrelationEvaluator",
            "run factor error",
        ]
        # 重建索引以确保所有评估器都存在
        sum_df = sum_df.reindex(index, axis=0)
        # 清理数据，按因子名称分组
        sum_df_clean = sum_df.T.groupby(level=0).apply(lambda x: x.reset_index(drop=True))

        # 计算运行成功率
        run_error = sum_df_clean["run factor error"].unstack().T.fillna(False).astype(bool)
        succ_rate = ~run_error
        succ_rate = succ_rate.mean(axis=0).to_frame("success rate")
        succ_rate_f = self.reformat_index(succ_rate)

        # 计算格式正确率
        format_issue = sum_df_clean[["FactorRowCountEvaluator", "FactorIndexEvaluator"]].apply(
            lambda x: np.mean(x.fillna(0.0)), axis=1
        )
        format_succ_rate = format_issue.unstack().T.mean(axis=0).to_frame("success rate")
        format_succ_rate_f = self.reformat_index(format_succ_rate)

        # 计算相关性
        corr = sum_df_clean["FactorCorrelationEvaluator"].fillna(0.0)
        # 如果设置了只分析格式正确的结果，则进行筛选
        if self.only_correct_format:
            corr = corr.loc[format_issue == 1.0]

        # 平均相关性
        corr_res = corr.unstack().T.mean(axis=0).to_frame("corr(only success)")
        corr_res = self.reformat_index(corr_res)

        # 最大相关性
        corr_max = corr.unstack().T.max(axis=0).to_frame("corr(only success)")
        corr_max_res = self.reformat_index(corr_max)

        # 最大值准确率
        value_max = sum_df_clean["FactorEqualValueRatioEvaluator"]
        value_max = value_max.unstack().T.max(axis=0).to_frame("max_value")
        value_max_res = self.reformat_index(value_max)

        # 平均值准确率（考虑格式正确性）
        value_avg = (
            (sum_df_clean["FactorEqualValueRatioEvaluator"] * format_issue)
            .unstack()
            .T.mean(axis=0)
            .to_frame("avg_value")
        )
        value_avg_res = self.reformat_index(value_avg)

        # 将所有结果合并到一个 DataFrame 中
        result_all = pd.concat(
            {
                "Avg Correlation": corr_res.iloc[:, 0],
                "Avg Format SR": format_succ_rate_f.iloc[:, 0],
                "Avg Run SR": succ_rate_f.iloc[:, 0],
                "Max Correlation": corr_max_res.iloc[:, 0],
                "Max Accuracy": value_max_res.iloc[:, 0],
                "Avg Accuracy": value_avg_res.iloc[:, 0],
            },
            axis=1,
        )

        # 对列和行进行排序
        df = result_all.sort_index(axis=1, key=self.result_all_key_order).sort_index(axis=0)
        print(df)

        print("\n按类别分组的平均值:")
        print(df.groupby("Category").mean())

        print("\n总体平均值:")
        print(df.mean())

        # 计算总体平均值并添加到 DataFrame 的末尾
        mean_values = df.fillna(0.0).mean()
        mean_df = pd.DataFrame(mean_values).T
        mean_df.index = pd.MultiIndex.from_tuples([("-", "-", "Average")], names=["Factor", "Category", "Difficulty"])
        df_w_mean = pd.concat([df, mean_df]).astype("float")

        return df_w_mean


class Plotter:
    """用于绘制分析结果的绘图器。"""
    @staticmethod
    def change_fs(font_size):
        """更改全局字体大小。"""
        plt.rc("font", size=font_size)
        plt.rc("axes", titlesize=font_size)
        plt.rc("axes", labelsize=font_size)
        plt.rc("xtick", labelsize=font_size)
        plt.rc("ytick", labelsize=font_size)
        plt.rc("legend", fontsize=font_size)
        plt.rc("figure", titlesize=font_size)

    @staticmethod
    def plot_data(data, file_name, title):
        """
        使用条形图绘制数据。

        Args:
            data (pd.DataFrame): 包含 'a' (x轴) 和 'b' (y轴) 列的数据。
            file_name (str): 保存图像的文件名。
            title (str): 图像的标题。
        """
        plt.figure(figsize=(10, 10))
        plt.ylabel("值")
        colors = ["#3274A1", "#E1812C", "#3A923A", "#C03D3E"]
        plt.bar(data["a"], data["b"], color=colors, capsize=5)
        # 在每个条形图上方显示数值
        for idx, row in data.iterrows():
            plt.text(idx, row["b"] + 0.01, f"{row['b']:.2f}", ha="center", va="bottom")
        plt.suptitle(title, y=0.98)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(file_name)


def main(
    path="git_ignore_folder/eval_results/res_promptV220240724-060037.pkl",
    round=1,
    title="不同方法的比较",
    only_correct_format=False,
):
    """
    主函数，用于运行基准分析和绘图。

    Args:
        path (str): 评估结果 pickle 文件的路径。
        round (int): 实验轮次。
        title (str): 图像的标题。
        only_correct_format (bool): 是否只分析格式正确的结果。
    """
    settings = BenchmarkSettings()
    benchmark = BenchmarkAnalyzer(settings, only_correct_format=only_correct_format)
    # 定义要处理的实验结果
    results = {
        f"第 {round} 轮实验": path,
    }
    # 处理结果
    final_results = benchmark.process_results(results)
    final_results_df = pd.DataFrame(final_results)

    # 设置绘图字体大小
    Plotter.change_fs(20)
    # 准备绘图数据
    plot_data = final_results_df.drop(["Max Accuracy", "Avg Accuracy"], axis=0).T
    plot_data = plot_data.reset_index().melt("index", var_name="a", value_name="b")
    # 绘制并保存图像
    Plotter.plot_data(plot_data, "./comparison_plot.png", title)


if __name__ == "__main__":
    # 使用 fire 库将 main 函数暴露为命令行接口
    fire.Fire(main)
