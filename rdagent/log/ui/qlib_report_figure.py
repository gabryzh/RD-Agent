import importlib
import math

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class BaseGraph:
    """基础图表类"""
    _name = None

    def __init__(
        self, df: pd.DataFrame = None, layout: dict = None, graph_kwargs: dict = None, name_dict: dict = None, **kwargs
    ):
        """
        :param df: 数据帧
        :param layout: `go.Layout` 参数
        :param graph_kwargs: 图表参数，例如: go.Bar(**graph_kwargs)
        :param name_dict: 名称字典
        :param kwargs: 其他关键字参数
        """
        self._df = df
        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
        self._name_dict = name_dict
        self.data = None

        self._init_parameters(**kwargs)
        self._init_data()

    def _init_data(self):
        """初始化数据"""
        if self._df.empty:
            raise ValueError("df 为空。")
        self.data = self._get_data()

    def _init_parameters(self, **kwargs):
        """初始化参数"""
        self._graph_type = self._name.lower().capitalize()
        if self._name_dict is None:
            self._name_dict = {_item: _item for _item in self._df.columns}

    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        """获取带有图表参数的实例"""
        try:
            _graph_module = importlib.import_module("plotly.graph_objs")
            _graph_class = getattr(_graph_module, graph_type)
        except AttributeError:
            _graph_module = importlib.import_module("qlib.contrib.report.graph")
            _graph_class = getattr(_graph_module, graph_type)
        return _graph_class(**kwargs)

    def _get_layout(self) -> go.Layout:
        """获取布局"""
        return go.Layout(**self._layout)

    def _get_data(self) -> list:
        """获取数据"""
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df.index, y=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data

    @property
    def figure(self) -> go.Figure:
        """获取图表 figure 对象"""
        _figure = go.Figure(data=self.data, layout=self._get_layout())
        _figure["layout"].update(template=None)
        return _figure


class SubplotsGraph:
    """创建与 df.plot(subplots=True) 相同的子图"""

    def __init__(
        self, df: pd.DataFrame = None, kind_map: dict = None, layout: dict = None,
        sub_graph_layout: dict = None, sub_graph_data: list = None, subplots_kwargs: dict = None, **kwargs,
    ):
        """
        :param df: pd.DataFrame
        :param kind_map: dict, 子图的图表类型和参数
        :param layout: `go.Layout` 参数
        :param sub_graph_layout: 每个子图的布局
        :param sub_graph_data: 每个子图的实例化参数
        :param subplots_kwargs: `plotly.tools.make_subplots` 的原始参数
        """
        # ... (初始化各种参数) ...
        self._init_figure()

    def _init_sub_graph_data(self):
        """初始化子图数据"""
        # ...

    def _init_subplots_kwargs(self):
        """初始化子图关键字参数"""
        # ...

    def _init_figure(self):
        """初始化 figure 对象"""
        self._figure = make_subplots(**self._subplots_kwargs)
        for column_name, column_map in self._sub_graph_data:
            # ... (根据参数创建并添加 trace 到 figure) ...
            pass
        # ... (更新布局) ...

    @property
    def figure(self):
        return self._figure


def _calculate_maximum(df: pd.DataFrame, is_ex: bool = False):
    """计算最大回撤的起始和结束日期"""
    # ...

def _calculate_mdd(series):
    """计算最大回撤序列"""
    return series - series.cummax()


def _calculate_report_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """计算报告所需的数据"""
    df = raw_df.copy(deep=True)
    # ... (计算累积收益、最大回撤、换手率等) ...
    return report_df


def report_figure(df: pd.DataFrame) -> list | tuple:
    """
    生成 Qlib 风格的回测报告图表。
    :param df: 包含 'bench', 'return', 'cost', 'turnover' 列的数据帧。
    :return: Plotly Figure 对象。
    """
    # 获取数据
    report_df = _calculate_report_data(df)

    # 计算最大回撤期间
    max_start_date, max_end_date = _calculate_maximum(report_df)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(report_df, True)

    # ... (准备数据和布局参数) ...

    # 创建 Figure
    figure = SubplotsGraph(
        df=report_df,
        layout=_layout_style,
        sub_graph_data=_column_row_col_dict,
        subplots_kwargs=_subplot_kwargs,
        kind_map=_default_kind_map,
        sub_graph_layout=_subplot_layout,
    ).figure
    return figure
