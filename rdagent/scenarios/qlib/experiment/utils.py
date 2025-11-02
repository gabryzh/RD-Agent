# 导入 random 模块，用于生成随机数
import random
# 导入 re 模块，用于正则表达式操作
import re
# 导入 shutil 模块，用于文件操作
import shutil
# 导入 Path 类，用于处理文件系统路径
from pathlib import Path

# 导入 pandas 用于数据处理
import pandas as pd
# 导入 Jinja2 模板引擎
from jinja2 import Environment, StrictUndefined

# 导入因子编码器的配置
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
# 导入 Qlib Docker 环境类
from rdagent.utils.env import QTDockerEnv


def generate_data_folder_from_qlib():
    """
    使用 Qlib Docker 环境生成因子研究所需的数据文件。
    它会运行 `factor_data_template/generate.py` 脚本，
    并将生成的 HDF5 数据文件和 README.md 复制到配置中指定的数据目录。
    """
    template_path = Path(__file__).parent / "factor_data_template"
    qtde = QTDockerEnv()
    # 准备 Docker 环境
    qtde.prepare()

    # 在 Docker 环境中运行数据生成脚本
    execute_log = qtde.check_output(
        local_path=str(template_path),
        entry=f"python generate.py",
    )

    # 断言确保数据文件已成功生成
    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5").exists(), (
        "daily_pv_all.h5 未生成。这表示脚本 "
        "rdagent/scenarios/qlib/experiment/factor_data_template/generate.py "
        "未正确执行。请检查日志: \n" + execute_log
    )
    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5").exists(), (
        "daily_pv_debug.h5 未生成。这表示脚本 "
        "rdagent/scenarios/qlib/experiment/factor_data_template/generate.py "
        "未正确执行。请检查日志: \n" + execute_log
    )

    # 创建数据目录并将生成的文件复制过去
    Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md",
    )

    # 创建调试数据目录并将生成的文件复制过去
    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md",
    )


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    根据文件类型获取文件的描述。

    Parameters
    ----------
    p : Path
        文件路径。
    variable_list : list
        （可选）相关的变量列表，用于筛选 HDF5 文件中的列。

    Returns
    -------
    str
        文件的描述。
    """
    p = Path(p)

    # Jinja2 模板，用于格式化文件描述
    JJ_TPL = Environment(undefined=StrictUndefined).from_string(
        """
# {{file_name}}

## 文件类型
{{type_desc}}

## 内容概览
{{content}}
"""
    )

    if p.name.endswith(".h5"):
        # 处理 HDF5 文件
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### 数据结构\n"
        df_info += (
            f"- 索引: 多重索引，层级为 {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- 索引: {df.index.name}\n"
        )

        df_info += "\n### 列\n"
        columns = df.dtypes.to_dict()
        grouped_columns = {}

        # 按前缀对列进行分组
        for col in columns:
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            # 如果提供了变量列表，只显示相关的列
            df_info += "#### 相关列:\n"
            relevant_line = ", ".join(f"{col}: {columns[col]}" for col in variable_list if col in columns)
            df_info += relevant_line + "\n"
        else:
            # 否则，显示所有列
            df_info += "#### 所有列:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                header = "其他列" if prefix == "other" else f"{prefix} 相关列"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols)
                line = ", ".join(f"{col}: {columns[col]}" for col in cols)
                df_info += line + "\n"

        # 如果存在 'REPORT_PERIOD' 列，显示示例数据
        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### 示例数据\n"
            df_info += f"显示股票 {one_instrument} 的数据:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 数据文件",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        # 处理 Markdown 文件
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown 文档",
                content=content,
            )

    else:
        # 不支持的文件类型
        raise NotImplementedError(
            f"不支持文件类型 {p.name}。请为其实现描述函数。",
        )


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    直接获取数据文件夹的信息，用于准备提示信息。

    Parameters
    ----------
    fname_reg : str
        用于筛选文件名的正则表达式。
    flags: int
        re.match 的标志。
    variable_mapping: dict
        文件名到相关变量列表的映射。

    Returns
    -------
        str
            数据文件夹的描述。
    """

    # 如果数据文件夹不存在，则先生成数据
    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) 我认为这里的写法有点硬编码。
        # get_data_folder_intro 并不意味着我们一定要生成数据文件夹。
        generate_data_folder_from_qlib()

    content_l = []
    # 遍历调试数据文件夹中的文件
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- 文件分割线 -------------\n".join(content_l)
