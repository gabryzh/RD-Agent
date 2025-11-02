# 导入 qlib 库
import qlib

# 初始化 qlib，指定数据存储路径
# provider_uri 指向 Qlib 数据存储的位置，这里使用了默认的 `~/.qlib/qlib_data/cn_data` 路径
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

# 从 qlib.data 模块导入 D，D 是 Qlib 提供的数据检索接口
from qlib.data import D

# 获取所有股票代码
instruments = D.instruments()
# 定义需要提取的字段，包括开盘价、收盘价、最高价、最低价、成交量和复权因子
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# 提取从 2008-12-29 至今的所有股票的日频数据
# .swaplevel() 交换索引的层级
# .sort_index() 对索引进行排序
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

# 将提取的全量数据保存为 HDF5 文件，用于正式实验
data.to_hdf("./daily_pv_all.h5", key="data")


# 再次定义需要提取的字段
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# 提取 2018-01-01 到 2019-12-31 期间、前 100 只股票的日频数据
# 这部分数据量较小，主要用于调试
data = (
    (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    # 选取前 100 只股票
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

# 将提取的调试数据保存为 HDF5 文件
data.to_hdf("./daily_pv_debug.h5", key="data")
