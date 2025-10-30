from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# 因为我们使用 isinstance 来区分不同类型的任务，所以我们需要使用子类来表示不同类型的任务
class DataLoaderTask(CoSTEERTask):
    """
    数据加载器任务类。
    继承自 CoSTEERTask，专门用于表示与数据加载器相关的任务。
    目前是一个空类，主要用于类型区分。
    """
    pass
