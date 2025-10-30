class WorkflowError(Exception):
    """
    工作流错误，表示当前循环无法处理的错误，阻止了进一步的进展。
    """


class FormatError(WorkflowError):
    """
    格式错误，在多次尝试后，我们无法以正确的格式获取答案以继续。
    """


class CoderError(WorkflowError):
    """
    编码器错误，在实现和运行代码时引发的异常。
    - 开始: 因子任务 => 因子生成器
    - 结束: 执行后获取数据帧

    数据帧值中更详细的评估由评估器管理。
    """

    # 注意：它对应于**组件**的错误
    caused_by_timeout: bool = False  # 错误是否由超时引起


class CodeFormatError(CoderError):
    """
    代码格式错误，由于格式错误，未找到生成的代码。
    """


class CustomRuntimeError(CoderError):
    """
    自定义运行时错误，生成的代码未能执行脚本。
    """


class NoOutputError(CoderError):
    """
    无输出错误，代码未能生成输出文件。
    """


class RunnerError(Exception):
    """
    运行器错误，在运行代码输出时引发的异常。
    """

    # 注意：它对应于整个**项目**的错误


FactorEmptyError = CoderError  # 未正确生成因子时引发的异常

ModelEmptyError = CoderError  # 未正确生成模型时引发的异常


class KaggleError(Exception):
    """
    Kaggle错误，在调用Kaggle API时引发的异常。
    """


class PolicyError(Exception):
    """
    策略错误，由于内容管理策略而引发的异常。
    """
