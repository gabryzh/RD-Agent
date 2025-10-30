from rdagent.core.conf import ExtendedBaseSettings


class BasePropSetting(ExtendedBaseSettings):
    """
    研发循环（RD Loop）提议和开发配置的通用部分。
    您可以在子类中添加以下配置，以区分不同的环境变量。
    """

    # 场景类的完整路径，例如：'rdagent.scenarios.qlib.scenario.QlibScenario'
    scen: str = ""

    # 知识库类的完整路径
    knowledge_base: str = ""

    # 知识库文件的存储路径
    knowledge_base_path: str = ""

    # 假设生成器类的完整路径
    hypothesis_gen: str = ""

    # 交互器类的完整路径
    interactor: str = ""

    # 从假设到实验转换器类的完整路径
    hypothesis2experiment: str = ""

    # 编码器（开发者）类的完整路径
    coder: str = ""

    # 运行器（开发者）类的完整路径
    runner: str = ""

    # 总结器（从实验到反馈）类的完整路径
    summarizer: str = ""

    # 演进的代数或次数
    evolving_n: int = 10
