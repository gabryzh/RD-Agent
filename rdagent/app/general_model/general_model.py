import fire

from rdagent.components.coder.model_coder.task_loader import (
    ModelExperimentLoaderFromPDFfiles,
)
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
)
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.qlib.developer.model_coder import QlibModelCoSTEER


def extract_models_and_implement(report_file_path: str) -> None:
    """
    这是一个研究助手，用于从报告文件或论文中自动实现模型。

    它从给定的PDF报告文件中提取模型，并实现必要的操作。

    参数：
    report_file_path (str): 报告文件的路径。该文件必须是PDF文件。

    PDF报告的示例URL：
    - https://arxiv.org/pdf/2210.09789
    - https://arxiv.org/pdf/2305.10498
    - https://arxiv.org/pdf/2110.14446
    - https://arxiv.org/pdf/2205.12454
    - https://arxiv.org/pdf/2210.16518

    返回：
    None
    """
    scenario = GeneralModelScenario()
    logger.log_object(scenario, tag="场景")
    # 保存相关图片
    img = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(img, tag="pdf_image")
    exp = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    logger.log_object(exp, tag="加载实验")
    exp = QlibModelCoSTEER(scenario).develop(exp)
    logger.log_object(exp, tag="开发后的实验")


if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
