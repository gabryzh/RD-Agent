from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from PIL import Image

if TYPE_CHECKING:
    from langchain_core.documents import Document

from rdagent.core.conf import RD_AGENT_SETTINGS


def load_documents_by_langchain(path: str) -> list[Document]:
    """
    使用 LangChain 从指定路径加载文档。

    Args:
        path (str): 包含文档的目录或文件的路径。

    Returns:
        list[Document]: 加载的 LangChain 文档对象列表。
    """
    if Path(path).is_dir():
        # 如果是目录，则加载目录下的所有 PDF 文件
        loader = PyPDFDirectoryLoader(path, silent_errors=True)
    else:
        # 如果是文件，则加载单个 PDF 文件
        loader = PyPDFLoader(path)
    return loader.load()


def process_documents_by_langchain(docs: list[Document]) -> dict[str, str]:
    """
    处理 LangChain 文档对象列表，并按文档名称（源文件路径）分组。

    Args:
        docs (list[Document]): LangChain 文档对象列表。

    Returns:
        dict[str, str]: 一个字典，键是文档名称，值是拼接后的文档内容。
    """
    content_dict = {}

    for doc in docs:
        # 获取文档的源路径
        source_path = Path(doc.metadata["source"])
        if source_path.exists():
            doc_name = str(source_path.resolve())
        else:
            doc_name = doc.metadata["source"]

        doc_content = doc.page_content

        # 将同一来源的页面内容拼接在一起
        if doc_name not in content_dict:
            content_dict[doc_name] = doc_content
        else:
            content_dict[doc_name] += doc_content

    return content_dict


def load_and_process_pdfs_by_langchain(path: str) -> dict[str, str]:
    """
    一个便捷函数，结合了加载和处理 PDF 的功能。
    """
    return process_documents_by_langchain(load_documents_by_langchain(path))


def load_and_process_one_pdf_by_azure_document_intelligence(
    path: Path,
    key: str,
    endpoint: str,
) -> str:
    """
    使用 Azure Document Intelligence 服务加载并处理单个 PDF 文件。

    Args:
        path (Path): PDF 文件的路径。
        key (str): Azure 服务的密钥。
        endpoint (str): Azure 服务的终结点 URL。

    Returns:
        str: 提取的文档内容。
    """
    # 获取 PDF 的总页数
    pages = len(PyPDFLoader(str(path)).load())

    # 初始化 Azure Document Analysis 客户端
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # 分析文档
    with path.open("rb") as file:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document",  # 使用预构建的文档模型
            file,
            pages=f"1-{pages}",
        )
        result = poller.result()

    return result.content


def load_and_process_pdfs_by_azure_document_intelligence(path: Path) -> dict[str, str]:
    """
    使用 Azure Document Intelligence 服务加载并处理目录或单个文件中的所有 PDF。

    Args:
        path (Path): 包含 PDF 的目录或单个 PDF 文件的路径。

    Returns:
        dict[str, str]: 一个字典，键是 PDF 文件路径，值是提取的内容。
    """
    # 从全局设置中断言密钥和终结点的存在
    assert RD_AGENT_SETTINGS.azure_document_intelligence_key is not None
    assert RD_AGENT_SETTINGS.azure_document_intelligence_endpoint is not None

    content_dict = {}
    ab_path = path.resolve()

    if ab_path.is_file():
        # 处理单个文件
        assert ".pdf" in ab_path.suffixes, "文件必须是 PDF 文件。"
        proc = load_and_process_one_pdf_by_azure_document_intelligence
        content_dict[str(ab_path)] = proc(
            ab_path,
            RD_AGENT_SETTINGS.azure_document_intelligence_key,
            RD_AGENT_SETTINGS.azure_document_intelligence_endpoint,
        )
    else:
        # 递归处理目录中的所有 PDF 文件
        for file_path in ab_path.rglob("*.pdf"):
            if file_path.is_file():
                content_dict[str(file_path)] = load_and_process_one_pdf_by_azure_document_intelligence(
                    file_path,
                    RD_AGENT_SETTINGS.azure_document_intelligence_key,
                    RD_AGENT_SETTINGS.azure_document_intelligence_endpoint,
                )
    return content_dict


def extract_first_page_screenshot_from_pdf(pdf_path: str) -> Image.Image:
    """
    从 PDF 文件中提取第一页的截图。

    Args:
        pdf_path (str): 本地 PDF 文件路径或 URL。

    Returns:
        Image.Image: PIL 图像对象。
    """
    if not Path(pdf_path).exists():
        # 如果是 URL，则下载内容
        response = requests.get(pdf_path)
        doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
    else:
        # 如果是本地文件，则直接打开
        doc = fitz.open(pdf_path)

    # 加载第一页并转换为图像
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    return image
