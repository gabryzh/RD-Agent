import os
import socket

import docker
import fire
import litellm
import typer
from litellm import completion, embedding
from litellm.utils import ModelResponse
from typing_extensions import Annotated

from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import cleanup_container


def check_docker_status() -> None:
    """检查Docker状态。"""
    container = None
    try:
        client = docker.from_env()
        client.images.pull("hello-world")
        container = client.containers.run("hello-world", detach=True)
        logs = container.logs().decode("utf-8")
        print(logs)
        logger.info("Docker状态正常。")
    except docker.errors.DockerException as e:
        logger.error(f"发生错误: {e}")
        logger.warning(
            "Docker状态异常，请检查Docker配置或重新安装。参考: https://docs.docker.com/engine/install/ubuntu/."
        )
    finally:
        cleanup_container(container, "健康检查")


def is_port_in_use(port):
    """检查端口是否被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_and_list_free_ports(start_port=19899, max_ports=10) -> None:
    """检查并列出空闲端口。"""
    is_occupied = is_port_in_use(port=start_port)
    if is_occupied:
        free_ports = []
        for port in range(start_port, start_port + max_ports):
            if not is_port_in_use(port):
                free_ports.append(port)
        logger.warning(
            f"端口 19899 被占用，请在运行 `rdagent ui` 命令时替换为可用端口。可用端口: {free_ports}"
        )
    else:
        logger.info("端口 19899 未被占用，您可以运行 `rdagent ui` 命令。")


def test_chat(chat_model, chat_api_key, chat_api_base):
    """测试聊天模型。"""
    logger.info(f"🧪 测试聊天模型: {chat_model}")
    try:
        if chat_api_base is None:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                messages=[
                    {"role": "user", "content": "你好!"},
                ],
            )
        else:
            response: ModelResponse = completion(
                model=chat_model,
                api_key=chat_api_key,
                api_base=chat_api_base,
                messages=[
                    {"role": "user", "content": "你好!"},
                ],
            )
        logger.info("✅ 聊天测试通过。")
        return True
    except Exception as e:
        logger.error(f"❌ 聊天测试失败: {e}")
        return False


def test_embedding(embedding_model, embedding_api_key, embedding_api_base):
    """测试嵌入模型。"""
    logger.info(f"🧪 测试嵌入模型: {embedding_model}")
    try:
        response = embedding(
            model=embedding_model,
            api_key=embedding_api_key,
            api_base=embedding_api_base,
            input="你好，世界!",
        )
        logger.info("✅ 嵌入测试通过。")
        return True
    except Exception as e:
        logger.error(f"❌ 嵌入测试失败: {e}")
        return False


def env_check():
    """检查环境变量和API配置。"""
    if "BACKEND" not in os.environ:
        logger.warning(
            "在您的配置中未找到BACKEND，请将其添加到您的.env文件中。"
            "您可以运行类似 `dotenv set BACKEND rdagent.oai.backend.LiteLLMAPIBackend` 的命令。"
        )

    chat_api_key = None
    if "DEEPSEEK_API_KEY" in os.environ:
        chat_api_key = os.getenv("DEEPSEEK_API_KEY")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("LITELLM_PROXY_API_KEY")
        embedding_api_base = os.getenv("LITELLM_PROXY_API_BASE")
        if "DEEPSEEK_API_BASE" in os.environ:
            chat_api_base = os.getenv("DEEPSEEK_API_BASE")
        elif "OPENAI_API_BASE" in os.environ:
            chat_api_base = os.getenv("OPENAI_API_BASE")
        else:
            chat_api_base = None
    elif "OPENAI_API_KEY" in os.environ:
        chat_api_key = os.getenv("OPENAI_API_KEY")
        chat_api_base = os.getenv("OPENAI_API_BASE")
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = chat_api_key
        embedding_api_base = chat_api_base
    else:
        logger.error("未找到有效配置，请检查您的.env文件。")
        return

    logger.info("🚀 开始测试...\n")
    result_embedding = test_embedding(
        embedding_model=embedding_model, embedding_api_key=embedding_api_key, embedding_api_base=embedding_api_base
    )
    result_chat = test_chat(chat_model=chat_model, chat_api_key=chat_api_key, chat_api_base=chat_api_base)

    if result_chat and result_embedding:
        logger.info("✅ 所有测试完成。")
    else:
        logger.error("一个或多个测试失败。请检查凭据或模型支持。")


def health_check(
    check_env: Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")] = True,
    check_docker: Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")] = True,
    check_ports: Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")] = True,
):
    """
    运行RD-Agent健康检查：
    - 检查Docker是否可用
    - 检查默认端口是否被占用
    - (可选) 检查API密钥和模型是否配置正确。

    参数：
        check_env (bool): 是否检查API密钥和模型配置。
        check_docker (bool): 检查Docker是否已安装并正在运行。
        check_ports (bool): 是否检查默认端口(19899)是否被占用。
    """
    check_any = False

    if check_env:
        check_any = True
        env_check()
    if check_docker:
        check_any = True
        check_docker_status()
    if check_ports:
        check_any = True
        check_and_list_free_ports()

    if not check_any:
        logger.warning("⚠️ 所有健康检查项均已禁用。请至少启用一项检查。")


if __name__ == "__main__":
    typer.run(health_check)
