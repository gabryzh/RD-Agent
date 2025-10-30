import importlib.metadata
import platform
import sys
from pathlib import Path

import docker
import requests
from setuptools_scm import get_version

from rdagent.log import rdagent_logger as logger


def sys_info():
    """收集系统相关信息"""
    method_list = [
        ["当前操作系统名称: ", "system"],
        ["处理器架构: ", "machine"],
        ["系统、版本和硬件信息: ", "platform"],
        ["系统版本号: ", "version"],
    ]
    for method in method_list:
        logger.info(f"{method[0]}{getattr(platform, method[1])()}")
    return None


def python_info():
    """收集Python相关信息"""
    python_version = sys.version.replace("\n", " ")
    logger.info(f"Python版本: {python_version}")
    return None


def docker_info():
    """收集Docker相关信息"""
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        if containers:
            containers.sort(key=lambda c: c.attrs["Created"])
            last_container = containers[-1]
            logger.info(f"容器ID: {last_container.id}")
            logger.info(f"容器名称: {last_container.name}")
            logger.info(f"容器状态: {last_container.status}")
            logger.info(f"容器使用的镜像ID: {last_container.image.id}")
            logger.info(f"容器使用的镜像标签: {last_container.image.tags}")
            logger.info(f"容器端口映射: {last_container.ports}")
            logger.info(f"容器标签: {last_container.labels}")
            logger.info(f"启动命令: {' '.join(client.containers.get(last_container.id).attrs['Config']['Cmd'])}")
        else:
            logger.info("没有正在运行的容器。")
    except docker.errors.DockerException:
        logger.warning("Docker守护进程未运行或未安装。")


def rdagent_info():
    """收集rdagent相关信息"""
    try:
        current_version = importlib.metadata.version("rdagent")
        logger.info(f"RD-Agent版本: {current_version}")
        api_url = "https://api.github.com/repos/microsoft/RD-Agent/contents/requirements.txt?ref=main"
        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()
            file_url = files["download_url"]
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                all_file_contents = file_response.text.split("\n")
            else:
                logger.warning(f"无法检索 {files['name']}，状态码: {file_response.status_code}")
                return
        else:
            logger.warning(f"无法检索文件夹中的文件，状态码: {response.status_code}")
            return

        package_list = [
            item.split("#")[0].strip() for item in all_file_contents if item.strip() and not item.startswith("#")
        ]
        package_version_list = []
        for package in package_list:
            if package == "typer[all]":
                package = "typer"
            try:
                version = importlib.metadata.version(package)
                package_version_list.append(f"{package}=={version}")
            except importlib.metadata.PackageNotFoundError:
                logger.warning(f"未找到软件包: {package}")
        logger.info(f"软件包版本: {package_version_list}")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("未安装rdagent。")
    return None


def collect_info():
    """打印有关系统和已安装软件包的信息。"""
    sys_info()
    python_info()
    docker_info()
    rdagent_info()
    return None
