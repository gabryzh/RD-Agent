from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import requests

from rdagent.log.base import Message, Storage
from rdagent.log.utils import extract_evoid, extract_loopid_func_name, gen_datetime

from .conf import UI_SETTING


class WebStorage(Storage):
    """
    Web 应用的存储。
    用于为 Web 应用提供数据。
    """

    def __init__(self, port: int, path: str) -> None:
        """
        使用指定的端口和标识符初始化存储对象。
        参数:
            port (int): 用于存储服务的端口号。
            path (str): 本地存储的唯一标识符，即日志路径。
        """
        self.url = f"http://localhost:{port}"
        self.path = path
        self.msgs = []

    def __str__(self):
        return f"WebStorage({self.url})"

    def log(self, obj: object, tag: str, timestamp: datetime | None = None, **kwargs: Any) -> str | Path:
        """
        将对象记录到 Web 服务器。
        """
        timestamp = gen_datetime(timestamp)
        # 如果是图片，则保存到静态文件夹
        if "pdf_image" in tag or "load_pdf_screenshot" in tag:
            obj.save(f"{UI_SETTING.static_path}/{timestamp.isoformat()}.jpg")

        try:
            # 将对象转换为 JSON 格式
            data = self._obj_to_json(obj=obj, tag=tag, id=self.path, timestamp=timestamp.isoformat())
            if not data:
                return "普通日志，已跳过"

            # 将消息添加到本地列表并通过 HTTP POST 发送
            if isinstance(data, list):
                for d in data:
                    self.msgs.append(d)
            else:
                self.msgs.append(data)
            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{self.url}/receive", json=data, headers=headers, timeout=1)
            return f"{resp.status_code} {resp.text}"
        except (requests.ConnectionError, requests.Timeout) as e:
            # 忽略连接错误
            pass

    def truncate(self, time: datetime) -> None:
        """截断指定时间之后的消息。"""
        self.msgs = [m for m in self.msgs if datetime.fromisoformat(m["msg"]["timestamp"]) <= time]

    def iter_msg(self, **kwargs: Any) -> Generator[Message, None, None]:
        """迭代存储的消息。"""
        for msg in self.msgs:
            yield Message(
                tag=msg["msg"]["tag"],
                level="INFO",
                timestamp=datetime.fromisoformat(msg["msg"]["timestamp"]),
                content=msg,
            )

    def _obj_to_json(
        self,
        obj: object,
        tag: str,
        id: str,
        timestamp: str,
    ) -> list[dict] | dict:
        """
        根据标签将不同的 Python 对象转换为特定格式的 JSON 字典。
        这是 UI 前端和后端之间的主要数据协议。
        """
        li, fn = extract_loopid_func_name(tag)
        ei = extract_evoid(tag)
        data = {}
        # 根据不同的 tag，将 obj 转换为不同的 JSON 格式
        if "hypothesis generation" in tag:
            # ... 处理假设生成 ...
            pass
        elif "pdf_image" in tag or "load_pdf_screenshot" in tag:
            # ... 处理 PDF 图片 ...
            pass
        elif "experiment generation" in tag or "load_experiment" in tag:
            # ... 处理实验生成 ...
            pass
        elif "direct_exp_gen" in tag:
            # ... 处理直接实验生成（数据科学场景） ...
            pass
        elif f"evo_loop_{ei}.evolving code" in tag and "running" not in tag:
            # ... 处理演进代码 ...
            pass
        elif f"evo_loop_{ei}.evolving feedback" in tag and "running" not in tag:
            # ... 处理演进反馈 ...
            pass
        elif "scenario" in tag:
            # ... 处理场景配置 ...
            pass
        elif "Quantitative Backtesting Chart" in tag:
            # ... 处理量化回测图表 ...
            pass
        elif "running" in tag:
            # ... 处理运行结果 ...
            pass
        elif "feedback" in tag:
            # ... 处理反馈 ...
            pass

        return data
