import multiprocessing
import os
import random
import signal
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import randomname
import typer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from rdagent.log.ui.conf import UI_SETTING

# 初始化 Flask 应用
app = Flask(__name__, static_folder=UI_SETTING.static_path)
CORS(app)

# 全局变量
rdagent_processes = defaultdict()
server_port = 19899


@app.route("/favicon.ico")
def favicon():
    """提供网站图标。"""
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


# 用于存储前端消息和指针的全局字典
msgs_for_frontend = defaultdict(list)
pointers = defaultdict(int)


@app.route("/trace", methods=["POST"])
def update_trace():
    """
    前端轮询此端点以获取新的日志消息（调试用）。
    """
    global pointers, msgs_for_frontend
    data = request.get_json()
    trace_id = data.get("id")
    return_all = data.get("all")
    reset = data.get("reset")
    msg_num = random.randint(1, 10)

    if reset:
        pointers[trace_id] = 0

    end_pointer = pointers[trace_id] + msg_num
    if end_pointer > len(msgs_for_frontend[trace_id]) or return_all:
        end_pointer = len(msgs_for_frontend[trace_id])

    returned_msgs = msgs_for_frontend[trace_id][pointers[trace_id] : end_pointer]

    pointers[trace_id] = end_pointer
    return jsonify(returned_msgs), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    处理上传请求，但不实际启动进程，而是读取一个预设的日志文件进行模拟（调试用）。
    """
    global rdagent_processes, server_port, msgs_for_frontend
    # 从表单中获取数据
    scenario = request.form.get("scenario")
    competition = request.form.get("competition")

    # 硬编码的日志文件夹路径
    log_folder_path = Path("/home/bowen/workspace/new_traces").absolute()

    # 根据场景确定要读取的日志路径
    if scenario == "Data Science":
        trace_path = log_folder_path / "o1-preview" / f"{competition[10:]}.1"
    else:
        trace_path = log_folder_path / scenario
    id = f"{scenario}/{randomname.get_name()}"

    def read_trace(log_path: Path, t: float = 0.2, id: str = "") -> None:
        """
        以模拟流式的方式读取日志追踪文件。
        """
        from rdagent.log.storage import FileStorage
        from rdagent.log.ui.storage import WebStorage

        fs = FileStorage(log_path)
        ws = WebStorage(port=1, path=log_path)
        msgs_for_frontend[id] = []
        for msg in fs.iter_msg():
            data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=id, timestamp=msg.timestamp.isoformat())
            if data:
                if isinstance(data, list):
                    for d in data:
                        time.sleep(t) # 模拟延迟
                        msgs_for_frontend[id].append(d["msg"])
                else:
                    time.sleep(t) # 模拟延迟
                    msgs_for_frontend[id].append(data["msg"])
        msgs_for_frontend[id].append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})

    # 在后台线程中读取日志，以避免阻塞响应
    threading.Thread(target=read_trace, args=(trace_path, 0.5, id), daemon=True).start()

    return jsonify({"id": id}), 200


@app.route("/receive", methods=["POST"])
def receive_msgs():
    """
    接收来自 rdagent 进程的日志消息（调试用）。
    """
    try:
        data = request.get_json()
        app.logger.info(data["msg"]["tag"])
        if not data:
            return jsonify({"error": "未收到 JSON 数据"}), 400
    except Exception as e:
        return jsonify({"error": "内部服务器错误"}), 500

    if isinstance(data, list):
        for d in data:
            msgs_for_frontend[d["id"]].append(d["msg"])
    else:
        msgs_for_frontend[data["id"]].append(data["msg"])

    return jsonify({"status": "成功"}), 200


@app.route("/control", methods=["POST"])
def control_process():
    """
    模拟控制 rdagent 进程（调试用）。
    """
    global rdagent_processes
    data = request.get_json()
    app.logger.info(data)
    if not data or "id" not in data or "action" not in data:
        return jsonify({"error": "请求中缺少 'id' 或 'action'"}), 400

    id = data["id"]
    action = data["action"]

    return jsonify({"status": "成功", "message": f"收到对 ID 为 '{id}' 的进程的操作 '{action}'"}), 200


@app.route("/test", methods=["GET"])
def test():
    """测试端点，用于检查服务器状态（调试用）。"""
    return {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}


@app.route("/", methods=["GET"])
def index():
    """提供主页 index.html。"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    """提供静态文件。"""
    return send_from_directory(app.static_folder, fn)


def main(port: int = 19899):
    """主函数，以调试模式启动 Flask 服务器。"""
    global server_port
    server_port = port
    app.run(debug=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
