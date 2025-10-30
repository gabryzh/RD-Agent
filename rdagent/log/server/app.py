import os
import random
import signal
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import randomname
import typer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from rdagent.log.storage import FileStorage
from rdagent.log.ui.conf import UI_SETTING
from rdagent.log.ui.storage import WebStorage
from rdagent.log.utils import is_valid_session

# 初始化 Flask 应用
app = Flask(__name__, static_folder=UI_SETTING.static_path)
CORS(app)

# 全局变量
rdagent_processes = defaultdict()
server_port = 19899
log_folder_path = Path(UI_SETTING.trace_folder).absolute()


@app.route("/favicon.ico")
def favicon():
    """提供网站图标。"""
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


# 用于存储前端消息和指针的全局字典
msgs_for_frontend = defaultdict(list)
pointers = defaultdict(lambda: defaultdict(int))  # pointers[trace_id][user_ip]


def read_trace(log_path: Path, id: str = "") -> None:
    """
    从给定的日志路径读取追踪信息并填充到 `msgs_for_frontend`。
    """
    fs = FileStorage(log_path)
    ws = WebStorage(port=1, path=log_path)
    msgs_for_frontend[id] = []
    last_timestamp = None
    for msg in fs.iter_msg():
        data = ws._obj_to_json(obj=msg.content, tag=msg.tag, id=id, timestamp=msg.timestamp.isoformat())
        if data:
            if isinstance(data, list):
                for d in data:
                    msgs_for_frontend[id].append(d["msg"])
                    last_timestamp = msg.timestamp
            else:
                msgs_for_frontend[id].append(data["msg"])
                last_timestamp = msg.timestamp

    # 如果最后一条消息的时间戳距离现在超过30分钟，则添加一个结束标记
    now = datetime.now(timezone.utc)
    if last_timestamp and (now - last_timestamp).total_seconds() > 1800:
        msgs_for_frontend[id].append({"tag": "END", "timestamp": now.isoformat(), "content": {}})


# 从日志文件夹加载所有追踪信息
for p in log_folder_path.glob("*/*/"):
    if is_valid_session(p):
        read_trace(p, id=str(p))


@app.route("/trace", methods=["POST"])
def update_trace():
    """
    前端轮询此端点以获取新的日志消息。
    """
    global pointers, msgs_for_frontend
    data = request.get_json()
    trace_id = data.get("id")
    return_all = data.get("all")
    reset = data.get("reset")
    msg_num = random.randint(1, 10)  # 每次返回随机数量的消息
    app.logger.info(data)
    log_folder_path = Path(UI_SETTING.trace_folder).absolute()
    if not trace_id:
        return jsonify({"error": "需要提供追踪 ID"}), 400
    trace_id = str(log_folder_path / trace_id)

    user_ip = request.remote_addr

    if reset:
        pointers[trace_id][user_ip] = 0

    start_pointer = pointers[trace_id][user_ip]
    end_pointer = start_pointer + msg_num
    if end_pointer > len(msgs_for_frontend[trace_id]) or return_all:
        end_pointer = len(msgs_for_frontend[trace_id])

    returned_msgs = msgs_for_frontend[trace_id][start_pointer:end_pointer]

    pointers[trace_id][user_ip] = end_pointer
    if returned_msgs:
        app.logger.info([msg["tag"] for msg in returned_msgs])
    return jsonify(returned_msgs), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    处理文件上传并启动 rdagent 进程。
    """
    global rdagent_processes, server_port
    # 从表单中获取数据
    scenario = request.form.get("scenario")
    files = request.files.getlist("files")
    competition = request.form.get("competition")
    loop_n = request.form.get("loops")
    all_duration = request.form.get("all_duration")

    # 根据场景生成追踪名称
    if scenario == "Data Science":
        competition = competition[10:]
        trace_name = f"{competition}-{randomname.get_name()}"
    else:
        trace_name = randomname.get_name()

    # 设置日志路径和标准输出路径
    log_trace_path = (log_folder_path / scenario / trace_name).absolute()
    stdout_path = log_folder_path / scenario / f"{trace_name}.stdout"
    if not stdout_path.exists():
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存上传的文件
    for file in files:
        if file:
            p = (log_folder_path / scenario / "uploads" / trace_name).resolve()
            sanitized_filename = secure_filename(file.filename)
            target_path = (p / sanitized_filename).resolve()
            if not sanitized_filename.lower().endswith(".pdf"):
                return jsonify({"error": "无效的文件类型"}), 400
            if os.path.commonpath([str(target_path), str(p)]) == str(p) and not target_path.is_file():
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                file.save(target_path)
            else:
                return jsonify({"error": "无效的文件路径"}), 400

    # 根据场景构建命令
    if scenario == "Finance Data Building":
        cmds = ["rdagent", "fin_factor"]
    elif scenario == "Finance Data Building (Reports)":
        cmds = ["rdagent", "fin_factor_report", "--report_folder", str(trace_files_path)]
    elif scenario == "Finance Model Implementation":
        cmds = ["rdagent", "fin_model"]
    elif scenario == "General Model Implementation":
        rfp = request.form.get("files")[0] if not files else str(trace_files_path / files[0].filename)
        cmds = ["rdagent", "general_model", "--report_file_path", rfp]
    elif scenario == "Finance Whole Pipeline":
        cmds = ["rdagent", "fin_quant"]
    elif scenario == "Data Science":
        cmds = ["rdagent", "data_science", "--competition", competition]

    # 添加时间控制参数
    if scenario != "Finance Data Building (Reports)" and loop_n:
        cmds += ["--loop_n", loop_n]
    if all_duration:
        cmds += ["--timeout", f"{all_duration}h"]

    app.logger.info(f"为 {log_trace_path} 启动进程，参数: {cmds}")
    # 启动子进程
    with stdout_path.open("w") as log_file:
        rdagent_processes[str(log_trace_path)] = subprocess.Popen(
            cmds,
            stdout=log_file,
            stderr=log_file,
            env={
                **os.environ,
                "LOG_TRACE_PATH": str(log_trace_path),
                "LOG_UI_SERVER_PORT": str(server_port),
            },
        )
    return jsonify({"id": f"{scenario}/{trace_name}"}), 200


@app.route("/receive", methods=["POST"])
def receive_msgs():
    """
    接收来自 rdagent 进程的日志消息。
    """
    try:
        data = request.get_json()
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
    控制 rdagent 进程（暂停、继续、停止）。
    """
    global rdagent_processes, msgs_for_frontend
    data = request.get_json()
    app.logger.info(data)
    if not data or "id" not in data or "action" not in data:
        return jsonify({"error": "请求中缺少 'id' 或 'action'"}), 400

    id = str(log_folder_path / data["id"])
    action = data["action"]

    if id not in rdagent_processes or rdagent_processes[id] is None:
        return jsonify({"error": "未找到给定 id 的正在运行的进程"}), 400

    process = rdagent_processes[id]

    if process.poll() is not None:
        msgs_for_frontend[id].append({"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}})
        return jsonify({"error": "进程已终止"}), 400

    try:
        if action == "pause":
            os.kill(process.pid, signal.SIGSTOP)
            return jsonify({"status": "已暂停"}), 200
        elif action == "resume":
            os.kill(process.pid, signal.SIGCONT)
            return jsonify({"status": "已恢复"}), 200
        elif action == "stop":
            process.terminate()
            process.wait()
            del rdagent_processes[id]
            msgs_for_frontend[id].append(
                {"tag": "END", "timestamp": datetime.now(timezone.utc).isoformat(), "content": {}}
            )
            return jsonify({"status": "已停止"}), 200
        else:
            return jsonify({"error": "未知操作"}), 400
    except Exception as e:
        return jsonify({"error": f"操作 {action} 进程失败, {e}"}), 500


@app.route("/test", methods=["GET"])
def test():
    """测试端点，用于检查服务器状态。"""
    global msgs_for_frontend, pointers
    msgs = {k: [i["tag"] for i in v] for k, v in msgs_for_frontend.items()}
    pointers = pointers
    return jsonify({"msgs": msgs, "pointers": pointers}), 200


@app.route("/", methods=["GET"])
def index():
    """提供主页 index.html。"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:fn>", methods=["GET"])
def server_static_files(fn):
    """提供静态文件。"""
    return send_from_directory(app.static_folder, fn)


def main(port: int = 19899):
    """主函数，启动 Flask 服务器。"""
    global server_port
    server_port = port
    app.run(debug=False, host="0.0.0.0", port=port)


if __name__ == "__main__":
    typer.run(main)
