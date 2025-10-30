from typing import Optional

import typer

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.utils.agent.tpl import T

# 创建一个typer应用程序实例，用于运行数据科学环境命令
app = typer.Typer(help="运行数据科学环境命令。")


@app.command()
def run(competition: str, cmd: str, local_path: str = "./", mount_path: str | None = None):
    """
    为特定竞赛启动数据科学环境并运行提供的命令。

    示例：
        1) 启动容器：
        dotenv run -- python -m rdagent.app.utils.ws nomad2018-predict-transparent-conductors "sleep 3600" --local-path your_workspace

        2) 然后运行以下命令进入最新的容器：
        - docker exec -it `docker ps --filter 'status=running' -l --format '{{.Names}}'` bash
        或者，您可以通过指定容器名称来附加到容器（在运行信息中找到它）：
        - docker exec -it sweet_robinson bash

    参数：
        competition: 竞赛的slug/文件夹名称。
        cmd: 要在环境中执行的shell命令或脚本入口点。
    """
    data_path = DS_RD_SETTING.local_data_path

    # 根据是否通过LLM采样数据来确定数据路径
    data_path = (
        f"{data_path}/{competition}" if DS_RD_SETTING.sample_data_by_LLM else f"{data_path}/sample/{competition}"
    )
    target_path = T("scenarios.data_science.share:scen.input_path").r()
    extra_volumes = {data_path: target_path}

    # 不设置时间限制并始终禁用缓存
    env = get_ds_env(
        extra_volumes=extra_volumes,
        running_timeout_period=None,
        enable_cache=False,
    )

    if mount_path is not None:
        env.conf.mount_path = mount_path

    env.run(entry=cmd, local_path=local_path)


if __name__ == "__main__":  # pragma: no cover
    app()
