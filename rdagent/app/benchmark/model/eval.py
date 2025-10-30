from pathlib import Path

from rdagent.components.coder.model_coder import ModelCoSTEER
from rdagent.components.loader.task_loader import ModelTaskLoaderJson, ModelWsLoader
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

if __name__ == "__main__":
    # 获取当前文件所在的目录
    DIRNAME = Path(__file__).absolute().resolve().parent

    from rdagent.components.coder.model_coder.benchmark.eval import ModelImpValEval
    from rdagent.components.coder.model_coder.one_shot import ModelCodeWriter

    # 定义基准测试文件夹的路径
    bench_folder = DIRNAME.parent.parent.parent / "components" / "coder" / "model_coder" / "benchmark"
    # 从JSON文件加载模型任务
    mtl = ModelTaskLoaderJson(str(bench_folder / "model_dict.json"))

    task_l = mtl.load()

    # FIXME: 其他模型效果不佳，暂时只测试A-DGN
    task_l = [t for t in task_l if t.name == "A-DGN"]

    # 创建模型实验
    model_experiment = QlibModelExperiment(sub_tasks=task_l)
    # mtg = ModelCodeWriter(scen=QlibModelScenario())
    mtg = ModelCoSTEER(scen=QlibModelScenario())

    # 开发模型实验
    model_experiment = mtg.develop(model_experiment)

    # TODO: 在@wenjun完善评估部分后，与基准框架对齐。
    # 目前，我们只是为了快速评估而手动构建了一个工作流。

    # 加载真实实现
    mil = ModelWsLoader(bench_folder / "gt_code")

    mie = ModelImpValEval()
    # 评估
    eval_l = []
    for impl in model_experiment.sub_workspace_list:
        print(impl.target_task)
        gt_impl = mil.load(impl.target_task)
        eval_l.append(mie.evaluate(gt_impl, impl))

    print(eval_l)
