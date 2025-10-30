from rdagent.core.developer import Developer
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.oai.llm_utils import md5_hash


class CachedRunner(Developer[ASpecificExp]):
    """
    一个带缓存功能的运行器。
    它继承自 Developer，并实现了基于任务信息的缓存键生成逻辑。
    """

    def get_cache_key(self, exp: Experiment) -> str:
        """
        根据实验中的所有任务信息生成一个缓存键。

        参数:
            exp (Experiment): 实验对象。

        返回:
            str: 根据任务信息生成的 MD5 哈希值，用作缓存键。
        """
        all_tasks = []
        # 收集所有基础实验中的子任务
        for based_exp in exp.based_experiments:
            all_tasks.extend(based_exp.sub_tasks)
        # 添加当前实验的子任务
        all_tasks.extend(exp.sub_tasks)
        # 将所有任务的信息整合成一个字符串
        task_info_list = [task.get_task_information() for task in all_tasks]
        task_info_str = "\n".join(task_info_list)
        # 返回任务信息字符串的 MD5 哈希值
        return md5_hash(task_info_str)

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        """
        将缓存的结果分配给当前的实验对象。

        参数:
            exp (Experiment): 当前的实验对象。
            cached_res (Experiment): 包含缓存结果的实验对象。

        返回:
            Experiment: 更新了结果的当前实验对象。
        """
        # 如果基础实验的结果为空，则用缓存的结果填充
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].result = cached_res.based_experiments[-1].result
        # 用缓存的结果更新当前实验的结果
        exp.result = cached_res.result
        return exp
