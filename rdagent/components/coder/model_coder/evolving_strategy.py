import json
from typing import Dict

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class ModelMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """
    针对模型生成的特定多进程演进策略。
    """
    def implement_one_task(
        self,
        target_task: ModelTask,
        queried_knowledge: CoSTEERQueriedKnowledge = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> str:
        """
        实现单个模型任务，生成代码。

        Args:
            target_task (ModelTask): 目标模型任务。
            queried_knowledge (CoSTEERQueriedKnowledge, optional): 查询到的相关知识。
            workspace (FBWorkspace | None, optional): 当前工作空间，包含已有代码。
            prev_task_feedback (CoSTEERSingleFeedback | None, optional): 上一个任务的反馈。

        Returns:
            str: 生成的模型代码。
        """
        # 获取模型任务的详细信息字符串
        model_information_str = target_task.get_task_information()

        # 从查询知识中提取相似成功案例和历史失败记录
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge.get(model_information_str, [])
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces.get(model_information_str, [])
            if queried_knowledge is not None
            else []
        )

        # 兼容不同版本的查询知识
        queried_former_failed_knowledge_to_render = (
            queried_former_failed_knowledge[0]
            if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
            else queried_former_failed_knowledge
        )

        # 构建 system prompt
        system_prompt = T(".prompts:evolving_strategy_model_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(filtered_tag="model"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            current_code=workspace.file_dict.get("model.py") if workspace else None,
        )

        # 准备渲染到 user prompt 的知识，并循环缩减以防止超长
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        for _ in range(10):  # 最多尝试10次来缩减 prompt 长度
            user_prompt = T(".prompts:evolving_strategy_model_coder.user").r(
                model_information_str=model_information_str,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )

            # 检查 token 数量是否在限制内
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < APIBackend().chat_token_limit
            ):
                break
            # 如果超长，优先缩减历史失败记录，其次缩减相似成功案例
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > 1:
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[1:]

        # 调用 LLM API 生成代码
        response = APIBackend(use_chat_cache=CoSTEER_SETTINGS.coder_use_cache).build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            json_target_type=Dict[str, str],
        )
        code = json.loads(response)["code"]
        return code

    def assign_code_list_to_evo(self, code_list: list[str], evo: 'EvolvingExperiment') -> 'EvolvingExperiment':
        """
        将生成的代码列表分配给演进实验中的各个子任务。

        Args:
            code_list (list[str]): 生成的代码字符串列表。
            evo ('EvolvingExperiment'): 正在进行的演进实验对象。

        Returns:
            'EvolvingExperiment': 更新了代码的演进实验对象。
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            # 如果子工作空间不存在，则创建一个新的
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = ModelFBWorkspace(target_task=evo.sub_tasks[index])
            # 将生成的代码注入到工作空间的 'model.py' 文件中
            evo.sub_workspace_list[index].inject_files(**{"model.py": code_list[index]})
        return evo
