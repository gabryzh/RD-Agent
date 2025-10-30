from __future__ import annotations

import json
import re
from typing import Dict

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    """
    针对因子生成的特定多进程演进策略。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        """
        使用 LLM 对历史错误进行总结。

        Args:
            target_task (FactorTask): 目标任务。
            queried_former_failed_knowledge_to_render (list): 当前任务的历史失败记录。
            queried_similar_error_knowledge_to_render (list): 其他相似任务的错误记录。

        Returns:
            str: LLM 生成的错误总结。
        """
        error_summary_system_prompt = T(".prompts:evolving_strategy_error_summary_v2_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            factor_information_str=target_task.get_task_information(),
            code_and_feedback=queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str(),
        )
        # 动态缩减 prompt
        for _ in range(10):
            error_summary_user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < APIBackend().chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

        error_summary_critics = APIBackend(
            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
        ).build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return error_summary_critics

    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> str:
        """
        实现单个因子任务，生成代码。
        """
        target_factor_task_information = target_task.get_task_information()

        # 提取相关知识
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge.get(target_factor_task_information, [])
            if queried_knowledge is not None
            else []
        )
        queried_similar_error_knowledge = (
            queried_knowledge.task_to_similar_error_successful_knowledge.get(target_factor_task_information, {})
            if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
            else {}
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces.get(target_factor_task_information, [[]])[0]
            if queried_knowledge is not None
            else []
        )
        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces.get(
            target_factor_task_information, [None, None]
        )[1]

        # 准备渲染到 prompt 的知识
        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge

        # 构建 system prompt
        system_prompt = T(".prompts:evolving_strategy_factor_implementation_v1_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
        )

        # 动态缩减 user prompt 以防止超长
        for _ in range(10):
            # 可选的错误总结步骤
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and queried_similar_error_knowledge_to_render
                and queried_former_failed_knowledge_to_render
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None

            # 构建 user prompt
            user_prompt = T(".prompts:evolving_strategy_factor_implementation_v2_user").r(
                factor_information_str=target_factor_task_information,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                error_summary_critics=error_summary_critics,
                latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
            )

            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < APIBackend().chat_token_limit
            ):
                break
            # 缩减逻辑
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(queried_similar_error_knowledge_to_render):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

        # 调用 LLM 生成代码，带重试机制
        for _ in range(10):
            try:
                response = APIBackend(
                    use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                ).build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str],
                )
                try:
                    code = json.loads(response)["code"]
                except json.decoder.JSONDecodeError:
                    # 如果JSON解析失败，尝试从代码块中提取
                    match = re.search(r"```python(.*?)```", response, re.DOTALL)
                    if match:
                        code = match.group(1).strip()
                    else:
                        raise  # 抛出异常以继续重试
                return code
            except (json.decoder.JSONDecodeError, KeyError):
                pass
        else:
            return ""  # 10次尝试后仍失败，返回空代码

    def assign_code_list_to_evo(self, code_list: list[str], evo: 'EvolvingExperiment') -> 'EvolvingExperiment':
        """
        将生成的代码列表分配给演进实验。
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_files(**{"factor.py": code_list[index]})
        return evo
