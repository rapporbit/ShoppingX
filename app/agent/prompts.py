"""提示词读取层：从 ``prompt/prompts.yml`` 加载并渲染。

System prompt 用 XML 分块，且**纯静态**——不含任何运行时注入位。
长期偏好 / 近期行为历史 / 会话级 P_t 这些**每轮必变**的运行时上下文一律**不进 system prompt**，
改由 ``main_agent._inject_runtime_context`` 拼进当轮 human message（见该函数的 prompt cache 说明）：
system prompt 逐字稳定才能成为跨轮 / 跨会话都命中的缓存前缀。主 / 子 AgentLoop 共用同一份 system
prompt（同质 fork 的硬约束）——静态化后主与子的 system 段字节相同，子 Agent 也能命中主 Agent 的缓存。
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# prompts.yml 在仓库根的 prompt/ 下；本文件位于 app/agent/，向上两级到根。
# 全仓只此一份提示词文件（旧的 full / slim 变体与 PROMPT_VARIANT 开关已删，历史版本见
# prompt/archive/，仅供翻阅、不参与运行）。system_prompt 即原「冲 2k」版：<2000 tok、无 few-shot。
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompt"
_PROMPTS_PATH = _PROMPT_DIR / "prompts.yml"


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, Any]:
    with _PROMPTS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# curator 与 preference_parse 共用的「长期偏好字段规则」占位符——两者落同一张表，规则必须逐字
# 一致，曾各抄一份已开始措辞漂移。YAML 锚点没法插进 block scalar 中间，故用显式占位符 + replace
# （不用 str.format：prompt 里有 {"size": "42"} 这类字面大括号，format 会炸）。
_PREF_RULES_PLACEHOLDER = "<<PREF_FIELD_RULES>>"


def _inject_pref_rules(text: str) -> str:
    rules = str(_load_prompts().get("pref_field_rules", "")).rstrip()
    return text.replace(_PREF_RULES_PLACEHOLDER, rules)


def get_sub_agent_brief() -> str:
    """子 Agent 的「执行方通则」——fork 时拼进 demands 前缀，**不进 system prompt**。

    这段只有执行方（子 Agent）用得上，写进 system prompt 的话主 loop 每一步都要为它付 token 却
    永远用不上。挪到运行时的 human message 后：system prompt 对主 / 子仍逐字相同（同质 fork 硬
    约束不破、cache 前缀不破），但只有真正 fork 出去的子 Agent 才会读到它。
    """
    brief: str = _load_prompts().get("sub_agent_brief", "")
    return brief


def get_system_prompt() -> str:
    """主 / 子 AgentLoop 共用的**纯静态** system prompt（无任何运行时变量，逐字稳定）。

    **不注入 few-shot**：当前 system prompt 是 2k 版，为压 token 整段删掉了 ``<examples>``。
    ``app.agent.fewshot`` 模块仍在（评测飞轮的「高分轨迹蒸馏」一腿还用它产出示例），只是不再拼进
    system prompt——要恢复注入，得先给模板加回 ``<examples>`` 段。

    **长期偏好 / 近期行为历史 / 会话级 P_t 一律不在此注入**：三者每轮必变（偏好按本轮 query 语义
    裁剪、历史每轮收尾覆盖、P_t 每轮 curator 更新），混进 system prompt 会打断本该跨轮稳定的
    prompt cache 前缀。它们改由 ``main_agent._inject_runtime_context`` 拼进当轮 human message——
    那是缓存断点之后、永不缓存的部分，把「每轮必变」彻底隔离在缓存区外（对齐 refdocs/05 §4.4
    「按易变性分层，越易变越靠后」）。
    """
    return str(_load_prompts()["system_prompt"])


def get_planner_prompt() -> str:
    """planner 工具的提示词。"""
    return _load_prompts()["planner_prompt"]


def get_shopping_summary_prompt() -> str:
    """shopping_summary 工具的提示词。"""
    return _load_prompts()["shopping_summary_prompt"]


def get_memory_curator_prompt() -> str:
    """记忆管家（curator）的提示词——独立于购物工作流的偏好判定器。"""
    return _inject_pref_rules(_load_prompts()["memory_curator_prompt"])


def get_preference_parse_prompt() -> str:
    """把用户手填的一句话拆成结构化偏好条目（偏好页面的「添加」入口用）。"""
    return _inject_pref_rules(_load_prompts()["preference_parse_prompt"])
