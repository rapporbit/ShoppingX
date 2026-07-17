"""**只给主 loop 的第一轮**开 reasoning，其余轮次一律走快档。

**为什么单独拎出第一轮。** 主 loop 的轮次不是等价的：

- 第 1 轮是整条链路上**唯一没有被机制锁死**的决策——读懂 query（购物 / 闲聊 / 追问）、决定先拆解
  还是先查品类常识、单平台直搜还是跨平台 fork。选错这一步，后面每一步都在错误的轨道上跑得飞快。
- 第 2 轮之后，决策空间已经被阶段机白名单 + 候选 id 化夹死了：模型基本只是在「当前阶段允许的那
  一两个工具」里挑一个、把几个 item_id 传进去。实测那几轮吐出的 tool_call 参数只有 43~239 个字符，
  却各花 7~10 秒——那些 thinking token 买不到任何东西。

所以整条主 loop 是「第一轮想清楚，剩下的照做」，而不是「全程不动脑」。主 loop 的**基座**模型因此
是快档（``main_agent._build_main_agent`` 用 ``get_fast_llm``），第一轮由本 Hook 顶成 reasoning——
两者模型名相同、工具表不变，override 不打断 prompt cache 前缀。

（曾经这里还有一个前端可切的「深度思考」档：主 loop 每轮都开 reasoning。删掉了——它比默认档慢一倍，
而慢出来的那些 thinking 全花在第 2 轮之后那些「决策空间已被机制夹死」的轮次上，买不到东西。真要
全程开思考，去掉本 Hook 的 override 判断即可，不必为此在产品面上留一个按钮。）

**例外：复用轮（planner 判 ``retrieval=reuse``）连第一轮也不开。** 那一轮的编排是确定的——plan 已经
说了「不检索，直接在上一轮候选里精挑」，第一轮要做的那几个选择一个都不在。

**与预算降档的关系（priority 的由来）。** 本 Hook priority=10，跑在 ``budget_router``（20）之前，
于是预算见底时 ``budget_router`` 会用便宜档模型**覆盖**掉这里写的 reasoning 模型——钱不够的时候，
「想清楚」让位于「跑完」。反过来排序就成了「预算都烧穿了还在第一轮开思考」。

**只作用于主 loop（depth 0）**：fork 子 Agent 也有自己的 round_number=1，但子的活儿是「按 demands
搜一个平台」，没有编排决策可言（能力同质、授权不同质，见 fork-guardrails-mechanism-not-prompt），
恒为快档。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.agent.llm import get_llm
from app.api.context import get_retrieval_mode
from app.harness.middleware import harness_hook
from app.utils.env import env_bool

logger = logging.getLogger("shoppingx.harness.reasoning")

# 关掉即回到「全程零 reasoning」。留这个闸是为了能一行配置做 A/B：第一轮开思考的正当性来自编排
# 质量（而不是延迟——实测它多花 ~1.8s，恰好被「少绕一轮」抵掉），要能随时关掉对照。
BOOST_ENABLED = env_bool("FAST_MODE_REASONING_FIRST_ROUND", True)


@harness_hook("pre_think", name="reasoning_boost", priority=10)
async def boost_first_round(context: dict[str, Any]) -> dict[str, Any] | None:
    """主 loop 第一轮 → 把本次模型调用换成 reasoning 模型（其余轮次维持基座的快档）。"""
    if not BOOST_ENABLED:
        return None
    if context.get("round_number") != 1 or current_fork_depth() >= 1:
        return None
    # 复用轮（planner 判 reuse）连第一轮也不开：plan 已经写死「不检索，直接在上一轮候选里精挑」，
    # 阶段机也已把 SEARCHING 跳过（hooks/phase_transition）。第一轮值得开思考，是因为它要在
    # 「单平台直搜 / 跨平台 fork / 先查品类常识」之间做选择——reuse 轮这些分支一个都不在。
    # 依赖 planner 开局预置（agent_middleware.abefore_agent）跑在 pre_think 之前；预置降级时这里
    # 读到默认值 search → 照常开 reasoning，落在安全侧。
    if get_retrieval_mode() == "reuse":
        logger.debug("复用轮：第一轮不开 reasoning（编排已由 plan 定死）")
        return None

    context["model_override"] = get_llm()
    logger.debug("主 loop 第一轮开 reasoning（编排决策轮）")
    return context
