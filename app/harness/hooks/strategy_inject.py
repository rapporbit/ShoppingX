"""成功策略的**注入位**与**结账位**（18-4）。

两个钩子一对，缺一不可：

- ``on_system_prompt``（装配期）：按本轮用户原话匹配在役策略，把 ``<learned_strategies>`` 块
  追加到主 Agent 的 system prompt 末尾，并记下注入了哪几条。
- ``on_session_end``（收尾）：拿本轮的成败给注入过的那几条结账——命中回血、连续失败淘汰。

**为什么注入进 system prompt 而不是像偏好那样每轮塞一条 system 消息。** 偏好那条必须等
planner 判出品类域才知道该给哪些（见 preference_inject），策略不需要——匹配依据是用户原话，
装配期就已经知道。而且策略要在**整轮的每一步**都生效（第 7 步决定收不收尾时同样该受
「预算陷阱先算到手价」约束），消息注入只影响它被塞进去的那一轮 think。

**为什么只给主 Agent（role="main"）。** worker 跑的是被收窄过的子任务，它那段专职 prompt 短
而具体，往里塞主 loop 的打法只会稀释。策略里凡是与检索有关的部分，主 Agent 会写进 demands。

**成败信号是粗的，而且刻意粗。** 判据只有一条：本轮调到了终结工具且最终回复非空。它不是
Rubric 分数——线上没有 judge，硬要在收尾时再调一次 LLM 打分，等于给每一轮加一次往返和一处
新的漂移源，还是拿抖动著称的尺子（记忆 rubric-judge-calibration-pitfalls）。这条粗信号只
负责抓「注入策略之后 Agent 干脆跑飞了」这种明确的坏，细粒度的判定留给离线门禁重放。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.agent.tool_registry import TERMINAL_TOOLS
from app.harness.middleware import harness_hook
from app.memory.strategies import (
    get_strategy_store,
    render_strategy_block,
    strategies_for_query,
)

logger = logging.getLogger("shoppingx.harness.strategy_inject")

#: 本轮注入了哪几条策略（dedup_key）。装配期写、收尾时读。**每轮必写**（没匹配上就写空元组），
#: 否则同进程连跑两轮时，第二轮会拿上一轮的清单去结账——记在错的策略头上比不记更糟。
_injected: ContextVar[tuple[str, ...]] = ContextVar("injected_strategies", default=())


def injected_strategy_keys() -> tuple[str, ...]:
    """本轮注入过的策略 key（测试与结账钩子用）。"""
    return _injected.get()


@harness_hook("on_system_prompt", name="strategy_inject", priority=50)
async def inject_strategies(context: dict[str, Any]) -> dict[str, Any] | None:
    """按本轮 query 匹配在役策略，追加进 system prompt 末尾。"""
    if context.get("role") != "main":
        return None
    query = str(context.get("query") or "")
    matched = await strategies_for_query(query)
    _injected.set(tuple(s.dedup_key for s in matched))
    block = render_strategy_block(matched)
    if not block:
        return None  # 一条都没匹配上 → 不塞空占位（省 token，也别给模型噪声）
    logger.info("注入 %d 条成功策略：%s", len(matched), ", ".join(s.dedup_key for s in matched))
    context.setdefault("append", []).append(block)
    return context


@harness_hook("on_session_end", name="strategy_feedback", priority=90)
async def settle_strategies(context: dict[str, Any]) -> dict[str, Any] | None:
    """给本轮注入过的策略结账。**只在主 loop 跑**（worker 从不注入，也就无账可结）。

    priority 90 排在输出审核（10）之后：审核可能把最终回复清成空串，那时这一轮该算失败——
    顺序反了就会把一次「输出全被判违规」记成成功。
    """
    keys = injected_strategy_keys()
    if not keys or current_fork_depth() >= 1:
        return None
    called = context.get("called_tools") or set()
    final = context.get("final_answer")
    success = bool(set(called) & TERMINAL_TOOLS) and bool(isinstance(final, str) and final.strip())
    await get_strategy_store().record_outcome(keys, success=success)
    return None
