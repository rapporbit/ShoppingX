"""塑形模型看到的上下文：偏好注入 / 成功策略注入与结账。

    on_system_prompt 50  strategy_inject     装配期按用户原话匹配在役策略，追加 <learned_strategies>
                                             （只给主 Agent）
    post_tool_call   50  preference_inject   planner 判出域后注入域内长期偏好
                                             （worker 由 task_dispatch 注入）
    on_session_end   90  strategy_feedback   给本轮注入过的策略结账：命中回血、连续失败淘汰

上下文压缩不在本仓做：交给框架 ``compress_context``（超阈值时 LLM 摘要进 ``state.summary``，
随 session.json 一起持久化）。cache_control 也不在这里打，落在 formatter 层。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.api.context import get_user_id
from app.harness.budgets import (
    TERMINAL_TOOLS,
)
from app.harness.middleware import harness_hook
from app.memory.injector import PREF_EMPTY, build_preference_block
from app.memory.strategies import get_strategy_store, render_strategy_block, strategies_for_query

logger = logging.getLogger("shoppingx.harness.context_shaping")


@harness_hook("post_tool_call", name="preference_inject", priority=50)
async def inject_domain_preferences(context: dict[str, Any]) -> dict[str, Any] | None:
    """planner 返回后注入域内长期偏好。一轮至多注入一次——阶段机保证 planner 只成功跑一次
    （跑完即离开 PLANNING，而 planner 不在后续阶段的白名单里）。"""
    if context.get("tool_name") != "planner" or current_fork_depth() >= 1:
        return None

    user_id = get_user_id() or ""
    if not user_id:
        return None  # 匿名用户没有长期偏好

    block = await build_preference_block(user_id)
    if not block or block == PREF_EMPTY:
        return None  # 本轮域内没有任何偏好 → 不塞空占位（省 token，也不给模型噪声）

    logger.info("注入域内长期偏好（planner 后）")
    context.setdefault("inject_messages", []).append(
        {
            "role": "system",
            "content": (
                "<user_long_term_preferences>\n"
                f"{block}\n"
                "</user_long_term_preferences>\n"
                "以上是该用户与**本轮品类相关**的长期偏好，已由系统自动生效（检索词与精挑打分里\n"
                "都已并入，见 memory.assemble）——**不要**再把它们转述进任何工具参数，重复一遍不会\n"
                "让它们更生效，只会让你替用户做了他没授权的决定。它们在这里只为一件事：让你在向\n"
                "用户解释「为什么选这几件」时，说得出是哪条偏好起了作用。"
            ),
        }
    )
    return context


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
