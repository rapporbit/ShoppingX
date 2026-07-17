"""pre_think 上下文治理：模型路由降级 + Cache-Breakpoint 压缩。

priority：

    20  budget_router      按剩余预算定档：换模型 / 注入 hint / 触发 fallback（Feedforward）
    90  context_compress   压缩历史视图 + 打缓存标记（Feedforward / Computational）

压缩排在最后：它要对**最终**送给模型的那份 messages 生效——前面所有 Hook（漂移纠正、断言纠正、
强制收尾、预算 hint）注入的内容都已经在列表里了，压缩看到的就是模型将看到的。

压缩只改「这一次请求送给模型的视图」，不改 state / checkpoint 里的原文——压缩是给模型的视图瘦身，
不是把历史删了。

**为什么 budget_router 取代了原来的 budget_hint。** 旧的 hint 只在「过软线」（成本 ≥ 80% 上限）
那一刻注入一句「省着点花」，然后就没有然后了——它是个提醒，不是控制。新的 router 把同一个位置
升级成四档路由（见 :mod:`app.agent.model_router`）：MINIMAL 档照样注入 hint（文案更硬），但同时
换便宜模型、并让 pre_tool_call 的预算闸提前收走成本放大器工具；FALLBACK 档直接不调 LLM。
软线（0.8）与 MINIMAL 分界（剩余 0.2）本就是同一条线，所以这不是新增一层，是把提醒换成了执行。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage

from app.agent import model_router
from app.agent.model_router import Tier
from app.compress.breakpoint import DEFAULT_KEEP_RECENT
from app.compress.compressor import DEFAULT_MAX_TOOL_TOKENS, mark_system_cache
from app.compress.pipeline import post_step_compress
from app.harness.middleware import harness_hook
from app.harness.state import GuardState
from app.observability import metrics
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.harness.compress")


def _compress_opts() -> tuple[int, int, bool]:
    return (
        env_int("COMPRESS_KEEP_RECENT", DEFAULT_KEEP_RECENT),
        env_int("COMPRESS_MAX_TOOL_TOKENS", DEFAULT_MAX_TOOL_TOKENS),
        env_bool("COMPRESS_CACHE_CONTROL", False),
    )


@harness_hook("pre_think", name="budget_router", priority=20)
async def route_by_budget(context: dict[str, Any]) -> dict[str, Any] | None:
    """按剩余预算定档，把决策写进 context 交给适配器执行（Hook 决策、适配器落地）。

    三个出口，都不在这里直接操作模型——Hook 拿不到 ``ModelRequest``：

    - ``model_override``：适配器 ``request.override(model=...)``（lite / minimal 档换便宜模型）
    - ``messages`` 追加 hint：minimal 档让模型自己也知道该收了
    - ``fallback_answer``：适配器**跳过模型调用**，直接把这段文本当 AIMessage 返回，loop 自然终止

    档位只降不升（成本单调增），所以每档只上报一次 metric——用 ``GuardState.last_tier`` 去重，
    否则一个 20 轮的任务会把 minimal 档记 15 次，降级率统计直接失真。

    全程走 ``model_router.xxx`` 而不是 ``from ... import xxx``：档位依赖全树成本，每次模型调用后都在
    变，必须现算；模块级引用也让单测能 monkeypatch 掉整条链（import 绑定的名字打不中）。
    """
    guard = context.get("_guard")
    tier = model_router.current_tier()

    entered_new_tier = not isinstance(guard, GuardState) or tier.label != guard.last_tier
    if isinstance(guard, GuardState) and entered_new_tier:
        if tier > Tier.MAIN:  # 只记降级，不记「留在 main」
            metrics.record_tier_change(tier.label)
            logger.info("预算降档：%s → %s", guard.last_tier, tier.label)
        guard.last_tier = tier.label

    if tier is Tier.MAIN:
        return None

    if tier is Tier.FALLBACK:
        # 连一次 LLM 调用都付不起了：用已有候选拼一个诚实的回答，不再进模型。
        context["fallback_answer"] = model_router.build_fallback_answer(
            context.get("original_query", "")
        )
        logger.warning("预算耗尽，走 fallback 规则兜底（不调 LLM）")
        return context

    model = model_router.tier_model(tier)
    if model is not None:
        context["model_override"] = model

    if tier is Tier.MINIMAL and entered_new_tier:
        # 只在**进入** minimal 那一轮注入：hint 经 persist_messages 落 state 后长驻历史，
        # 每轮重复注入只会攒出一摞相同提醒（且每次都斩断一次缓存前缀）。档位只降不升，
        # 「进入过」等价于「此后每轮都看得到」。
        messages = context.get("messages")
        if isinstance(messages, list):
            hint = SystemMessage(content=model_router.MINIMAL_HINT)
            messages.append(hint)
            context.setdefault("persist_messages", []).append(hint)
    return context


@harness_hook("pre_think", name="context_compress", priority=90)
async def compress_context(context: dict[str, Any]) -> dict[str, Any] | None:
    """压缩历史视图 + 给 system 段单独打缓存标记。"""
    messages = context.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    keep_recent, max_tool_tokens, enable_cache_control = _compress_opts()
    context["messages"] = post_step_compress(
        messages,
        keep_recent=keep_recent,
        max_tool_tokens=max_tool_tokens,
        enable_cache_control=enable_cache_control,
    )

    # LangChain 把 system prompt 放在 request.system_message（不在 messages 里），
    # apply_cache_control 够不到它。system prompt 纯静态、是全天不变的最长缓存层，单独打一个
    # cache_control 标记（对齐 refdocs/05 §4.4「system+tools 独立缓存层」）。
    if enable_cache_control:
        marked = mark_system_cache(context.get("system_message"))
        if marked is not None:
            context["system_message"] = marked
    return context
