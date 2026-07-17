"""Harness Hook Pipeline：Agent 全生命周期的统一治理入口。

6 个 Hook 点覆盖 Agent 完整生命周期：

    on_session_start → pre_think → pre_tool_call → post_tool_call → post_reflect → on_session_end

每个 Hook 接收 context dict、返回（可能修改过的）context dict 或 None（不修改）。
Hook 按 priority 升序执行（低 priority 先执行），单个 Hook 异常不中断整个 Pipeline。
pre_tool_call Hook 可 raise ``HookRejectSignal`` 拒绝当前工具调用。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from app.observability import metrics

logger = logging.getLogger("shoppingx.harness")

#: 同一硬闸对同一目标（escape_key）累计拒绝达到该次数后，后续调用放行（逃生门阈值）。
ESCAPE_AFTER_REJECTS = 2

HookFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]

HOOK_POINTS: list[str] = [
    "on_session_start",
    "pre_think",
    "pre_tool_call",
    "post_tool_call",
    "post_reflect",
    "on_session_end",
]


class HookRejectSignal(Exception):
    """pre_tool_call Hook 抛出此异常 → Pipeline 立即停止后续 Hook、拒绝本次工具调用。

    ``raw=True`` 表示 ``reason`` 本身就是一条**写给模型看的完整哨兵文案**（见
    :mod:`app.harness.sentinels`）：适配器原样回给模型，不加 ``[Harness 拒绝]`` 前缀——那些文案
    仔细写过「为什么被拒 + 你现在该干什么」，加前缀只会稀释指令。``raw=False``（默认）用于结构化
    的短理由（如阶段门），适配器加前缀标明来源。

    **逃生门（效率闸专用）**：``escape_key`` 非 None 即接入统一逃生机制（见
    :func:`_try_escape`）——同一 (gate, escape_key) 连拒达到 :data:`ESCAPE_AFTER_REJECTS` 次后
    放行本次调用，``on_escape`` 在放行时执行配套回退（如阶段回滚）。安全闸（白名单/深度/
    预算/终结）**不得声明** ``escape_key``，它们的判定不会「立错墙」，必须永远硬。
    """

    def __init__(
        self,
        reason: str = "",
        *,
        raw: bool = False,
        escape_key: str | None = None,
        on_escape: Callable[[], None] | None = None,
    ) -> None:
        self.reason = reason
        self.raw = raw
        self.escape_key = escape_key
        self.on_escape = on_escape
        super().__init__(reason)


class HarnessMiddleware:
    """Agent Harness 的统一 Hook Pipeline。

    - Hook 按注册 priority 升序依次执行（同 priority 保持注册顺序）
    - 单个 Hook 异常不中断整个 Pipeline（catch + log），除 ``HookRejectSignal``
    - Hook 可修改 context（如截断工具返回、注入 hint）
    - pre_tool_call Hook 可 raise ``HookRejectSignal`` 拒绝工具调用
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[tuple[str, HookFn, int]]] = defaultdict(list)

    def register(self, hook_point: str, name: str, fn: HookFn, *, priority: int = 100) -> None:
        if hook_point not in HOOK_POINTS:
            raise ValueError(f"未知 Hook 点: {hook_point}，可选: {HOOK_POINTS}")
        self._hooks[hook_point].append((name, fn, priority))
        self._hooks[hook_point].sort(key=lambda t: t[2])

    def list_hooks(self, hook_point: str | None = None) -> list[tuple[str, str, int]]:
        """列出已注册 Hook，返回 ``[(hook_point, name, priority), ...]``。"""
        if hook_point is not None:
            return [(hook_point, n, p) for n, _, p in self._hooks.get(hook_point, [])]
        return [(hp, n, p) for hp, hooks in self._hooks.items() for n, _, p in hooks]

    async def run(self, hook_point: str, context: dict[str, Any]) -> dict[str, Any]:
        """依次执行 ``hook_point`` 上注册的所有 Hook，返回最终 context。

        - Hook 返回 ``None`` → context 不变
        - Hook 返回 dict → 替换 context
        - Hook raise ``HookRejectSignal`` → 设置 ``_rejected`` 并立即返回（仅 pre_tool_call 有意义）
        - 其它异常 → 记日志并继续执行后续 Hook

        **fail-open 是自觉取舍**：治理 Hook 自身出 bug 时宁可放行这次调用，也不让控制面拖垮
        主链路。对安全层（L1 白名单 / L3 过滤）同样成立——它们的判定是纯集合 / 正则查找，
        异常面近零；将来若引入带外部依赖（IO / LLM）的安全 Hook，须在 Hook 内部自带
        fail-closed（catch 后主动 raise ``HookRejectSignal``），不能指望 Pipeline 兜。
        """
        hooks = self._hooks.get(hook_point, [])
        for name, fn, _priority in hooks:
            t0 = time.monotonic()
            try:
                result = await fn(context)
                if result is not None:
                    context = result
            except HookRejectSignal as sig:
                if hook_point == "pre_tool_call" and _try_escape(name, sig, context):
                    continue  # 逃生放行：跳过本闸，后续 Hook 照常执行
                metrics.record_gate_event(name, "reject")
                context["_rejected"] = True
                context["_reject_reason"] = sig.reason
                context["_reject_raw"] = sig.raw
                context["_rejected_by"] = name
                logger.info("Hook [%s] rejected at %s: %s", name, hook_point, sig.reason[:80])
                return context
            except Exception:
                logger.error("Hook [%s] at %s 执行异常，跳过", name, hook_point, exc_info=True)
                continue
            elapsed = (time.monotonic() - t0) * 1000
            if elapsed > 100:
                logger.info("Hook [%s] at %s took %.0fms", name, hook_point, elapsed)
        return context


def _try_escape(gate: str, sig: HookRejectSignal, context: dict[str, Any]) -> bool:
    """效率闸的确定性逃生门：同一 (gate, escape_key) 连拒达阈值后放行，不依赖模型配合。

    硬闸分两类。**安全闸**（白名单/深度/预算/终结/子搜上限）的判定是精确事实，永远硬拒，
    不接入本机制；**效率闸**（postfork 直搜/websearch 动机闸）依据的是上游判定
    （planner 意图、「fork 即检索阶段」语义），而上游判定是假设不是承诺——2026-07-14 线上
    死锁即 planner 把换品类误判成 reuse，彼时的阶段白名单闸拦死 item_search 27 轮直到用户
    取消（该白名单已在重构第三段拆除，reuse 改走小预算，见 tool_gates）。模型对
    同一堵墙的反复坚持本身就是「墙可能立错了」的强信号，达到阈值即认输放行。

    放行是**闩锁语义**（计数不清零）：同 key 首次逃生后，后续命中直接放行。模型赢过一次说明
    这堵墙大概率立错了位置，每次放行前再攒 2 轮拒绝纯烧延迟（每轮一次 LLM 往返）。放开的只是
    效率禁令——检索预算、token 预算、深度等安全闸照常生效，liveness 看门狗是最终兜底。

    ``on_escape`` 回退动作的异常不冒泡（照常放行）：逃生门是为了解死锁，回退挂了不能反而把
    调用拦回去。
    """
    if sig.escape_key is None:
        return False
    guard = context.get("_guard")
    counts = getattr(guard, "gate_reject_counts", None)
    if counts is None:
        return False  # 无 GuardState（异常路径）退回旧行为：始终拒绝
    key = f"{gate}:{sig.escape_key}"
    # 批次原子化：同一条 AI 消息的并行调用（同 think_step）按**批次起点**的计数统一裁决，
    # 且同批多次拒绝只记一次「连拒」（4 个并行调用是一次决策，不是四次坚持）。否则前几个
    # 被拒攒计数、后几个触发逃生——谁被拦由到达顺序决定，与需求无关（badcase 4c0ac682）。
    step = getattr(guard, "think_step", -1)
    if guard.escape_snapshot_step != step:
        guard.escape_snapshot_step = step
        guard.escape_snapshot = dict(counts)
    rejects = guard.escape_snapshot.get(key, 0)
    if rejects < ESCAPE_AFTER_REJECTS:
        counts[key] = rejects + 1  # 同批幂等：按快照+1，不随批内调用数累加
        return False
    logger.warning(
        "硬闸逃生：%s 对 %s 连拒 %d 次后放行（疑似上游判定失准）", gate, sig.escape_key, rejects
    )
    metrics.record_gate_event(gate, "escape")
    if sig.on_escape is not None:
        try:
            sig.on_escape()
        except Exception:
            logger.error("逃生回退动作异常（gate=%s），仍放行本次调用", gate, exc_info=True)
    return True


# 全局单例——所有 Hook 注册到这里，整个进程共用。
harness = HarnessMiddleware()


def harness_hook(hook_point: str, *, name: str, priority: int = 100) -> Callable[[HookFn], HookFn]:
    """装饰器：自动注册 Hook 到全局 ``harness`` 单例。"""

    def decorator(fn: HookFn) -> HookFn:
        harness.register(hook_point, name, fn, priority=priority)
        return fn

    return decorator
