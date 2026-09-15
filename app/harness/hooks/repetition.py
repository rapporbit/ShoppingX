"""重复调用：两种「模型在打转」的机制性止损。

    pre_tool_call   48  tool_breaker_gate    工具级熔断（**必须是最后一道**：allow() 有副作用）
    post_tool_call   5  tool_breaker_record  记成功 / 失败
    post_tool_call  20  result_nudges        LoopDetector 滑窗计数
                                             + 按优先级追加**至多一条**系统提示

**提示优先级链**（互斥）：越树检索「强制收敛」 > item_search 的 filtered_out 证据
> item_picker 收尾提示 > 循环提示。收尾排在循环之前：精选已就绪时催收尾比催换思路更对。
``result_nudges`` 必须晚于 ``safety.truncate_result``(10)，否则刚贴上的提示会被截掉。

这里曾有 tool_memo_replay / tool_memo_record 一对（同参数幂等工具回放上次结果、不真执行）。
2026-09-15 删：round3 后普通轮只跑 1 次 item_search，autopick 不走 pre_tool_call，回放几乎命不中；
而它是绕过 post_tool_call 的旁路（要自己喂 LoopDetector、缓存要避开尾部通告），复杂度不抵收益。
同参重复由 LoopDetector 提示 + 检索预算硬挡兜底。
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.sentinels import (
    SUMMARY_NUDGE,
    converge_directive,
    tool_breaker_open,
)
from app.harness.state import guard_of
from app.utils import shared_breaker
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.harness.repetition")


def _filtered_out_note(context: dict[str, Any], tool_name: str) -> str | None:
    """item_search 探测到「库里有货、只是被硬条件挡了」时的提示语。

    要治的是一个 P0 级误报：候选池空 / 很薄时，模型默认走「空召回硬路径」如实说「没找到」——
    但真相常常是「库里有，只是都超了你的预算 20 美元」。这两句话对用户的价值天差地别，而在
    工具返回体里它们长得一模一样。探测召回把证据拿了回来，这里负责让模型**必须**看见它。

    提示只给动机与事实，不改机制：放不放宽预算是用户的决定，模型该问（ask_user）而不是自己松。
    """
    if tool_name != "item_search":
        return None
    blocked = context.get("call_filtered_out")
    if not isinstance(blocked, list) or not blocked:
        return None
    sample = blocked[0] if isinstance(blocked[0], dict) else {}
    title = str(sample.get("title", ""))[:60]
    reason = str(sample.get("reason", ""))
    head = (
        f"库里其实还有 {len(blocked)} 件相关商品，是被本次检索的硬条件挡在候选池外的"
        f"（例：{title} —— {reason}）。"
    )
    if context.get("call_filtered_price_only"):
        return head + (
            "**不要对用户说「没找到这类商品」**——如实说「符合的都在预算之外，最低约 $X」，"
            "并用 ask_user 问是否放宽预算 / 换个方向；不要自作主张放宽用户给的预算。"
        )
    return head + (
        "**不要对用户说「没找到这类商品」**——如实说明是被哪个条件（排除偏好 / 评分门槛）"
        "筛掉的，让用户自己决定要不要松这一条。"
    )


@harness_hook("post_tool_call", name="result_nudges", priority=20)
async def append_nudges(context: dict[str, Any]) -> dict[str, Any] | None:
    """循环检测 + 按优先级追加**至多一条**系统提示。"""
    guard = guard_of(context)
    if guard is None:
        return None

    tool_name = context.get("tool_name", "")
    # 循环检测对每次真实执行的工具都记一笔（被闸拦下的哨兵不走 post_tool_call，天然不计）。
    # 带回新候选的检索是产出性重试、不计入打转阈值（call_candidates 数的是渲染层的 fresh 批，
    # 与池内重复的那部分已折叠成 id、不在其中——见 item_search.known_ids / _count_candidates）。
    looped = guard.detector.record(tool_name, progressed=context.get("call_candidates", 0) > 0)

    result = context.get("tool_result")
    if not isinstance(result, str):
        return None

    converge_count = context.get("converge_count")
    filtered_note = _filtered_out_note(context, tool_name)

    if converge_count is not None:
        suffix = "\n\n[系统提示] " + converge_directive(converge_count)
    elif filtered_note is not None:
        # 诚实证据（别把「有货但超预算」说成「没货」）——丢了它直接踩 P0。
        suffix = "\n\n[系统提示] " + filtered_note
    elif tool_name == "item_picker":
        suffix = "\n\n" + SUMMARY_NUDGE
    elif looped:
        suffix = "\n\n[系统提示] " + guard.detector.nudge_message(tool_name)
        logger.info("LoopDetector 命中：%s 短时间内重复调用", tool_name)
    else:
        return None

    context["tool_result"] = result + suffix
    return context


TOOL_BREAKER_ENABLED = env_bool("HARNESS_TOOL_BREAKER", True)
_FAILURE_THRESHOLD = env_int("HARNESS_TOOL_BREAKER_THRESHOLD", 3)
_RECOVERY_TIMEOUT = float(env_int("HARNESS_TOOL_BREAKER_RECOVERY_SEC", 60))

# 每个工具一个断路器，进程级共享（跨会话累积失败——某平台 API 挂了就是挂了，不该每个会话重新试
# 三次）。``all_breakers()`` 会枚举它们，metrics 里能看到状态。
_breakers: dict[str, CircuitBreaker] = {}


def get_tool_breaker(tool_name: str) -> CircuitBreaker:
    """取（或懒建）某个工具的断路器。"""
    breaker = _breakers.get(tool_name)
    if breaker is None:
        breaker = CircuitBreaker(
            f"tool:{tool_name}",
            failure_threshold=_FAILURE_THRESHOLD,
            recovery_timeout=_RECOVERY_TIMEOUT,
        )
        _breakers[tool_name] = breaker
    return breaker


def reset_tool_breakers() -> None:
    """清空所有工具断路器（测试用）。"""
    for breaker in _breakers.values():
        breaker.reset()
    _breakers.clear()


@harness_hook("pre_tool_call", name="tool_breaker_gate", priority=48)
async def check_tool_breaker(context: dict[str, Any]) -> dict[str, Any] | None:
    """断路器 OPEN 且未到恢复窗口 → 快速失败，工具不执行。

    ``allow()`` **有副作用**（可能把 OPEN 推进到 HALF_OPEN 以放行一次探测），放行后必须成对地
    记一次成败，否则半开探测悬空、断路器再也回不到 CLOSED。

    所以本闸的 priority 必须是 pre_tool_call 里**最后一个**（48 > retrieval_charge 的 45）：排在
    前面的话，深度闸 / 阶段门 / 预算闸任何一道后置拒绝，都会让这次「已放行的探测」没有对应的成败
    记录。放在最后 → 只要它放行，工具就一定执行，成败一定会被记上。

    代价：被本闸拒绝时，前面 retrieval_charge 的检索计数已经自增了一次。这是可接受的——熔断本就是
    异常路径，且少算一次检索额度只会让 Agent 更早收敛，方向是安全的。
    """
    if not TOOL_BREAKER_ENABLED:
        return None
    tool_name = context.get("tool_name", "")
    if not tool_name:
        return None

    breaker = get_tool_breaker(tool_name)
    # 走 shared_breaker 而不是直接 ``breaker.allow()``：``BREAKER_SHARED=0``（默认）时它就是
    # 后者的透明转发，开着时多问一次 Redis —— 让别的副本已经踩满的熔断在本副本立即生效。
    if not await shared_breaker.allow(breaker):
        logger.warning("工具 %s 处于熔断态，快速失败", tool_name)
        raise HookRejectSignal(tool_breaker_open(tool_name), raw=True)
    context["_breaker_armed"] = tool_name  # 已放行：适配器/下游必须成对记一次成败
    return context


@harness_hook("post_tool_call", name="tool_breaker_record", priority=5)
async def record_tool_outcome(context: dict[str, Any]) -> dict[str, Any] | None:
    """工具正常返回 → 记一次成功（复位失败计数 / 半开探测成功即恢复 CLOSED）。

    失败路径不在这里：工具抛异常时 post_tool_call 根本不会跑，由适配器捕获异常后
    调 ``record_failure`` 并重新抛出。
    """
    if not TOOL_BREAKER_ENABLED:
        return None
    tool_name = context.get("tool_name", "")
    if tool_name:
        await shared_breaker.record_success(get_tool_breaker(tool_name))
    return None
