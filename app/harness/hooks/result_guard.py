"""post_tool_call 结果守卫：截断 + 循环检测 + 分级提示 + 终结标记（Feedback / Computational）。

工具**执行之后**、结果回到模型之前的加工。Hook 改写 ``context["tool_result"]``，适配器负责把它
写回 ToolMessage。

priority：

    10  truncate_result   过长结果尾部截断（先截断，后追加提示——否则提示会被截掉）
    20  result_nudges     循环检测 + 分级提示（优先级链见下）
    30  mark_terminal     主 loop 真实执行了终结工具 → 置位，令后续工具被 terminal_reached_gate 拦

**提示优先级链**（互斥，只追加最靠前的一条——都贴上去会互相稀释）：

    越树检索预算「强制收敛」 > per-sub 检索「预算可见」批注 > item_picker 收尾提示 > 循环提示

收尾提示排在循环提示之前：精选已就绪时，催收尾比催换思路更对。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.harness.budgets import SUB_ITEM_SEARCH_CAP, TERMINAL_TOOLS
from app.harness.middleware import harness_hook
from app.harness.sentinels import SUMMARY_NUDGE, converge_directive, sub_search_budget_note
from app.harness.state import GuardState
from app.harness.truncation import truncate_tool_result

logger = logging.getLogger("shoppingx.harness.result_guard")


def _state(context: dict[str, Any]) -> GuardState | None:
    guard = context.get("_guard")
    return guard if isinstance(guard, GuardState) else None


@harness_hook("post_tool_call", name="truncate_result", priority=10)
async def truncate_result(context: dict[str, Any]) -> dict[str, Any] | None:
    """工具返回过长时按 token 预算截断并留提示。

    必须排在 ``result_nudges`` 之前：先截断、再追加系统提示，否则刚贴上的提示会被截掉。
    """
    guard = _state(context)
    result = context.get("tool_result")
    if guard is None or not isinstance(result, str):
        return None
    truncated = truncate_tool_result(result, guard.max_tool_tokens)
    if truncated != result:
        context["tool_result"] = truncated
        return context
    return None


def _budget_note(guard: GuardState, tool_name: str) -> str | None:
    """「预算可见」批注：只对子（depth≥1）的 item_search 给；主 loop 由全树预算的强制收敛管。"""
    if tool_name != "item_search" or current_fork_depth() < 1:
        return None
    return sub_search_budget_note(guard.item_search_calls, SUB_ITEM_SEARCH_CAP)


@harness_hook("post_tool_call", name="result_nudges", priority=20)
async def append_nudges(context: dict[str, Any]) -> dict[str, Any] | None:
    """循环检测 + 按优先级追加**至多一条**系统提示。"""
    guard = _state(context)
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

    converge_note = context.get("converge_note")
    converge_count = context.get("converge_count")
    budget_note = _budget_note(guard, tool_name)

    if converge_note is not None:
        # 复用轮小预算的软线文案（tool_gates.charge_retrieval 填入），比通用收敛指令更具体。
        suffix = "\n\n[系统提示] " + converge_note
    elif converge_count is not None:
        suffix = "\n\n[系统提示] " + converge_directive(converge_count)
    elif budget_note is not None:
        suffix = "\n\n[系统提示] " + budget_note
    elif tool_name == "item_picker":
        suffix = "\n\n" + SUMMARY_NUDGE
    elif looped:
        suffix = "\n\n[系统提示] " + guard.detector.nudge_message(tool_name)
        logger.info("LoopDetector 命中：%s 短时间内重复调用", tool_name)
    else:
        return None

    context["tool_result"] = result + suffix
    return context


@harness_hook("post_tool_call", name="mark_terminal", priority=30)
async def mark_terminal(context: dict[str, Any]) -> dict[str, Any] | None:
    """本次是主 loop 真实执行的终结工具 → 置位，令后续工具被终结硬停闸拦下。

    只在工具真执行（过了各闸）后调，故被深度闸/预算闸拦掉的终结调用不会误置位。
    """
    guard = _state(context)
    if guard is None:
        return None
    if current_fork_depth() == 0 and context.get("tool_name") in TERMINAL_TOOLS:
        guard.terminal_reached = True
    return None
