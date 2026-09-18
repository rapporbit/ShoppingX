"""终结：让循环在该停的时候停下（refdocs 14 <termination> 是 P0——Agent 最常见的失败是不收尾）。

    pre_think        5  liveness_watchdog     停滞 → 收敛指令 → 硬停交部分结果
    pre_tool_call    5  terminal_reached_gate 本轮已调过终结工具 → 拦下一切后续工具
    post_tool_call  30  mark_terminal         终结工具真实执行后置位，令上面那道闸生效
    post_reflect    60  terminal_enforcer     纯文字收尾、没调终结工具 → 当场重发模型
                                              （配额 per-call）

四个钩子是一条链：mark_terminal 置位 → terminal_reached_gate 拦；terminal_enforcer 催软的，
看门狗兜硬的。硬停通路本身在适配器 ``on_model_call``（直接合成收尾消息、置 terminal_reached）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.harness.budgets import (
    MAX_TERMINAL_NUDGE_RETRIES,
    TERMINAL_TOOLS,
    is_terminal_call,
)
from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.msgs import system_message
from app.harness.sentinels import (
    TERMINAL_REACHED_DENIED,
    TERMINAL_TOOL_NUDGE,
)
from app.harness.signals import candidate_count
from app.harness.state import GuardState, guard_of
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.harness.termination")


@harness_hook("pre_tool_call", name="terminal_reached_gate", priority=5)
async def check_terminal_reached(context: dict[str, Any]) -> dict[str, Any] | None:
    """终结硬停（over-loop 治理）：本轮已调过终结工具收尾 → 之后任何工具一律拦下。

    断掉「调完 shopping_summary 又 item_search / 再 picker」的打转尾巴，逼模型直接输出收尾文案。

    **批次原子化**（同 ``middleware.py`` 逃生门的口径）：同一条 AI 消息里并行发出的终结工具放行。
    模型说「这两件分开下两个单」时会同轮发两个 ``create_order``，框架并发跑；布尔位下谁先跑完
    post 就把兄弟调用拦死，第二张确认卡静默消失、收尾文案却照说「已生成两张」（2026-09-16 实测）。
    这类兄弟调用是**一次决策**，不是收尾后的新动作。放行只开给终结工具本身：同批的
    ``shopping_summary`` + ``item_search`` 照拦（那才是打转），下一个 think_step 再调也照拦。
    """
    guard = guard_of(context)
    if guard is None:
        return None
    if guard.terminal_reached:
        same_batch = guard.terminal_step >= 0 and guard.think_step == guard.terminal_step
        if same_batch and is_terminal_call(context.get("tool_name"), context.get("tool_args")):
            return None
        raise HookRejectSignal(TERMINAL_REACHED_DENIED, raw=True)
    return None


@harness_hook("post_tool_call", name="mark_terminal", priority=30)
async def mark_terminal(context: dict[str, Any]) -> dict[str, Any] | None:
    """本次是真实执行的终结工具 → 置位，令后续工具被终结硬停闸拦下。

    只在工具真执行（过了各闸）后调，故被预算闸拦掉的终结调用不会误置位。
    一并记下批次号（当前 ``think_step``），供硬停闸放行同批的兄弟终结调用。
    """
    guard = guard_of(context)
    if guard is None:
        return None
    if is_terminal_call(context.get("tool_name"), context.get("tool_args")):
        guard.terminal_reached = True
        guard.terminal_step = guard.think_step
    return None


@harness_hook("post_reflect", name="terminal_enforcer", priority=60)
async def enforce_terminal(context: dict[str, Any]) -> dict[str, Any] | None:
    """模型没调工具就想收尾、且**本轮**从未调过终结工具 → 请适配器重发一次模型。

    「本轮调过哪些工具」只认 ``called_tools``（``HarnessSession`` 每轮新建，工具真执行成功
    才记——被闸拒绝的、返回 ERROR 的都不算）。此前这里是自己去扫 ``messages`` 找 tool_result
    （审查报告 P1-4：同一事实四个来源），**那个来源在续聊轮是错的**：messages 含恢复回来的
    历史，上一轮调过 shopping_summary，这一轮模型空口收尾也会被判成「调过了」而放行。
    """
    guard = context.get("_guard")
    if not isinstance(guard, GuardState):
        return None
    if guard.terminal_nudge_retries >= MAX_TERMINAL_NUDGE_RETRIES:
        return None  # 模型持续不听指令时不无限重试

    if context.get("response_has_tool_calls"):
        return None  # 它还在调工具，loop 会继续，不需要催
    if context.get("response_ai_message") is None:
        return None
    called: set[str] = context.get("called_tools", set())
    if called & TERMINAL_TOOLS:
        return None  # 本轮已调过终结工具，这是它之后的自然收尾文字，正常放行
    if guard.terminal_reached:
        # 适配器合成的收尾（预算 FALLBACK 档 / 看门狗硬停）不经工具、但已置位 terminal_reached。
        # AgentScope 里 on_model_call 在 on_reasoning 内层，合成回复后 post_reflect 照跑——
        # 这里若只认 called_tools 就会催重发，on_reply 吞 ReplyEnd 再合成同一段，空转到 max_iters。
        return None

    guard.terminal_nudge_retries += 1
    context["retry_nudge"] = TERMINAL_TOOL_NUDGE
    logger.info("模型未调终结工具就想收尾，追加提示重发一次")
    return context


WATCHDOG_STALL_SEC = env_int("WATCHDOG_STALL_SEC", 45)
WATCHDOG_GRACE_SEC = env_int("WATCHDOG_GRACE_SEC", 30)

_CONVERGE_NOTICE = (
    "[系统看门狗] 任务已较长时间没有实质进展。请立即停止当前方向的重试，"
    "就用手头已有的信息收尾：\n"
    "- 已有候选 → 立刻调 shopping_summary 给出清单，如实说明未完成的部分与原因；\n"
    "- 没有候选或非购物请求 → 立刻调 chat_fallback 如实说明目前做不到、建议用户怎么调整。\n"
    "除这两个终结工具外，不要再调用其他工具。"
)


def _partial_answer() -> str:
    """硬停时交给用户的部分结果——如实报告进展到哪、建议怎么重试。"""
    n = candidate_count()
    if n > 0:
        return (
            "抱歉，这个请求处理了很久仍未收敛，为免让你干等，我先停在这里。\n\n"
            f"目前进展：已检索到 {n} 件候选商品，但还没完成按你条件的精挑与最终清单。\n"
            "你可以把需求说得更具体一点（明确品类、预算、必须满足的条件），"
            "或者拆成几个小问题再发给我，我会重新处理。"
        )
    return (
        "抱歉，这个请求处理了很久仍未取得实质进展，为免让你干等，我先停在这里。\n"
        "换个说法或把需求拆小一点再试一次，我会重新处理。"
    )


@harness_hook("pre_think", name="liveness_watchdog", priority=5)
async def check_liveness(context: dict[str, Any]) -> dict[str, Any] | None:
    """每次唤起模型前查一次停滞时长。"""
    guard = context.get("_guard")
    if not isinstance(guard, GuardState):
        return None

    now = time.monotonic()
    if guard.last_progress_at <= 0:
        guard.last_progress_at = now  # 开表：从第一次 Think 起算
        return None

    stall = now - guard.last_progress_at
    if stall < WATCHDOG_STALL_SEC:
        guard.watchdog_nudged_at = 0.0  # 有过进展即解除武装（与 awrap_tool_call 的复位互为冗余）
        return None

    if guard.watchdog_nudged_at <= 0:
        guard.watchdog_nudged_at = now
        logger.warning("看门狗：%d 秒无实质进展，注入强制收敛指令", int(stall))
        context["messages"] = [*context["messages"], system_message(_CONVERGE_NOTICE, context)]
        return context

    if now - guard.watchdog_nudged_at < WATCHDOG_GRACE_SEC:
        return None

    logger.error(
        "看门狗：收敛指令后 %d 秒仍无进展，硬停交部分结果（停滞共 %d 秒）",
        int(now - guard.watchdog_nudged_at),
        int(stall),
    )
    context["fallback_answer"] = _partial_answer()
    return context
