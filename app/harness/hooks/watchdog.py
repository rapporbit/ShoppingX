"""liveness 看门狗（pre_think）：停滞过久 → 强制收敛指令 → 宽限后仍无进展 → 硬停交部分结果。

harness 重构第一段（对应 2026-07-14 线上死锁复盘的原则）：**系统永远要有产出**——
「慢 + 一个诚实的部分答案」永远好过「卡死 + 空白」。逃生门（phase_check）治的是已知的
死锁形态，看门狗兜的是**所有还没被发现的**停滞：它不关心卡在哪道闸、哪个判定，只看
「多久没有实质进展」这一个系统自己可观测的信号，不依赖模型配合。

「实质进展」的口径 = 工具**真实执行**成功（awrap_tool_call 里更新 ``last_progress_at``）。
被闸拦下的哨兵、tool_memo 的回放都走 early-return，天然不算——模型撞墙撞得再勤也不续命。

两级动作（都只在主 loop，子 Agent 有自己的 90s 超时）：
1. 停滞 ≥ ``WATCHDOG_STALL_SEC``：往本次模型调用注入收敛指令，指路终结工具
   （与 terminal_enforcer 的纪律一致，不会被「没调终结工具」的纠正回路顶回来）。
2. 指令后宽限 ``WATCHDOG_GRACE_SEC`` 仍无进展：置 ``fallback_answer`` 走预算耗尽同款
   硬停通路（awrap_model_call 直接合成收尾消息、置 terminal_reached，不再唤模型）。

与 ``MAIN_AGENT_TIMEOUT_SEC``（300s，报错收场）的关系：看门狗在远早于它的位置给用户一个
**体面的**部分结果；外层超时降级为最后的保险丝，正常情况下永远不该被摸到。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import SystemMessage

from app.agent.fork_guard import current_fork_depth
from app.harness.middleware import harness_hook
from app.harness.signals import candidate_count
from app.harness.state import GuardState
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.harness.watchdog")

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
    """每次唤起模型前查一次停滞时长。仅主 loop（depth 0）。"""
    if current_fork_depth() >= 1:
        return None
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
        context["messages"] = [*context["messages"], SystemMessage(content=_CONVERGE_NOTICE)]
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
