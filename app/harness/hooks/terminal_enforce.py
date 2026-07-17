"""post_reflect 终结纪律：不让「没调终结工具就直接吐文字收尾」蒙混过关。

P0 收尾纪律，机制兜底而非只靠 prompt——同款失败已在真实跑测中复现：模型直接输出对比文案就把
loop 停了，没走 shopping_summary / chat_fallback，前端拿不到商品卡。

**为什么必须在本轮当场重发模型，而不能像别的 Hook 那样注入 inject_messages 等下一轮**：模型这次
没产出任何 tool_call，Agent 框架据此判定「该结束了」——根本不会有下一轮。所以本 Hook 只置一个
``retry_nudge``，由适配器在同一次 ``awrap_model_call`` 里追加提示、**重发一次模型**。

子 Agent（depth≥1）不受影响——它们的正常收尾本就是直接吐文字，不该被拉去调不属于自己权限的终结
工具（会被深度闸拦、变死循环）。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import ToolMessage

from app.agent.fork_guard import current_fork_depth
from app.harness.budgets import MAX_TERMINAL_NUDGE_RETRIES, TERMINAL_TOOLS
from app.harness.middleware import harness_hook
from app.harness.sentinels import TERMINAL_TOOL_NUDGE
from app.harness.state import GuardState

logger = logging.getLogger("shoppingx.harness.terminal")


def _has_terminal_tool_call(messages: Any) -> bool:
    """本轮消息历史里是否已经真实调用过终结工具（ToolMessage.name 命中）。"""
    if not messages:
        return False
    return any(isinstance(m, ToolMessage) and m.name in TERMINAL_TOOLS for m in messages)


@harness_hook("post_reflect", name="terminal_enforcer", priority=60)
async def enforce_terminal(context: dict[str, Any]) -> dict[str, Any] | None:
    """模型没调工具就想收尾、且本轮从未调过终结工具 → 请适配器重发一次模型。"""
    if current_fork_depth() >= 1:
        return None

    guard = context.get("_guard")
    if not isinstance(guard, GuardState):
        return None
    if guard.terminal_nudge_retries >= MAX_TERMINAL_NUDGE_RETRIES:
        return None  # 模型持续不听指令时不无限重试

    if context.get("response_has_tool_calls"):
        return None  # 它还在调工具，loop 会继续，不需要催
    if context.get("response_ai_message") is None:
        return None
    if _has_terminal_tool_call(context.get("messages")):
        return None  # 之前已调过终结工具，这是它之后的自然收尾文字，正常放行

    guard.terminal_nudge_retries += 1
    context["retry_nudge"] = TERMINAL_TOOL_NUDGE
    logger.info("模型未调终结工具就想收尾，追加提示重发一次")
    return context
