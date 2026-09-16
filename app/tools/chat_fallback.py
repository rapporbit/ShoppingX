"""chat_fallback —— 非购物意图的收尾出口（终结性）。

不是每句话都是购物需求（「你好」「你能干嘛」「谢谢」），也不是每次回答都有商品卡（品类怎么挑、
某款口碑如何）。Think 判定本轮不出清单时调它收尾，而不是硬套 planner→检索那套流程空转。

**终结性**（在 ``TERMINAL_TOOLS`` 里）：调用即收尾，是另一条明确的「话讲完了」出口，
和 ``shopping_summary`` 一道堵住「不收尾死循环」。

**文案由主模型在 ``message`` 里给，工具原样透出**（与 shopping_summary 同一条口径）。此前是
工具再调一次 fast 模型「归纳成一两句」，对「你好」无所谓，对知识类长回答就是毁灭性的：
2026-09-16 实测，模型把 1500+ 字符的电动牙刷选购指南写进 message，收尾却只剩「好的，随时告诉
我具体需求！」——落盘的 summary.md 和会话历史里存的都是那句客套话。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.invoke import call_text
from app.agent.llm import get_fast_llm
from app.api import monitor
from app.tools._args import StrListArg
from app.tools._candidates import hydrate
from app.tools._shell import tool
from app.tools.item_picker import _preview_item

_SYSTEM = (
    "你是 ShoppingX 购物助手。用户这句不是购物检索需求，请用一两句话友好回应，"
    "并自然地把话题引回「我可以帮你跨平台找商品、比价、算到手价」。不要编造商品或价格。"
)


class ChatFallbackOutput(BaseModel):
    """chat_fallback 的结构化返回（终结性）。"""

    reply: str
    items: list[dict[str, Any]] = Field(default_factory=list, description="附在回复下的商品卡")


@tool
async def chat_fallback(message: str, item_ids: StrListArg | None = None) -> ChatFallbackOutput:
    """非检索意图的收尾（终结性）：闲聊/问能力，或把已查到的知识类回答讲完。
    参数 message = **直接发给用户的完整回复原文**（工具原样透出、不改写，Markdown 表格列表都行）；
    item_ids 可选，把要附的商品卡 id 带上（缺收货信息时带用户要买的那件）。
    """
    await monitor.report_tool_start("chat_fallback", message=message)
    cards = [_preview_item(c) for c in hydrate(list(item_ids or []))]
    if cards:
        await monitor.report_items_preview(cards)
    reply = (message or "").strip()
    if not reply:
        # 入参没给文案才退回内部 LLM（与 shopping_summary 同口径）。
        # **主路径为什么不再改写**：模型常把整篇答案写在 message 里（实测 research 型问答给过
        # 1500+ 字符的选购指南），再让 fast 模型「归纳成一两句」就是把答案换成客套话——落盘的
        # summary.md、会话历史 final_text 全变成那句客套话，用户和下一轮上下文都拿不到原答案。
        # 用量由 call_text 入账（工具内部 LLM 调用不经过 agent middleware，不入账就是漏账）。
        try:
            # 快档：说一句人话不是推理题，此前误用主档还开着 thinking，给问候付 reasoning 解码。
            reply = await call_text(
                get_fast_llm(), [("system", _SYSTEM), ("user", "用户没说什么具体的，招呼一句。")]
            )
        except Exception:
            # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
            await monitor.report_tool_end("chat_fallback", error=True)
            raise
    out = ChatFallbackOutput(reply=reply, items=cards)
    await monitor.report_tool_end("chat_fallback")
    return out
