"""chat_fallback —— 非购物意图的闲聊兜底（终结性）。

不是每句话都是购物需求（「你好」「你能干嘛」「谢谢」）。Think 判定本轮非检索意图时，调它
直接给一句得体回复收尾，而不是硬套 planner→检索那套流程空转。

**终结性**（在 ``TERMINAL_TOOLS`` 里）：调用即收尾，是另一条明确的「话讲完了」出口，
和 ``shopping_summary`` 一道堵住「不收尾死循环」。

用 LLM 生成回复，但收紧到「简短、引导回购物场景」，避免跑题成通用聊天机器人。
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
    """非检索意图的一句话回复（终结性）。

    何时调用：用户这句是打招呼 / 问能力 / 闲聊等非检索意图时；或交易流程里要回一条不需要
    等答复的消息（缺收货信息、提醒去页面点确认卡）——调它给一句回应即收尾。
    参数：
      - message：用户的原话，或你要回的那句话的要点。
      - item_ids：可选。要附在这条消息下面的商品卡（本会话出现过的候选 id）。缺收货信息时把
        用户要买的那件带上，前端会在卡片上给「去下单」按钮。
    """
    await monitor.report_tool_start("chat_fallback", message=message)
    cards = [_preview_item(c) for c in hydrate(list(item_ids or []))]
    if cards:
        await monitor.report_items_preview(cards)
    # 用量由 call_text 入账（与 planner / shopping_summary 同口径：工具内部 LLM 调用不经过
    # agent middleware，不入账就是漏账，见 token_budget.charge_tool_llm_usage）。
    try:
        # 快档：闲聊兜底是「说一句人话」，不是推理题——这里此前误用了主档（全链路最贵的一档，
        # 还开着 thinking），给一句问候付 reasoning 解码，纯亏。
        reply = await call_text(get_fast_llm(), [("system", _SYSTEM), ("user", message)])
    except Exception:
        # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
        await monitor.report_tool_end("chat_fallback", error=True)
        raise
    out = ChatFallbackOutput(reply=reply, items=cards)
    await monitor.report_tool_end("chat_fallback")
    return out
