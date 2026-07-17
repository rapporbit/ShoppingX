"""chat_fallback —— 非购物意图的闲聊兜底（终结性）。

不是每句话都是购物需求（「你好」「你能干嘛」「谢谢」）。Think 判定本轮非检索意图时，调它
直接给一句得体回复收尾，而不是硬套 planner→检索那套流程空转。

**终结性**（在 ``TERMINAL_TOOLS`` 里）：调用即收尾，是另一条明确的「话讲完了」出口，
和 ``shopping_summary`` 一道堵住「不收尾死循环」。

用 LLM 生成回复，但收紧到「简短、引导回购物场景」，避免跑题成通用聊天机器人。
"""

from __future__ import annotations

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tools import tool
from pydantic import BaseModel

from app.agent.llm import get_llm
from app.agent.token_budget import charge_tool_llm_usage
from app.api import monitor

_SYSTEM = (
    "你是 ShoppingX 购物助手。用户这句不是购物检索需求，请用一两句话友好回应，"
    "并自然地把话题引回「我可以帮你跨平台找商品、比价、算到手价」。不要编造商品或价格。"
)


class ChatFallbackOutput(BaseModel):
    """chat_fallback 的结构化返回（终结性）。"""

    reply: str


@tool
async def chat_fallback(message: str) -> ChatFallbackOutput:
    """非购物意图的闲聊兜底（终结性）。

    何时调用：用户这句是打招呼 / 问能力 / 闲聊等非检索意图时——调它给一句回应即收尾。
    参数：
      - message：用户的原话。
    """
    await monitor.report_tool_start("chat_fallback", message=message)
    # usage 经 callback 收集入账（与 planner / shopping_summary 同口径：工具内部 LLM 调用
    # 不经过 agent middleware，不挂就是漏账，见 token_budget.charge_tool_llm_usage）。
    usage_cb = UsageMetadataCallbackHandler()
    try:
        resp = await get_llm().ainvoke(
            [("system", _SYSTEM), ("user", message)], config={"callbacks": [usage_cb]}
        )
    except Exception:
        # 模型调用失败也要补一条 end 事件，否则前端（M8）会看到工具「永远在跑」。
        await monitor.report_tool_end("chat_fallback", error=True)
        raise
    finally:
        charge_tool_llm_usage(usage_cb.usage_metadata)
    reply = resp.content if isinstance(resp.content, str) else str(resp.content)
    out = ChatFallbackOutput(reply=reply)
    await monitor.report_tool_end("chat_fallback")
    return out
