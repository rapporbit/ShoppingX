"""recall_memories —— 按主题主动翻一遍长期记忆（只读、非终结）。

**为什么自动注入之外还要一个口**：``context_shaping.preference_inject`` 注入的是**本轮域内**
的偏好（域由 planner 判出，见 ``injector._in_scope``）——搜背包时不会把「买鞋只穿宽楦」推给
模型，这是对的，否则每轮都塞满不相干的条目。但用户说「我以前买过的那双鞋」「你还记得我不
喜欢什么材质吗」时，要的恰恰是域外那些。没有这个工具，模型只能回「我不记得」，而库里明明有。

匹配沿用 ``forget_preferences`` 的口径：**互为子串的确定性匹配，不做语义猜测**。记忆类的 bug
不会崩、只会把推荐做反（查不到就当用户没说过），所以宁可漏也不要糊。
"""

from typing import Any

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.memory.injector import format_history, format_preferences
from app.memory.store import get_store
from app.tools._shell import tool

#: 一次最多回多少条，防止老用户的全量记忆灌爆上下文。
RECALL_MAX_ENTRIES = 20


class RecallMemoriesOutput(BaseModel):
    """recall_memories 的结构化返回。"""

    preferences: str = Field(default="", description="命中的长期偏好，每行一条")
    history: str = Field(default="", description="行为历史（搜索 / 购买）")
    count: int = Field(default=0, description="命中的偏好条数")
    note: str = Field(default="", description="给模型的简短说明")


def _hit(topic: str, content: str, keywords: list[str]) -> bool:
    """topic 与这条记忆是否互为子串命中（与 injector.forget_preferences 同一口径）。"""
    if not topic:
        return True
    low = content.lower()
    if topic in low or low in topic:
        return True
    return any(k and (k.lower() in topic or topic in k.lower()) for k in keywords)


@tool
async def recall_memories(topic: str = "") -> RecallMemoriesOutput:
    """翻用户的长期记忆（偏好 + 历史）。何时调用：用户提到「我以前 / 我之前买的 / 你还记得吗」，
    或需要**本轮品类之外**的偏好——自动注入给你的只有本轮品类域内那几条。
    参数 topic：留空回全部；给词则按该词筛（确定性子串匹配，不做语义联想）。
    """
    await monitor.report_tool_start("recall_memories", topic=topic)
    user_id = get_user_id() or ""
    if not user_id:
        await monitor.report_tool_end("recall_memories", count=0)
        return RecallMemoriesOutput(note="匿名会话没有长期记忆，登录后才会跨会话记住偏好")

    store = get_store()
    low = (topic or "").strip().lower()
    entries = [e for e in await store.read(user_id) if _hit(low, e.content, list(e.keywords))]
    history: list[Any] = list(await store.read_history(user_id))

    prefs_text = format_preferences(entries[:RECALL_MAX_ENTRIES])
    note = ""
    if not entries:
        # 查不到要说清「库里确实没有」，别让模型把空结果说成「你没告诉过我」之外的话。
        note = f"没有与「{topic}」相关的长期偏好" if low else "这个用户还没有沉淀任何长期偏好"
    elif len(entries) > RECALL_MAX_ENTRIES:
        note = f"命中 {len(entries)} 条，只回了最先的 {RECALL_MAX_ENTRIES} 条"

    await monitor.report_tool_end("recall_memories", count=len(entries))
    return RecallMemoriesOutput(
        preferences=prefs_text,
        history=format_history(history),
        count=len(entries),
        note=note,
    )
