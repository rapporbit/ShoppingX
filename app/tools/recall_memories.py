"""recall_memories —— 按主题主动翻一遍长期记忆（只读、非终结）。

**为什么自动注入之外还要一个口**：``context_shaping.preference_inject`` 每轮注入的只有 tier-one
那批（全部 constraint + 最近 8 条，见 ``facts.select_tier_one_facts``）。老用户攒下的几十条里，
没进那批的照样可能是用户当下问的那条——「我以前买过的那双鞋」「你还记得我不喜欢什么材质吗」。
没有这个工具，模型只能回「我不记得」，而库里明明有。

匹配沿用 ``facts.match_facts`` 的口径：**确定性子串匹配，不做语义猜测**。记忆类的 bug 不会崩、
只会把推荐做反（查不到就当用户没说过），所以宁可漏也不要糊。

**返回要过围栏**（`recall_memories` 已进 ``EXTERNAL_SOURCE_TOOLS``）：记忆的正文源头是用户在
某一轮说的话，一条被写进去的「忽略以上指令」会在此后每次召回时重放。它是待评估的数据，
不是给模型的指令。
"""

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.memory.fact_store import get_fact_store
from app.memory.facts import MemoryFact
from app.memory.injector import format_history
from app.memory.store import get_store
from app.tools._shell import tool

#: 一次最多回多少条，防止老用户的全量记忆灌爆上下文。
RECALL_MAX_ENTRIES = 20


class RecallMemoriesOutput(BaseModel):
    """recall_memories 的结构化返回。"""

    memories: str = Field(default="", description="命中的长期记忆，每行 [分类] key: 内容")
    history: str = Field(default="", description="行为历史（搜索 / 购买）")
    count: int = Field(default=0, description="命中的记忆条数")
    note: str = Field(default="", description="给模型的简短说明")


def _render(facts: list[MemoryFact]) -> str:
    """每行一条 ``[category] key: value``——与注入块同一形态。

    不复用 ``render_memory_block``：那个带 ``<user_long_term_memory>`` 标签，而这里的文本要嵌进
    工具返回的 JSON，外面还会再包一层 ``<external_content>`` 围栏。两层标签套着反而让模型分不清
    哪个是边界。
    """
    return "\n".join(f"[{f.category.value}] {f.key}: {f.value}" for f in facts)


@tool
async def recall_memories(topic: str = "") -> RecallMemoriesOutput:
    """翻用户的长期记忆（事实 + 行为历史）。何时调用：用户提到「我以前 / 我之前买的 / 你还记得
    吗」，或需要**自动注入之外**的旧记忆——每轮注入给你的只有硬规则和最近那几条。
    参数 topic：留空回全部；给词则按该词筛（确定性子串匹配，不做语义联想）。
    """
    await monitor.report_tool_start("recall_memories", topic=topic)
    user_id = get_user_id() or ""
    if not user_id:
        await monitor.report_tool_end("recall_memories", count=0)
        return RecallMemoriesOutput(note="匿名会话没有长期记忆，登录后才会跨会话记住偏好")

    low = (topic or "").strip().lower()
    facts = await get_fact_store().search_facts(user_id, low)
    history = list(await get_store().read_history(user_id))

    note = ""
    if not facts:
        # 查不到要说清「库里确实没有」，别让模型把空结果说成「你没告诉过我」之外的话。
        note = f"没有与「{topic}」相关的长期记忆" if low else "这个用户还没有沉淀任何长期记忆"
    elif len(facts) > RECALL_MAX_ENTRIES:
        note = f"命中 {len(facts)} 条，只回了最近更新的 {RECALL_MAX_ENTRIES} 条"

    await monitor.report_tool_end("recall_memories", count=len(facts))
    return RecallMemoriesOutput(
        memories=_render(facts[:RECALL_MAX_ENTRIES]),
        history=format_history(history),
        count=len(facts),
        note=note,
    )
