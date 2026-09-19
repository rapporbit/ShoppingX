"""save_memory —— 把用户当场说出口的长期事实写进记忆（非终结、非只读）。

**为什么在回合后的 curator 之外还要这个口**：curator 跑在会话结束之后，用户在这一轮说「记住我
不吃坚果」时，本轮剩下的检索还用不上它，而且用户得不到任何「记住了」的回执。参考实现
（`commerce-agents` 的 `save_memory`）同样是把写入摆在模型手里、当场生效。两条写路径不冲突：
它们都过 :func:`app.memory.facts.validate_fact` 这一道门，并按 ``key`` 覆盖写同一张表。

**遗忘也走这里**：用户说「别记了 / 我改主意了」时，正确做法是用同一个 key 覆盖成新值
（「不要塑料」→「塑料也可以」），不是删。模型手里没有删除口——删只由用户在偏好页点（见
`MemoryFactStore.delete_fact` 的注释）。所以本工具的回执**不说「已删除」**。
"""

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_thread_id, get_user_id
from app.memory.fact_store import get_fact_store
from app.memory.facts import (
    MEMORY_DISABLED_TEXT,
    MemoryWriteRejected,
    memory_enabled,
    validate_fact,
)
from app.tools._shell import tool


class SaveMemoryOutput(BaseModel):
    """save_memory 的结构化返回。"""

    saved: bool = Field(default=False, description="是否真的写进了长期记忆")
    key: str = Field(default="", description="规范化之后的 key（同 key 覆盖写）")
    value: str = Field(default="", description="规范化之后的内容")
    category: str = Field(default="", description="preference / constraint / context")
    note: str = Field(default="", description="给模型的简短说明，可直接转述给用户")


@tool
async def save_memory(key: str, value: str, category: str = "preference") -> SaveMemoryOutput:
    """把一条**跨会话都成立**的用户事实记进长期记忆（非终结）。何时调用：用户说「记住 X」
    「我一直 / 从不 X」「以后都按 X 来」，或说了一条会影响将来每次购物的硬规则。
    参数 key：这条事实的主题标识（英文小写下划线，如 ``material_avoid`` / ``default_ship_to``），
    同 key 会**覆盖**旧值——用户改主意时用原 key 写新值，不要另起一个 key。
    参数 value：一句话写清内容。参数 category：``constraint`` 一直成立的硬规则（每轮必注入）、
    ``preference`` 取向、``context`` 身份背景。只记长期成立的，本轮一次性的需求不要记。
    """
    await monitor.report_tool_start("save_memory", key=key, category=category)
    if not memory_enabled():
        # 部署把记忆整个关了。必须明说没存——含糊的失败会让模型回执「记住了」。
        await monitor.report_tool_end("save_memory", saved=False)
        return SaveMemoryOutput(note=MEMORY_DISABLED_TEXT)

    user_id = get_user_id() or ""
    if not user_id:
        await monitor.report_tool_end("save_memory", saved=False)
        return SaveMemoryOutput(note="匿名会话不沉淀长期记忆，登录后才会跨会话记住")

    try:
        fact = validate_fact(key, value, category, source_session=get_thread_id() or "")
    except MemoryWriteRejected as exc:
        # 被写入过滤器挡下（空值 / PII）。异常消息本身就是给用户的话，且不回显 value。
        await monitor.report_tool_end("save_memory", saved=False)
        return SaveMemoryOutput(note=str(exc))

    ok = await get_fact_store().upsert_facts(user_id, [fact])
    await monitor.report_tool_end("save_memory", saved=ok, key=fact.key)
    if not ok:
        # 库挂了。**不能说「已记住」**——用户据此不会再说第二遍，这条就永远丢了。
        return SaveMemoryOutput(
            key=fact.key,
            value=fact.value,
            category=fact.category.value,
            note="这次没存进长期记忆（存储暂时不可用），本轮我仍会按它来，但下次请再说一遍",
        )
    return SaveMemoryOutput(
        saved=True,
        key=fact.key,
        value=fact.value,
        category=fact.category.value,
        note=f"已记住：{fact.value}（{fact.category.value}，key={fact.key}）",
    )
