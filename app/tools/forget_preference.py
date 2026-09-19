"""forget_preference —— 撤回 / 忘掉一条已沉淀的长期偏好（非终结性）。

这是长期记忆写入（现由会话结束后的记忆管家 ``app/memory/curator.py`` 统一负责）的**反向操作**,
补上 refdocs/06 §6.1 点名却一直没接入口的 ``delete``:Store 早有 ``delete`` 口,但没有调用方——
用户没法主动「忘掉我不要塑料这条」。有了它,长期记忆才能**改、能删**,而非只增不减。它是购物工作流
里**唯一**保留的记忆相关工具（写入已剥离给 curator）。

**非终结**——忘完继续跑,收尾仍由 shopping_summary / chat_fallback 负责。删除走确定性匹配
(``injector.forget_preferences``:content / keyword 互为子串命中即删),不猜、宁可漏删也不误删。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.memory.injector import forget_preferences
from app.tools._shell import tool


class ForgetPreferenceOutput(BaseModel):
    """forget_preference 的结构化返回。"""

    removed: list[str] = Field(default_factory=list, description="被撤回的偏好 content 列表")
    count: int = Field(default=0, description="撤回条数")
    note: str = Field(default="", description="给模型的简短说明")


@tool
async def forget_preference(
    description: str = "",
    key: str | None = None,
    polarity: Literal["like", "dislike"] | None = None,
) -> ForgetPreferenceOutput:
    """清掉**旧偏好库**里的一条条目（非终结）。注意：它看不到 <user_long_term_memory> 里的事实
    ——那些要撤回请用原 key 调 save_memory 写新值。只有用户提到的明显是 M1 之前沉淀、且这里
    列不出来的旧偏好时才用它。参数：key（旧条目的 dedup_key）或 description 关键词模糊匹配；
    polarity 可选限定 like/dislike 一侧。
    """
    await monitor.report_tool_start(
        "forget_preference", description=description, key=key, polarity=polarity
    )
    user_id = get_user_id() or ""
    if key:
        removed = await forget_preferences(user_id, "", dedup_keys=[key])
    else:
        removed = await forget_preferences(user_id, description, polarity=polarity)
    await monitor.report_tool_end("forget_preference", removed=len(removed))
    if not user_id:
        note = "匿名会话没有可撤回的长期偏好"
    elif removed:
        note = f"已撤回 {len(removed)} 条偏好"
    else:
        note = "未找到匹配的偏好,未撤回任何条目"
    return ForgetPreferenceOutput(removed=removed, count=len(removed), note=note)
