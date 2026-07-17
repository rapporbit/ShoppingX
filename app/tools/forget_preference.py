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

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.memory.injector import forget_preferences


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
    """撤回 / 忘掉一条已沉淀的长期偏好(跨会话生效,非终结性,忘完继续跑)。

    何时调用:用户明确要求撤回某条长期偏好时(「别再记着我不要塑料了」「取消我喜欢小众的偏好」
    「把 X 从偏好里删掉」)。
    参数:
      - description:要撤回的偏好描述或关键词,如「塑料」「小众设计」。与已存偏好的 content 或
        keyword 互为子串即命中删除;匹配不上则不删(宁可漏删也不误删他条)。
        如果已知精确 key 则优先用 key 参数。
      - key:精确的偏好 dedup_key(可在 <user_long_term_preferences> 中查看已有条目的 [dedup_key])。
        传了 key 则直接按 dedup_key 删除,不走模糊匹配。
      - polarity:可选。限定只在 like(喜欢)或 dislike(排斥)一侧删;不传则两侧都查。
        使用 key 直接删除时此参数无效。
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
