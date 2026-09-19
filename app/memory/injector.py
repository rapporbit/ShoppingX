"""行为历史的渲染与写入 —— 把「用户做过什么」接进 AgentLoop 的两端。

**偏好那一腿已经整条摘掉**（M4）：长期记忆改成 ``key / value / category`` 的事实模型，
读走 :func:`app.memory.facts.select_tier_one_facts` + ``render_memory_block``（每轮注入）
与 ``recall_memories``（按需召回），写走 ``save_memory`` 工具与回合后抽取，两端都只经
:mod:`app.memory.fact_store`。本模块原有的 ``build_preference_block`` /
``persist_new_preferences`` / ``forget_preferences`` / ``_in_scope`` 随之删除——
记忆从此**只经模型上下文生效**，不再有一条绕过模型直接改检索结果的通路。

剩下的两件事都只关于行为历史：

- **读 / 注入**：:func:`build_history_block` 渲染成文本，由主 loop 拼进当轮 human message
  （不是 system prompt——历史每轮都可能变，混进 system 会打断跨轮稳定的 prompt cache 前缀）。
- **写**：:func:`record_search_history` 在 ``run_agent`` 收尾机制性写入，不靠模型调工具。
"""

from __future__ import annotations

import logging

from app.api.context import get_thread_id
from app.memory.store import HistoryEntry, PreferenceStore, get_store

logger = logging.getLogger(__name__)

# 空占位文本（供上层注入时判空跳过——空块不拼进当轮 human，避免给模型无意义的「暂无」噪声）。
HISTORY_EMPTY = "（暂无历史记录）"

_HISTORY_NOUN = {"purchase": "购买", "search": "搜索"}
_KIND_ORDER = {"purchase": 0, "search": 1}


def format_history(entries: list[HistoryEntry]) -> str:
    """把行为历史渲染成可注入文本（主 loop 拼进当轮 human；对齐 refdocs/06 §3.2）。

    每种 kind 现在可有多条（``HISTORY_MAX_PER_KIND``），故按「最近 / 更早」标注新旧——否则
    三行都叫「上次搜索」，模型无从判断哪条才是最新的一次。组内一律新→旧，最新的排第一行。
    """
    if not entries:
        return HISTORY_EMPTY
    # 固定 purchase 在前、search 在后，组内新→旧：注入文本对同一份数据稳定（利于 prompt cache）。
    grouped: dict[str, list[HistoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.kind, []).append(entry)

    lines: list[str] = []
    for kind in sorted(grouped, key=lambda k: _KIND_ORDER.get(k, 9)):
        items = sorted(grouped[kind], key=lambda e: e.created_at, reverse=True)
        noun = _HISTORY_NOUN.get(kind, kind)
        for idx, entry in enumerate(items):
            prefix = "最近" if idx == 0 else "更早"
            lines.append(f"- {prefix}{noun}：{entry.content}")
    return "\n".join(lines)


async def build_history_block(user_id: str, store: PreferenceStore | None = None) -> str:
    """读出用户行为历史并格式化为可注入文本（``get_system_prompt`` 的 recent_history 实参）。

    无 user_id（匿名）直接返回占位，不碰 Store。历史条目少（每种 kind 几条），全量注入即可。
    """
    if not user_id:
        return format_history([])
    st = store or get_store()
    return format_history(await st.read_history(user_id))


async def record_search_history(
    user_id: str,
    content: str,
    source_session: str | None = None,
    store: PreferenceStore | None = None,
) -> None:
    """记一条 ``search`` 行为历史（每 kind 保留最近 ``HISTORY_MAX_PER_KIND`` 条 + 30 天 TTL）。

    在 ``run_agent`` 收尾**机制性**写入（不靠模型调工具），**只记用户搜了什么、不记结果**——
    检索结果是系统的输出而非用户的表态，记进来会让烂召回反过来污染下一轮上下文（见调用点注释）。
    ``purchase`` 类留待将来接入下单流程再写。匿名 / 空内容跳过。
    """
    if not user_id or not content.strip():
        return
    session = (source_session or get_thread_id()) or ""
    st = store or get_store()
    await st.write_history(
        user_id, HistoryEntry(kind="search", content=content, source_session=session)
    )
