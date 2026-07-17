"""偏好注入与写回 —— 把长期记忆接进 AgentLoop 的两端。

- **读 / 注入（每轮开局）**：从 Store 读出偏好，格式化成一段文本，由主 loop 拼进**当轮 human
  message**（不是 system prompt——偏好每轮都变，混进 system prompt 会打断跨轮稳定的 prompt cache
  前缀）。全量注入，不做语义裁剪：域隔离（``PrefDomain``）已把本轮相关的偏好压到个位数。
- **写回 / 沉淀**：判定与写入**唯一**经由会话结束后独立运行的记忆管家 ``app/memory/curator.py``
  （``remember_preference`` 工具与 ``shopping_summary`` 的偏好抽取均已废弃，避免「购物 loop 里
  顺手判偏好」这种 afterthought 式误判）。:func:`persist_new_preferences` 是**唯一落库口**，
  矛盾消解用 ``keys_to_supersede`` 引用旧条目的 dedup_key，先删旧再写新（recency-wins）。

**执行侧只有三档**（Mmem 从六条通路收敛而来）：

1. :func:`dislike_exclude_terms` —— **硬淘汰**，只收用户在偏好页面亲手勾了「绝不推荐」的条目。
2. :func:`dislike_attenuate_terms` —— **减分不淘汰**，收 curator 从对话里学到的全部 dislike。
3. 正向 like → item_picker 加分。

1 和 2 的分界是**信息的来源**，不是 LLM 猜的置信度：让每轮都在猜的模型去决定「这件商品用户永远
不该看到」，风险和收益完全不匹配——猜错了用户搜不到东西，还完全归因不了。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Protocol

from app.api.context import get_session_domains, get_thread_id, record_learned_pref
from app.memory.domains import ALL_DOMAINS, DOMAIN_GLOBAL, DOMAIN_OTHER
from app.memory.store import (
    HistoryEntry,
    PreferenceEntry,
    PreferenceStore,
    Source,
    get_store,
)

logger = logging.getLogger(__name__)

_POLARITY_CN = {"like": "偏好", "dislike": "排斥"}

# content 回退切词的分隔符（无 keywords 的条目用）：空白 / 斜杠 / 中英逗号 / 顿号。
_TERM_SEP_RE = re.compile(r"[\s/，,、]+")


def _in_scope(entry: PreferenceEntry, domains: list[str]) -> bool:
    """这条偏好在本轮生效吗？—— 域隔离的**唯一**判定点，三条读取通路共用。

    - ``global``：跨品类底线（过敏 / 伦理 / 收货地）→ 永远生效。
    - 与本轮品类域匹配 → 生效。
    - 其余（含 ``other``）→ **本轮完全不生效**：不注入、不淘汰、不减分、不进检索词。跨域的偏好
      连「减分」都不该做——一条本轮无关的偏好，减分也是噪声。

    ``domains`` 为空（planner 没跑 / 判不出域 / 闲聊轮）→ **只有 global 生效**（fail-closed）。

    这里曾是 fail-open（空域一律放行），而那正是「买旅行包时差点被『买跑鞋不要皮革』误杀」那条
    bug 的最后一块拼图：偏好块拼进当轮 human 时 planner 还没跑，域必然是空的，于是域闸整个短路，
    模型看到跨域偏好、把它转述进 item_picker 拿到了硬淘汰权。注入挪到 planner 之后
    （``harness.hooks.preference_inject``）后，空域只剩一种含义——**planner 真的判不出品类**——
    这时正确的失效方向是保守：宁可让偏好本轮不生效（用户再说一遍即可），也不能让它在一个我们
    根本不知道是什么品类的轮次里静默杀商品（用户归因不了，只会觉得「这破 Agent 老搜不出东西」）。
    与 :data:`DOMAIN_OTHER` 落保守档是同一条哲学（见 ``memory.domains`` 的模块 docstring）。
    """
    if not domains:
        return entry.domain == DOMAIN_GLOBAL
    return entry.domain == DOMAIN_GLOBAL or entry.domain in domains


class NewPreferenceLike(Protocol):
    """鸭子类型协议：curator 产出的 ``_PersistentPref`` 需满足的最小结构。

    用 Protocol 而非直接 import 具体类，避免记忆层反向依赖工具/agent 层（保持分层单向）。
    仅作类型提示，不做运行时判定。声明成只读 ``property``（而非可变属性）——persist 只读这些
    字段，协变的只读成员才能接受 polarity 为 ``Literal["like","dislike"]`` 这类 ``str`` 子类型。
    """

    @property
    def content(self) -> str: ...
    @property
    def category(self) -> str: ...
    @property
    def polarity(self) -> str: ...
    @property
    def domain(self) -> str: ...
    @property
    def slug(self) -> str: ...


# 空占位文本（供上层注入时判空跳过——空块不拼进当轮 human，避免给模型无意义的「暂无」噪声）。
PREF_EMPTY = "（暂无沉淀偏好）"
HISTORY_EMPTY = "（暂无历史记录）"


def format_preferences(entries: list[PreferenceEntry]) -> str:
    """把偏好条目渲染成一段可注入文本（由主 loop 拼进当轮 human message）。

    每条一行「- [dedup_key] 内容（归类，极性）」，展示 dedup_key 让 LLM 看到已有条目的引用
    handle，便于 curator 在判定 keys_to_supersede 时**引用**（而不是重新构造）要顶替的旧条目。
    无偏好时给一句占位，让模型明确知道「这是新用户」。
    """
    if not entries:
        return PREF_EMPTY
    lines = []
    for e in entries:
        tag = _POLARITY_CN.get(e.polarity, e.polarity)
        meta = f"{e.category}，{tag}" if e.category else tag
        lines.append(f"- [{e.dedup_key}] {e.content}（{meta}）")
    return "\n".join(lines)


async def build_preference_block(
    user_id: str,
    query: str = "",
    store: PreferenceStore | None = None,
) -> str:
    """读出**本轮域内**的用户偏好并格式化为可注入文本（主 loop 拼进当轮 human message）。

    域过滤（见 :func:`_in_scope`）之后全量注入，不再做语义 top-k 裁剪：域隔离已经把本轮相关的
    偏好压到个位数，叠在上面的语义裁剪是第二层解法、纯属冗余，还得为此让记忆层依赖
    ``app.recall.towers`` 的远程 embedding。删掉它，``store`` 因此不再依赖召回层。

    ``query`` 形参保留（调用方已在传），当前不参与选择。匿名 / 未登录直接返回占位，不碰 Store。
    """
    if not user_id:
        return format_preferences([])
    st = store or get_store()
    domains = get_session_domains()
    return format_preferences([e for e in await st.read(user_id) if _in_scope(e, domains)])


def _terms_of(entry: PreferenceEntry) -> list[str]:
    """一条偏好可用于匹配的原子词：优先 ``keywords``，没有则把 ``content`` 按分隔符切。

    整句切不出原子词就退化为「不匹配」——安全方向（不会误杀）。
    """
    return [t for t in (entry.keywords or _TERM_SEP_RE.split(entry.content)) if t.strip()]


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

    无 user_id（匿名）直接返回占位，不碰 Store。历史条目少（每种 kind 一条），全量注入即可，
    不像偏好那样需要 read_relevant 语义裁剪。
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


async def forget_preferences(
    user_id: str,
    description: str = "",
    polarity: str | None = None,
    store: PreferenceStore | None = None,
    dedup_keys: list[str] | None = None,
) -> list[str]:
    """撤回（删除）匹配的长期偏好，返回被删条目的 content 列表。

    两种删除模式：
    1. **精确 dedup_key 删除**：传 ``dedup_keys`` 列表，直接按 :attr:`PreferenceEntry.dedup_key`
       删，无需模糊匹配。
    2. **模糊匹配删除**：传 ``description``，确定性匹配（不猜）——``description`` 与该条
       content 互为子串，或与其某个 keyword 互为子串（大小写不敏感）。可选 ``polarity`` 只
       在 like / dislike 一侧删。匹配不上就不删（宁可漏删也不误删他条）。
    匿名返回空。
    """
    if not user_id:
        return []
    st = store or get_store()
    removed: list[str] = []

    # 模式 1：精确 dedup_key 删除
    if dedup_keys:
        all_entries = await st.read(user_id)
        by_key = {e.dedup_key: e for e in all_entries}
        for k in dedup_keys:
            entry = by_key.get(k)
            if entry is not None:
                await st.delete(user_id, k)
                removed.append(entry.content)
        return removed

    # 模式 2：模糊匹配删除（原逻辑）
    desc = (description or "").strip().lower()
    if not desc:
        return []
    for e in await st.read(user_id):
        if polarity and e.polarity != polarity:
            continue
        content_low = e.content.lower()
        kws = [k.lower() for k in e.keywords]
        hit = (
            desc in content_low
            or content_low in desc
            or any(k and (k in desc or desc in k) for k in kws)
        )
        if hit:
            await st.delete(user_id, e.dedup_key)
            removed.append(e.content)
    return removed


async def persist_new_preferences(
    user_id: str,
    new_prefs: Sequence[NewPreferenceLike],
    source_session: str | None = None,
    keys_to_supersede: list[str] | None = None,
    store: PreferenceStore | None = None,
    source: Source = "agent",
) -> list[PreferenceEntry]:
    """把新偏好落库，返回实际写入的条目——长期库的**唯一**落库口。

    ``source`` 区分写入方：``agent``（默认，curator 从对话里学到）/ ``user``（用户在偏好页面手填，
    见 ``POST /api/preferences/{uid}``）。两条入口共用这一个落库口，是为了让「先删 supersede、
    再 upsert、记进本轮累加器」这套语义只有一份实现。

    ``keys_to_supersede`` 由 curator 判定——引用已展示给它的旧条目 :attr:`PreferenceEntry.dedup_key`
    （见 :func:`format_preferences`），写前先把这些条目从 Store 删掉（recency-wins 矛盾消解）。
    每条 ``new_prefs`` 需满足 :class:`NewPreferenceLike`（content/category/polarity/domain/slug），
    ``dedup_key`` 由 :class:`PreferenceEntry` 从这些结构化字段派生，不再需要 curator 手拼、
    也不需要多路径字段名兼容——curator 是长期库唯一写入口（``remember_preference`` 工具与
    ``shopping_summary`` 的偏好抽取已废弃）。

    无 user_id 则跳过（匿名会话不沉淀）。``source_session`` 缺省从 ``ContextVar`` 取当前
    thread_id（CONVENTIONS：上下文不手动透传），显式传值仅用于测试 / 离线脚本这类
    无请求上下文的场景。
    """
    if not user_id or not new_prefs:
        return []
    session = (source_session or get_thread_id()) or ""
    st = store or get_store()

    # curator 驱动的矛盾消解：先删旧 dedup_key 再写新偏好
    if keys_to_supersede:
        for k in keys_to_supersede:
            await st.delete(user_id, k)
            logger.info("curator 矛盾消解：删除旧偏好 dedup_key=%s", k)

    written: list[PreferenceEntry] = []
    for p in new_prefs:
        polarity = p.polarity if p.polarity in ("like", "dislike") else "like"
        keywords = list(getattr(p, "keywords", None) or [])
        category = p.category if p.category else "other"
        domain = p.domain if p.domain in ALL_DOMAINS else DOMAIN_OTHER
        # **杀伤力只由用户授予**：blocking 只认 source="user" 的写入（偏好页面的「绝不推荐」勾选）。
        # curator 走的是 source="agent"，它无论如何都拿不到硬淘汰权——这道闸设在唯一落库口上，
        # 而不是指望每条调用路径自觉，正是为了让「LLM 不得决定永久排除」成为机制而非约定。
        blocking = bool(getattr(p, "blocking", False)) and source == "user"
        entry = PreferenceEntry(
            slug=p.slug,
            content=p.content,
            category=category,  # type: ignore[arg-type]
            polarity=polarity,  # type: ignore[arg-type]
            domain=domain,
            source=source,
            blocking=blocking,
            source_session=session,
            keywords=keywords,
        )
        await st.write(user_id, entry)
        written.append(entry)
        # 唯一落库口，写成功即记进本轮累加器，供 run_agent 汇总进 learned_preferences、并经 AGUI
        # 推给前端那一行「记住了 …」（dedup_key = 那行 ✕ 的删除 handle）。非 run_agent 上下文
        # （含手填走的 API 路径）累加器为 None → 自动 no-op。
        record_learned_pref(entry.content, entry.dedup_key)
    return written
