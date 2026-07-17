"""长期记忆（Mmem 重构后）：SQLite Store + 偏好注入 + 行为历史 + 收藏。

**相对改造前，这里少了四类测试——因为它们的被测实现已经不存在了：**

- 「本地 JSON 文件损坏 → 隔离重建」「Redis 故障 → 降级」：双后端已被 SQLite 取代。
- 「read_relevant 语义 top-k 召回」：域隔离（PrefDomain）已把本轮相关偏好压到个位数，
  叠在上面的语义裁剪纯属冗余，连带删掉了记忆层对 app.recall.towers 的依赖。
- 「recency_weight 半衰期衰减」：衰减唯一的消费者是 user 塔画像，而那里把全部 like 加权平均成
  一个向量，权重再准也被稀释没了。改成 last_confirmed_at 只在偏好页面显示「N 个月没用过」。
- 「user_id 路径穿越净化」：偏好不再落成以 user_id 命名的文件，穿越面本身消失了。

**新增的核心不变式是 blocking**：只有用户在偏好页面亲手勾「绝不推荐」的条目才有硬淘汰权；
curator 从对话里学到的一律只减分。见 ``test_agent_learned_dislike_never_gets_blocking``。

库由 conftest 建（DATABASE_URL 指向临时 SQLite + init_db 跑迁移）。测试间靠**唯一 user_id**
隔离，而不是各建各的库——这也顺带测到了 Store 的多用户隔离本身。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agent.prompts import get_system_prompt
from app.api.context import set_session_domains
from app.memory.assemble import assemble
from app.memory.injector import (
    PREF_EMPTY,
    build_history_block,
    build_preference_block,
    forget_preferences,
    format_history,
    format_preferences,
    persist_new_preferences,
    record_search_history,
)
from app.memory.store import (
    FavoriteItem,
    HistoryEntry,
    PreferenceEntry,
    PreferenceStore,
    get_store,
)
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


@contextmanager
def in_domains(*domains: str) -> Iterator[None]:
    """模拟「planner 已判出品类域」的上下文。

    ``_in_scope`` 是 fail-closed 的：域为空 → 只有 global 生效。真实链路里读偏好的两条路
    （assemble / build_preference_block）都跑在 planner 之后，域必然已就位；单测直调时得自己
    把它摆上，否则测的是「判不出品类」那条保守分支，而不是偏好本身的授权规则。
    """
    with thread_scope("t-mem", Path(tempfile.mkdtemp())):
        set_session_domains(list(domains))
        yield


def _uid() -> str:
    """每个测试一个独立 user_id——库是共享的，隔离靠身份而非各建各的库。"""
    return f"u-{uuid4().hex[:8]}"


@pytest.fixture
def store() -> PreferenceStore:
    return get_store()


def _pref(**kw: object) -> PreferenceEntry:
    """造一条偏好，只填关心的字段。"""
    base: dict[str, object] = {
        "slug": "leather",
        "content": "不接受皮革材质",
        "category": "material",
        "domain": "footwear",
        "polarity": "dislike",
    }
    base.update(kw)
    return PreferenceEntry(**base)  # type: ignore[arg-type]


# ---------- dedup_key 由结构化字段派生（不由 LLM 手拼）----------
def test_dedup_key_derived_from_structured_fields() -> None:
    assert _pref().dedup_key == "dislike:material:footwear:leather"


def test_dedup_key_differs_by_domain() -> None:
    """同一句「不要皮革」，在鞋和沙发下是**两条**偏好——域是去重身份的一部分。"""
    assert _pref(domain="footwear").dedup_key != _pref(domain="furniture").dedup_key


# ---------- 写入 → 跨会话读出（核心验收点）----------
async def test_write_then_read_across_sessions(store: PreferenceStore) -> None:
    uid = _uid()
    await store.write(uid, _pref(source_session="sess-1"))
    entries = await store.read(uid)
    assert len(entries) == 1
    assert entries[0].content == "不接受皮革材质"
    assert entries[0].source_session == "sess-1"


async def test_user_isolation(store: PreferenceStore) -> None:
    a, b = _uid(), _uid()
    await store.write(a, _pref())
    assert await store.read(a) != []
    assert await store.read(b) == []


async def test_anonymous_write_read_noop(store: PreferenceStore) -> None:
    """匿名（空 user_id）不落库、读空——不该为没登录的人攒偏好。"""
    await store.write("", _pref())
    assert await store.read("") == []


# ---------- upsert：碰撞合并（内容覆盖 + created_at 保留 + 刷新确认时间）----------
async def test_upsert_overwrites_content_preserves_created_at(store: PreferenceStore) -> None:
    uid = _uid()
    await store.write(uid, _pref(content="旧表述"))
    first = (await store.read(uid))[0]
    await store.write(uid, _pref(content="新表述"))
    entries = await store.read(uid)
    assert len(entries) == 1  # 同 dedup_key 不堆重复条目
    assert entries[0].content == "新表述"
    assert entries[0].created_at == first.created_at  # 保留最早创建时间
    assert entries[0].last_confirmed_at >= first.last_confirmed_at  # 重复提及=确认还活着


async def test_delete(store: PreferenceStore) -> None:
    uid = _uid()
    await store.write(uid, _pref())
    await store.delete(uid, "dislike:material:footwear:leather")
    assert await store.read(uid) == []


async def test_delete_missing_key_is_silent(store: PreferenceStore) -> None:
    await store.delete(_uid(), "dislike:material:footwear:nonexistent")  # 不抛


# ---------- 核心不变式：杀伤力只由用户授予 ----------
def _draft(**kw: object) -> SimpleNamespace:
    """模拟 curator 的 _PersistentPref / parser 的 UserPrefDraft（鸭子类型）。"""
    base: dict[str, object] = {
        "content": "不接受皮革",
        "category": "material",
        "domain": "footwear",
        "slug": "leather",
        "polarity": "dislike",
        "keywords": ["leather", "皮革"],
    }
    base.update(kw)
    return SimpleNamespace(**base)


async def test_agent_learned_dislike_never_gets_blocking(store: PreferenceStore) -> None:
    """**本次重构最重要的一条**：curator 学到的偏好拿不到硬淘汰权，就算它自己填了 blocking=True。

    这道闸设在唯一落库口上（source="agent" 一律 blocking=False），而不是指望每条调用路径自觉——
    「LLM 不得决定永久排除」必须是机制，不是约定。
    """
    uid = _uid()
    await persist_new_preferences(uid, [_draft(blocking=True)], source="agent")
    entry = (await store.read(uid))[0]
    assert entry.blocking is False
    assert entry.is_blocking is False
    # 于是它只能减分，进不了硬淘汰名单
    with in_domains("footwear"):
        bundle = await assemble(uid)
    assert bundle.exclude == []
    assert "leather" in bundle.penalty


async def test_user_marked_blocking_gets_hard_exclusion(store: PreferenceStore) -> None:
    """用户在偏好页面亲手勾「绝不推荐」→ 才进 item_picker 的硬淘汰名单。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(blocking=True)], source="user")
    entry = (await store.read(uid))[0]
    assert entry.is_blocking is True
    with in_domains("footwear"):
        bundle = await assemble(uid)
    assert "leather" in bundle.exclude
    assert bundle.penalty == []  # 已升到硬淘汰，不再重复减分


async def test_user_pref_without_blocking_only_attenuates(store: PreferenceStore) -> None:
    """用户手填但没勾「绝不推荐」（「不太喜欢皮革」）→ 仍然只减分。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(blocking=False)], source="user")
    with in_domains("footwear"):
        bundle = await assemble(uid)
    assert bundle.exclude == []
    assert "leather" in bundle.penalty


async def test_dislike_terms_fall_back_to_content_tokens(store: PreferenceStore) -> None:
    """没给 keywords 的条目回退切 content——切不出原子词就不匹配（安全，不误杀）。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(keywords=[], content="塑料 廉价")], source="agent")
    with in_domains("footwear"):
        terms = (await assemble(uid)).penalty
    assert "塑料" in terms and "廉价" in terms


async def test_dislike_terms_anonymous_empty() -> None:
    assert (await assemble("")).exclude == []
    assert (await assemble("")).penalty == []


# ---------- 用户手填的两条保护（agent 不得覆盖）----------
async def test_user_pref_not_overwritten_by_agent(store: PreferenceStore) -> None:
    """用户亲手写的内容不被 curator 的推断悄悄改掉——「我明明改过了」是记忆最伤信任的失败。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(content="绝对不要皮革")], source="user")
    await persist_new_preferences(uid, [_draft(content="似乎不太喜欢皮革")], source="agent")
    entry = (await store.read(uid))[0]
    assert entry.content == "绝对不要皮革"
    assert entry.source == "user"


async def test_user_pref_survives_own_edit(store: PreferenceStore) -> None:
    """用户自己再次手填时照常覆盖——那本来就是他要改。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(content="旧的")], source="user")
    await persist_new_preferences(uid, [_draft(content="新的")], source="user")
    assert (await store.read(uid))[0].content == "新的"


async def test_agent_cannot_strip_user_blocking(store: PreferenceStore) -> None:
    """curator 后续的同 key 写入不能把用户勾的「绝不推荐」降级掉。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(blocking=True)], source="user")
    await persist_new_preferences(uid, [_draft(blocking=False)], source="agent")
    assert (await store.read(uid))[0].is_blocking is True


# ---------- 写回：persist_new_preferences（唯一落库口）----------
async def test_persist_roundtrip_and_keywords(store: PreferenceStore) -> None:
    uid = _uid()
    written = await persist_new_preferences(uid, [_draft()], source_session="s1")
    assert len(written) == 1
    entry = (await store.read(uid))[0]
    assert entry.keywords == ["leather", "皮革"]
    assert entry.source_session == "s1"


async def test_persist_coerces_bad_polarity_and_domain(store: PreferenceStore) -> None:
    """LLM 吐了枚举外的值：polarity 兜到 like、domain 兜到 other（保守档，只本轮生效）。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(polarity="whatever", domain="不存在的域")])
    entry = (await store.read(uid))[0]
    assert entry.polarity == "like"
    assert entry.domain == "other"


async def test_persist_empty_or_anonymous_noop(store: PreferenceStore) -> None:
    assert await persist_new_preferences("", [_draft()]) == []
    assert await persist_new_preferences(_uid(), []) == []


async def test_persist_supersede_deletes_old_keys(store: PreferenceStore) -> None:
    """矛盾消解：curator 判定新偏好顶替旧的 → 先删旧 dedup_key 再写新（recency-wins）。"""
    uid = _uid()
    await persist_new_preferences(uid, [_draft(slug="plastic", content="不要塑料")])
    old_key = "dislike:material:footwear:plastic"
    assert {e.dedup_key for e in await store.read(uid)} == {old_key}

    await persist_new_preferences(
        uid,
        [_draft(slug="metal", polarity="like", content="改要金属")],
        keys_to_supersede=[old_key],
    )
    keys = {e.dedup_key for e in await store.read(uid)}
    assert old_key not in keys
    assert "like:material:footwear:metal" in keys


# ---------- 注入：format + build_preference_block + 不进 system prompt ----------
def test_format_preferences_renders_polarity() -> None:
    text = format_preferences([_pref(), _pref(slug="canvas", polarity="like", content="喜欢帆布")])
    assert "排斥" in text and "偏好" in text
    assert (
        "[dislike:material:footwear:leather]" in text
    )  # dedup_key 是 curator 判 supersede 的 handle


def test_format_preferences_empty() -> None:
    assert format_preferences([]) == PREF_EMPTY


async def test_build_block_anonymous_returns_placeholder() -> None:
    assert await build_preference_block("") == PREF_EMPTY


async def test_preferences_not_in_system_prompt() -> None:
    """偏好**不进 system prompt**：它每轮都变，混进去会打断跨轮稳定的 prompt cache 前缀。

    它走当轮 human message（见 main_agent._inject_runtime_context）。这里守的是「system prompt
    保持纯静态」这个前提——它一旦被破坏，缓存命中率会静默垮掉，而没人会立刻发现。
    """
    assert "不接受皮革材质" not in get_system_prompt()


# ---------- 个性化：like 偏好词拼进检索词（取代了 user 塔向量画像）----------
async def test_like_search_terms_only_like(store: PreferenceStore) -> None:
    """dislike **绝不**进检索词：embedding 对否定算子编码极弱、对主题词极强，「不要皮革」拼进
    query 等于把请求向量往「皮革」那片推，召回**更多**皮革——与意图正好相反。"""
    uid = _uid()
    await store.write(uid, _pref(keywords=["leather"]))  # dislike 皮革
    await store.write(
        uid, _pref(slug="canvas", polarity="like", content="喜欢帆布", keywords=["canvas"])
    )
    with in_domains("footwear"):
        assert (await assemble(uid)).search_terms == ["canvas"]


async def test_slot_prefs_never_reach_search_terms(store: PreferenceStore) -> None:
    """槽位型偏好（收货地）不进检索词——把国家码拼进商品语义检索纯属噪声。

    实测：「常用收货地：CN」(keywords=["CN"]) 让检索 query 变成「旅行三件套 cn」。它是给
    planner 的收货国解析用的**客观事实**，不是能出现在商品标题里的属性。判据就是这个。
    """
    uid = _uid()
    await store.write(
        uid,
        PreferenceEntry(
            slug="ship_to",
            content="常用收货地：CN",
            category="location",
            domain="global",
            polarity="like",
            keywords=["CN"],
        ),
    )
    await store.write(
        uid, _pref(slug="canvas", polarity="like", content="喜欢帆布", keywords=["canvas"])
    )
    with in_domains("footwear"):
        b = await assemble(uid)
    assert b.search_terms == ["canvas"]  # 收货地被挡在外面
    assert "cn" not in b.exclude and "cn" not in b.penalty


async def test_like_search_terms_ignores_entries_without_keywords(store: PreferenceStore) -> None:
    """没给 keywords 的 like 条目不进检索词——content 是整句（「喜欢小众设计的帆布包」），
    整句拼进 query 会把语义带偏。安全方向：宁可不个性化，也不污染检索。"""
    uid = _uid()
    await store.write(
        uid, _pref(slug="niche", polarity="like", content="喜欢小众设计", keywords=[])
    )
    assert (await assemble(uid)).search_terms == []


async def test_like_search_terms_anonymous_empty() -> None:
    assert (await assemble("")).search_terms == []


# ---------- 无会话上下文展示面的黑名单：blocking_exclude_terms ----------
async def test_blocking_exclude_terms_user_blocking_only_no_domain_gate(
    store: PreferenceStore,
) -> None:
    """只收用户亲手勾的 blocking（agent 推断的 dislike 拿不到硬淘汰权）、**不做域过滤**（给
    /api/similar 这类没有 planner / 品类域的展示面用）、且中文词已归一出英文变体可直接匹标题。"""
    from app.memory.assemble import blocking_exclude_terms

    uid = _uid()
    await store.write(
        uid,
        _pref(
            slug="plastic", content="绝不推荐塑料", keywords=["塑料"], source="user", blocking=True
        ),
    )
    await store.write(
        uid, _pref(slug="gaudy", category="style", content="不喜欢花哨", keywords=["花哨"])
    )
    terms = await blocking_exclude_terms(uid)
    assert "塑料" in terms and "plastic" in terms  # 归一是扩展：原词保留 + 英文变体
    assert "花哨" not in terms  # agent 推断的 dislike 不进（授权只由用户给）
    # domain=footwear 的条目照样返回——这条通路刻意不设域闸（docstring 论证）。
    assert await blocking_exclude_terms("") == []  # 匿名直通


# ---------- 撤回：forget_preferences ----------
async def test_forget_by_keyword_match(store: PreferenceStore) -> None:
    uid = _uid()
    await persist_new_preferences(uid, [_draft()])
    removed = await forget_preferences(uid, description="皮革")
    assert removed == ["不接受皮革"]
    assert await store.read(uid) == []


async def test_forget_by_dedup_key(store: PreferenceStore) -> None:
    uid = _uid()
    await persist_new_preferences(uid, [_draft()])
    removed = await forget_preferences(uid, dedup_keys=["dislike:material:footwear:leather"])
    assert len(removed) == 1
    assert await store.read(uid) == []


async def test_forget_no_match_deletes_nothing(store: PreferenceStore) -> None:
    uid = _uid()
    await persist_new_preferences(uid, [_draft()])
    assert await forget_preferences(uid, description="完全无关的东西") == []
    assert len(await store.read(uid)) == 1


async def test_forget_anonymous_noop() -> None:
    assert await forget_preferences("", description="皮革") == []


# ---------- 行为历史（与偏好正交：不去重、不合并、每 kind 留最近 N 条）----------
async def test_history_keeps_recent_n_per_kind(store: PreferenceStore) -> None:
    uid = _uid()
    for i in range(5):
        await store.write_history(uid, HistoryEntry(kind="search", content=f"搜了第 {i} 次"))
    entries = await store.read_history(uid)
    assert len(entries) == 3  # HISTORY_MAX_PER_KIND
    assert entries[0].content == "搜了第 4 次"  # 新→旧


async def test_history_isolated_from_preferences(store: PreferenceStore) -> None:
    """历史和偏好分表存：写历史不该污染偏好库，反之亦然。"""
    uid = _uid()
    await store.write_history(uid, HistoryEntry(kind="search", content="搜了跑鞋"))
    await store.write(uid, _pref())
    assert len(await store.read(uid)) == 1
    assert len(await store.read_history(uid)) == 1


async def test_record_search_history_and_block(store: PreferenceStore) -> None:
    uid = _uid()
    await record_search_history(uid, "搜了「旅行收纳袋」")
    block = await build_history_block(uid)
    assert "旅行收纳袋" in block
    assert "最近搜索" in block


async def test_record_search_history_anonymous_noop(store: PreferenceStore) -> None:
    await record_search_history("", "搜了点东西")  # 不抛、不落库


def test_format_history_marks_recency() -> None:
    """同 kind 多条时标「最近 / 更早」——否则三行都叫「上次搜索」，模型无从判断哪条最新。"""
    text = format_history(
        [
            HistoryEntry(kind="search", content="新的"),
            HistoryEntry(kind="search", content="旧的"),
        ]
    )
    assert "最近搜索" in text and "更早搜索" in text


# ---------- 收藏（♡）：与偏好 / 历史严格分家；只经**行为亲和**一条窄路影响 Agent ----------
def _fav(item_id: str = "i-1", **kw: object) -> FavoriteItem:
    base: dict[str, object] = {"item_id": item_id, "title": "帆布收纳袋", "platform": "amazon"}
    base.update(kw)
    return FavoriteItem(**base)  # type: ignore[arg-type]


async def test_favorites_crud_and_idempotent(store: PreferenceStore) -> None:
    uid = _uid()
    await store.write_favorite(uid, _fav())
    await store.write_favorite(uid, _fav(title="帆布收纳袋（改价）", price_usd=19.9))
    favs = await store.read_favorites(uid)
    assert len(favs) == 1  # 同 item_id 覆盖 → 重复点 ♡ 幂等
    assert favs[0].price_usd == 19.9

    await store.delete_favorite(uid, "i-1")
    assert await store.read_favorites(uid) == []


async def test_favorites_never_leak_into_prefs_or_history(store: PreferenceStore) -> None:
    """收藏**不会变成一条偏好**：收藏一件商品推不出任何偏好（可能只是想再比比价）。

    这条测试守的是一个**设计边界**而不是实现细节——哪天有人「顺手」把收藏喂进 prompt，它会在这里炸。

    **边界后来收窄过一次（Mmem 之后加了行为亲和）**：收藏不再是「纯展示、完全不影响 Agent」，它经
    :mod:`app.memory.affinity` 聚合成 item_picker 的弱加分项。但下面每一条断言依然成立，而且必须
    继续成立——那条新路是**刻意修得很窄**的：收藏不变成偏好条目（不进 ``store.read``）、不变成行为
    历史、不进 system prompt（``build_preference_block``）、不进检索词（``search_terms``）、更不淘汰
    商品（``exclude``）。它只在打分层微调排序。哪天有人想把收藏"提拔"成正经偏好，这里照样会炸。
    亲和那条路自己的不变式在 ``tests/test_affinity.py`` 里守。
    """
    uid = _uid()
    await store.write_favorite(uid, _fav())
    assert await store.read(uid) == []
    assert await store.read_history(uid) == []
    assert await build_preference_block(uid) == PREF_EMPTY
    assert (await assemble(uid)).search_terms == []
    assert (await assemble(uid)).exclude == []


async def test_favorites_user_isolation(store: PreferenceStore) -> None:
    a, b = _uid(), _uid()
    await store.write_favorite(a, _fav())
    assert len(await store.read_favorites(a)) == 1
    assert await store.read_favorites(b) == []
