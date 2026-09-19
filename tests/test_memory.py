"""行为历史与收藏 —— 用户级持久数据里**不属于长期记忆**的那两样。

长期记忆（偏好 / 约束 / 背景）的测试在 ``test_memory_facts.py`` 与 ``test_curator.py``：
M1~M4 把它换成了 ``key / value / category`` 的事实模型，同 key 覆盖，只经模型上下文生效。
原来这里那一大片测试（dedup_key 派生、blocking 授权、域闸 ``_in_scope``、
``persist_new_preferences``、``forget_preferences``、like 词拼进检索词）连同被测实现一起删了。

留在这里的两样各有各的道理：

- **行为历史**是「做过什么」的事实快照，既不去重也不合并，每 kind 留最近 N 条 + TTL 过期。
- **收藏**是用户手工攒的清单，只经 :mod:`app.memory.affinity` 一条窄路进 item_picker 的弱加分
  （那条路的不变式在 ``test_affinity.py``）。

库由 conftest 建（DATABASE_URL 指向临时 SQLite + init_db 跑迁移）。测试间靠**唯一 user_id**
隔离，而不是各建各的库——这也顺带测到了 Store 的多用户隔离本身。
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agent.prompts import get_system_prompt
from app.memory.assemble import assemble
from app.memory.fact_store import get_fact_store
from app.memory.facts import MemoryFact
from app.memory.injector import (
    build_history_block,
    format_history,
    record_search_history,
)
from app.memory.store import (
    FavoriteItem,
    HistoryEntry,
    PreferenceStore,
    get_store,
)

pytestmark = pytest.mark.anyio


def _uid() -> str:
    """每个测试一个独立 user_id——库是共享的，隔离靠身份而非各建各的库。"""
    return f"u-{uuid4().hex[:8]}"


@pytest.fixture
def store() -> PreferenceStore:
    return get_store()


async def test_memory_not_in_system_prompt() -> None:
    """用户数据**不进 system prompt**：它每轮都变，混进去会打断跨轮稳定的 prompt cache 前缀。

    长期记忆走 planner 之后的 system 消息（``harness.hooks.context_shaping``），历史走当轮
    human message（``session_io.inject_runtime_context``）。这里守的是「system prompt 保持纯
    静态」这个前提——它一旦被破坏，缓存命中率会静默垮掉，而没人会立刻发现。
    """
    prompt = get_system_prompt()
    assert "不接受皮革材质" not in prompt
    assert "上次搜索" not in prompt


# ---------- 行为历史（与偏好正交：不去重、不合并、每 kind 留最近 N 条）----------
async def test_history_keeps_recent_n_per_kind(store: PreferenceStore) -> None:
    uid = _uid()
    for i in range(5):
        await store.write_history(uid, HistoryEntry(kind="search", content=f"搜了第 {i} 次"))
    entries = await store.read_history(uid)
    assert len(entries) == 3  # HISTORY_MAX_PER_KIND
    assert entries[0].content == "搜了第 4 次"  # 新→旧


async def test_history_isolated_from_long_term_memory(store: PreferenceStore) -> None:
    """历史和长期记忆分表存：写历史不该污染记忆库，反之亦然。"""
    uid = _uid()
    await store.write_history(uid, HistoryEntry(kind="search", content="搜了跑鞋"))
    await get_fact_store().upsert_facts(uid, [MemoryFact(key="shoe_taste", value="喜欢跑鞋")])
    assert len(await get_fact_store().get_facts(uid)) == 1
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
    继续成立——那条新路是**刻意修得很窄**的：收藏不变成长期记忆事实（不进 ``get_facts``）、不变成
    行为历史、更不淘汰商品（``exclude``）。它只在打分层微调排序。哪天有人想把收藏"提拔"成正经
    偏好，这里照样会炸。亲和那条路自己的不变式在 ``tests/test_affinity.py`` 里守。
    """
    uid = _uid()
    await store.write_favorite(uid, _fav())
    assert await get_fact_store().get_facts(uid) == []
    assert await store.read_history(uid) == []
    assert (await assemble(uid)).exclude == []


async def test_favorites_user_isolation(store: PreferenceStore) -> None:
    a, b = _uid(), _uid()
    await store.write_favorite(a, _fav())
    assert len(await store.read_favorites(a)) == 1
    assert await store.read_favorites(b) == []
