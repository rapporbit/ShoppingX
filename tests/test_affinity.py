"""行为亲和：收藏（隐式行为）→ item_picker 弱加分。

守的四条不变式，全部对应 :mod:`app.memory.affinity` 里写死的纪律：
① 证据阈——收藏一件不算偏好；② 只加分不淘汰；③ 显式表达压过行为；④ 归因不静默（理由里说得出出处）。
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.memory.affinity import affinity_terms
from app.memory.assemble import assemble
from app.memory.store import FavoriteItem, PreferenceEntry, PreferenceStore, get_store
from app.tools.item_picker import item_picker
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


@pytest.fixture
def store() -> PreferenceStore:
    return get_store()


def _uid() -> str:
    return f"u-{uuid4().hex[:8]}"


async def _fav_titles(store: PreferenceStore, uid: str, *titles: str) -> None:
    for i, title in enumerate(titles):
        await store.write_favorite(uid, FavoriteItem(item_id=f"f-{i}", title=title))


def _cand(item_id: str, title: str, rating: float = 4.0, price: float = 50.0) -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, title=title, platform="amazon", price_usd=price, rating=rating
    )


# ---------- ① 证据阈：一件不算，两件才算 ----------
async def test_single_favorite_is_not_a_preference(store: PreferenceStore) -> None:
    """收藏一件推不出偏好——可能只是随手存个链接，或者想再比比价。"""
    uid = _uid()
    await _fav_titles(store, uid, "Canvas Travel Backpack")
    assert await affinity_terms(uid) == []


async def test_two_favorites_make_a_signal(store: PreferenceStore) -> None:
    uid = _uid()
    await _fav_titles(store, uid, "Canvas Travel Backpack", "Canvas Tote Bag with Nylon Lining")
    assert await affinity_terms(uid) == ["canvas"]  # nylon 只出现一次，不到阈值


async def test_repeated_token_in_one_title_counts_once(store: PreferenceStore) -> None:
    """一件收藏只投一票：标题里把 cotton 堆砌两遍，是电商 SEO，不是两个证据。"""
    uid = _uid()
    await _fav_titles(store, uid, "Cotton Shirt, 100% Cotton, Premium Cotton")
    assert await affinity_terms(uid) == []


async def test_long_token_absorbs_short_one(store: PreferenceStore) -> None:
    """ "genuine leather" 命中时 "leather" 不再单独计数，否则皮革类的证据数系统性翻倍。"""
    uid = _uid()
    await _fav_titles(store, uid, "Genuine Leather Wallet", "Genuine Leather Belt")
    assert await affinity_terms(uid) == ["genuine leather"]


async def test_anonymous_user_has_no_affinity() -> None:
    assert await affinity_terms("") == []


async def test_negated_material_is_not_evidence(store: PreferenceStore) -> None:
    """**「不含 X」的商品不是 X 的证据**——否则学出来的偏好正好是反的。

    收藏 vegan / leather-free 包的人，恰恰是**不想要**皮革的。裸子串匹配（``"leather" in title``）
    会把这两件数成两条皮革证据，于是系统反过来给真皮款加分。抽取侧必须走 term_hits 的否定修饰判定。
    """
    uid = _uid()
    await _fav_titles(store, uid, "Vegan Leather-Free Tote Bag", "Leather-Free Canvas Backpack")
    assert "leather" not in await affinity_terms(uid)


# ---------- ③ 显式表达压过行为 ----------
async def test_explicit_dislike_beats_favorites(store: PreferenceStore) -> None:
    """嘴上说不要皮革、手上收藏过两件皮革 —— 以说的为准，不能一边减分一边加分。

    **keywords 刻意用中文单词 ['皮革']**：这是 curator 从中文对话里抽词的真实形态。初版测试图省事写
    了英文 ['genuine leather']，恰好与亲和 token 精确相等 → 压制看着生效，实则只守住了「英文、单词、
    完全同形」这条最窄的路，给 bug 发了通行证（真实链路里中文 dislike 一条都压不住）。
    """
    uid = _uid()
    await _fav_titles(store, uid, "Genuine Leather Bag", "Genuine Leather Wallet")
    await store.write(
        uid,
        PreferenceEntry(
            polarity="dislike",
            category="material",
            slug="leather",
            domain="global",
            content="不要皮革",
            keywords=["皮革"],  # ← curator 的真实产物：中文原子词
        ),
    )
    bundle = await assemble(uid)
    assert "皮革" in bundle.penalty  # agent 学到的 dislike → 减分
    # 亲和词是 'genuine leather'（英文、且比 blocked 里的 'leather' 更长）——跨语言 + 长词变体
    # 两道坎都得跨过去才压得住。
    assert bundle.affinity == []


async def test_out_of_scope_dislike_still_suppresses_affinity(store: PreferenceStore) -> None:
    """**域闸不该让弱证据翻身**。

    这条守的是一个真洞（初版实现里真的有）：域判不出的轮次里，用户明说的 dislike 被 fail-closed 的
    域闸挡在 penalty 之外，而不受域闸约束的行为亲和照常加分——净效果是「他说不要皮革，系统反而把皮革
    顶上去了」。域闸是给**杀伤力**设的闸，不该顺带给我们自己的推断开绿灯。
    """
    uid = _uid()
    await _fav_titles(store, uid, "Genuine Leather Bag", "Genuine Leather Wallet")
    await store.write(
        uid,
        PreferenceEntry(
            polarity="dislike",
            category="material",
            slug="genuine_leather",
            domain="footwear",  # 域外：本轮（无会话域）它进不了 penalty
            content="买鞋时不要皮革",
            keywords=["genuine leather"],
        ),
    )
    bundle = await assemble(uid)
    assert bundle.penalty == []  # 域闸照常挡住它的杀伤力（这是对的，不该改）
    assert bundle.affinity == []  # 但它仍然压得住行为亲和 —— 不许弱证据反超


# ---------- ②④ 只加分不淘汰 + 归因不静默 ----------
async def test_affinity_lifts_but_never_drops(store: PreferenceStore, tmp_path: Path) -> None:
    """亲和只影响排序：命中的上浮，没命中的**照样在清单里**（行为是弱证据，无权淘汰）。"""
    uid = _uid()
    await _fav_titles(store, uid, "Canvas Backpack", "Canvas Duffel Bag")

    with thread_scope("t-aff", tmp_path, user_id=uid):
        out = await item_picker.ainvoke(
            {
                "candidates": [
                    _cand("a", "Nylon Packing Cubes", rating=4.0),
                    _cand("b", "Canvas Packing Cubes", rating=4.0),
                ]
            }
        )

    ids = [p.item_id for p in out.picks]
    assert ids == ["b", "a"]  # 帆布那件上浮
    assert out.excluded == []  # 但尼龙那件没被淘汰
    canvas_pick = next(p for p in out.picks if p.item_id == "b")
    assert "收藏" in (canvas_pick.pick_reason or "")  # 归因写明出处，不静默改排序
    assert canvas_pick.pref_matched is False  # 推断不冒充用户明说的偏好
