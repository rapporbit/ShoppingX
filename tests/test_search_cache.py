"""阶段 3 · 两级检索缓存：key 归一化 / 两级命中 / singleflight / 退化。

测的是**省没省掉那次回源**，不是「返回值对不对」——后者在缓存前后必然一样，只有回源计数能
证明缓存真的起了作用。每条用例自己数 ``fetch`` 被调了几次。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from conftest import FakeRedis  # tests/ 的 conftest（pytest 已把它放进 sys.path）

from app.recall import search_cache
from app.recall.schemas import RecallCandidate
from app.utils.dependency import DependencyDown


def _candidate(item_id: str = "A1") -> RecallCandidate:
    return RecallCandidate(
        item_id=item_id, platform="amazon", title="canvas travel bag", price=20.0, score=0.9
    )


class _Counter:
    """回源闭包 + 调用计数。"""

    def __init__(self, result: list[RecallCandidate] | None = None) -> None:
        self.calls = 0
        self.result = result if result is not None else [_candidate()]

    async def __call__(self) -> list[RecallCandidate]:
        self.calls += 1
        return self.result


@pytest.fixture
def l2() -> Iterator[FakeRedis]:
    """给这一组用例一个假 L2（conftest 的 autouse fixture 默认把 L2 摘掉了）。"""
    client = FakeRedis()
    search_cache.set_client(client)
    search_cache.reset_cache()
    yield client
    search_cache.set_client(None)
    search_cache.reset_cache()


# ---------- key 归一化 ----------
def test_key_ignores_platform_order_and_case() -> None:
    """平台顺序 / 大小写 / 空白不该分裂出两份缓存——搜的是同一批货。"""
    a = search_cache.make_key("Canvas Bag", 20, ["ebay", "amazon"])
    b = search_cache.make_key("  canvas bag ", 20, ["Amazon", "EBAY"])
    assert a == b


def test_key_separates_top_k_and_filters() -> None:
    """top_k 与过滤条件必须进 key。

    top_k 不截断复用：召回是 ANN，limit 变了拿到的不是同一个前缀（HNSW 的 ef 随 limit 变），
    从 30 条里切 20 条与真搜 20 条不是一回事。
    """
    base = search_cache.make_key("bag", 20, ["amazon"])
    assert search_cache.make_key("bag", 30, ["amazon"]) != base
    assert search_cache.make_key("bag", 20, ["amazon"], price_usd_max=50) != base
    assert search_cache.make_key("bag", 20, ["amazon"], min_rating=4.0) != base


def test_key_changes_with_index_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """换索引 = 旧 key 全作废，不必手工清缓存。"""
    before = search_cache.make_key("bag", 20, ["amazon"])
    monkeypatch.setenv("RETRIEVAL_INDEX_VERSION", "v2")
    assert search_cache.make_key("bag", 20, ["amazon"]) != before


# ---------- L1 ----------
async def test_second_call_hits_l1() -> None:
    fetch = _Counter()
    first = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    second = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 1, "第二次还回源了"
    assert [c.item_id for c in second] == [c.item_id for c in first]


async def test_disabled_switch_always_refetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """总开关关掉 = 回到改造前，每次都回源（回滚开关要真能回滚）。"""
    monkeypatch.setenv("RETRIEVAL_CACHE_ENABLED", "0")
    fetch = _Counter()
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 2


async def test_empty_result_uses_short_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """空结果要缓存（挡穿透），但 TTL 短——库一扩「这个词没货」就不再成立。

    把空结果 TTL 压到 0 再查一次：短 TTL 真的生效的话，第二次必须重新回源。
    """
    monkeypatch.setenv("RETRIEVAL_CACHE_EMPTY_TTL", "0")
    fetch = _Counter(result=[])
    await search_cache.cached_recall("nothing", 20, ["amazon"], fetch=fetch)
    await search_cache.cached_recall("nothing", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 2
    # 有货的那批照旧走正常 TTL，不受影响。
    good = _Counter()
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=good)
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=good)
    assert good.calls == 1


# ---------- L2 ----------
async def test_l2_serves_a_fresh_process(l2: FakeRedis) -> None:
    """L1 清掉（= 换了个副本）仍能从 L2 拿到，且回填 L1。

    这是「两级」的意义所在：一个副本查过的词，另一个副本不必再打一次 Qdrant。
    """
    fetch = _Counter()
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 1

    search_cache.reset_cache()  # 只清 L1，L2 留着
    again = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 1, "L2 没命中"
    assert [c.item_id for c in again] == ["A1"]

    # 回填过 L1：把 L2 抽走也还能命中。
    search_cache.set_client(None)
    await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 1


async def test_undecodable_payload_falls_back_to_fetch(l2: FakeRedis) -> None:
    """缓存里躺着上一个版本写的东西（字段变了）→ 当没命中，不是让这次检索失败。"""
    key = search_cache.make_key("bag", 20, ["amazon"])
    await l2.set(key, '[{"nope": 1}]')
    fetch = _Counter()
    out = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
    assert fetch.calls == 1
    assert [c.item_id for c in out] == ["A1"]


async def test_broken_l2_degrades_to_fetch() -> None:
    """L2 每次都炸：结果照给，只是没省到——缓存的任何一层都不该反噬主链路。"""

    class _BoomRedis:
        async def get(self, *_a: object, **_kw: object) -> str:
            raise ConnectionError("连不上")

        async def set(self, *_a: object, **_kw: object) -> bool:
            raise ConnectionError("连不上")

        async def delete(self, *_a: object, **_kw: object) -> int:
            raise ConnectionError("连不上")

    search_cache.set_client(_BoomRedis())
    try:
        fetch = _Counter()
        out = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
        assert [c.item_id for c in out] == ["A1"]
        assert fetch.calls == 1
        # L1 照常工作（它不依赖 Redis）。
        await search_cache.cached_recall("bag", 20, ["amazon"], fetch=fetch)
        assert fetch.calls == 1
    finally:
        search_cache.set_client(None)
        search_cache.reset_cache()


# ---------- singleflight 与异常 ----------
async def test_concurrent_same_key_fetches_once() -> None:
    """同轮 batch 跨平台搜同一个词会在同一瞬间撞上同一个 key：只该回源一次。"""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_fetch() -> list[RecallCandidate]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return [_candidate()]

    tasks = [
        asyncio.create_task(search_cache.cached_recall("bag", 20, ["amazon"], fetch=slow_fetch))
        for _ in range(5)
    ]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks)
    assert calls == 1, f"回源了 {calls} 次"
    assert all([c.item_id for c in r] == ["A1"] for r in results)


async def test_fetch_error_propagates_and_is_not_cached() -> None:
    """回源抛 DependencyDown 要原样抛：吞成空列表会让「服务挂了」长得和「库里没货」一样。

    也不能把这次失败缓存下来——否则依赖恢复了，用户还要等一个 TTL 才搜得到东西。
    """
    calls = 0

    async def boom() -> list[RecallCandidate]:
        nonlocal calls
        calls += 1
        raise DependencyDown("qdrant", "dense 检索失败")

    for _ in range(2):
        with pytest.raises(DependencyDown):
            await search_cache.cached_recall("bag", 20, ["amazon"], fetch=boom)
    assert calls == 2


async def test_inflight_is_cleaned_up_after_error() -> None:
    """失败后在飞表要清干净：留着的话同 key 的下一次会 await 一个已经失败的 Future。"""

    async def boom() -> list[RecallCandidate]:
        raise DependencyDown("qdrant", "炸")

    with pytest.raises(DependencyDown):
        await search_cache.cached_recall("bag", 20, ["amazon"], fetch=boom)
    assert search_cache._inflight == {}

    ok = _Counter()
    out = await search_cache.cached_recall("bag", 20, ["amazon"], fetch=ok)
    assert [c.item_id for c in out] == ["A1"]


def test_metrics_layer_labels_are_stable() -> None:
    """命中分层复用既有的 cache_events 指标（cache=retrieval），不另起一个指标名。"""
    from app.observability.metrics import CACHE_EVENTS

    search_cache._observe("l1")
    assert CACHE_EVENTS.labels(cache="retrieval", result="l1")._value.get() >= 1
