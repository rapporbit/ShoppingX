"""E 块 · 语义缓存的确定性单测：精确层 + 语义层 + 阈值/容量 + category_insight 集成。

语义层用可控的归一化向量（手造，不依赖真 embedding），断言「近义命中、不近不命中」。
集成测 category_insight：第二次相同/近义 query 不再走 Hybrid 召回（命中缓存）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.recall.semantic_cache import SemanticCache


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)  # L2 归一，点积即余弦


# ---------- 精确层 ----------
def test_exact_hit_and_miss() -> None:
    c: SemanticCache[str] = SemanticCache("t")
    assert c.get_exact("k") is None
    c.put("k", "V", _unit(1, 0))
    assert c.get_exact("k") == "V"
    assert c.get_exact("other") is None


# ---------- 语义层 ----------
def test_semantic_hit_above_threshold() -> None:
    c: SemanticCache[str] = SemanticCache("t", threshold=0.9)
    c.put("k", "V", _unit(1, 0))
    near = _unit(0.97, 0.05)  # 与 (1,0) 余弦 ≈ 0.998 > 0.9
    assert c.get_semantic(near) == "V"


def test_semantic_miss_below_threshold() -> None:
    c: SemanticCache[str] = SemanticCache("t", threshold=0.9)
    c.put("k", "V", _unit(1, 0))
    far = _unit(0.2, 1.0)  # 余弦 ≈ 0.196 < 0.9
    assert c.get_semantic(far) is None


def test_semantic_group_isolation() -> None:
    c: SemanticCache[str] = SemanticCache("t", threshold=0.5)
    c.put("k", "QUICK", _unit(1, 0), group="quick")
    # 同向量但不同 group → 不命中（quick/deep 结果不同，不能互串）。
    assert c.get_semantic(_unit(1, 0), group="deep") is None
    assert c.get_semantic(_unit(1, 0), group="quick") == "QUICK"


def test_semantic_dim_mismatch_is_skipped_not_crash() -> None:
    # towers 远程↔本地回退可能产出不同维向量；混维不能让 np.dot 抛 ValueError 打挂工具。
    c: SemanticCache[str] = SemanticCache("t", threshold=0.5)
    c.put("k", "V", _unit(1, 0, 0))  # 缓存里是 3 维
    assert c.get_semantic(_unit(1, 0)) is None  # 查询是 2 维 → 跳过该项、不命中、不抛错


def test_capacity_evicts_oldest() -> None:
    c: SemanticCache[str] = SemanticCache("t", max_entries=2, threshold=0.99)
    c.put("a", "A", _unit(1, 0))
    c.put("b", "B", _unit(0, 1))
    c.put("cc", "C", _unit(1, 1))  # 超容量，淘汰最旧的 a
    assert c.get_semantic(_unit(1, 0)) is None  # a 的向量已被淘汰
    assert c.get_semantic(_unit(0, 1)) == "B"


# ---------- category_insight 集成 ----------
async def test_category_insight_caches_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.category_insight as ci

    ci._cache.clear()
    calls = 0

    async def _fake_resolve(query: str) -> tuple:
        nonlocal calls
        calls += 1
        return "", 0.0, []  # 空结果也照样缓存（card_count=0 是合法结论）

    monkeypatch.setattr(ci, "_resolve_and_fetch", _fake_resolve)

    async def _fake_encode(text: str) -> np.ndarray:
        return _unit(1, 0)

    monkeypatch.setattr(ci, "_encode_category", _fake_encode)

    out1 = await ci.category_insight.ainvoke({"category": "luggage", "depth": "quick"})
    out2 = await ci.category_insight.ainvoke({"category": "luggage", "depth": "quick"})
    assert out1.category == out2.category
    assert calls == 1  # 第二次精确命中缓存，没再走 Hybrid 召回


async def test_category_insight_depth_not_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.category_insight as ci

    ci._cache.clear()
    calls = 0

    async def _fake_resolve(query: str) -> tuple:
        nonlocal calls
        calls += 1
        return "", 0.0, []

    async def _fake_encode(text: str) -> np.ndarray:
        return _unit(1, 0)

    monkeypatch.setattr(ci, "_resolve_and_fetch", _fake_resolve)
    monkeypatch.setattr(ci, "_encode_category", _fake_encode)

    await ci.category_insight.ainvoke({"category": "luggage", "depth": "quick"})
    await ci.category_insight.ainvoke({"category": "luggage", "depth": "deep"})
    assert calls == 2  # quick / deep 各算一次，不互相命中


async def test_category_insight_exact_works_without_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编码失败（towers 挂）时：语义层失效，但**精确层仍生效**——相同 query 第二次仍命中、不重算。"""
    import app.tools.category_insight as ci

    ci._cache.clear()
    calls = 0

    async def _fake_resolve(query: str) -> tuple:
        nonlocal calls
        calls += 1
        return "", 0.0, []

    async def _no_encode(text: str) -> Any:
        return None  # 编码不可用

    monkeypatch.setattr(ci, "_resolve_and_fetch", _fake_resolve)
    monkeypatch.setattr(ci, "_encode_category", _no_encode)

    await ci.category_insight.ainvoke({"category": "luggage", "depth": "quick"})
    await ci.category_insight.ainvoke({"category": "luggage", "depth": "quick"})
    assert calls == 1  # 精确层不依赖向量，相同 query 第二次走快路径
