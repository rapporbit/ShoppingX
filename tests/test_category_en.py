"""精排 query 用英文品类名：标题是英文，中文品类词打分偏低（rank-eval 2026-10-02）。

锁两件事：① 普通轮精排 query 优先 ``category_en``、空时退回中文 ``category``；
② 英文名的性别 / 颜色修饰被机械剔掉（带修饰的 query 把真跑鞋的路径分压到 0.15）。
"""

import json
from pathlib import Path

import pytest

from app.api.context import set_turn_constraints
from app.memory.turn_constraints import TurnConstraints, clean_category_en
from app.tools.item_picker import item_picker
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Men's Running Shoes Black", "running shoes"),
        ("women's handbag", "handbag"),
        ("kids backpack", "backpack"),
        ("golden retriever toy", "golden retriever toy"),  # 词边界：golden ≠ gold
        ("black", ""),  # 剔空 → 调用方退回中文品类
    ],
)
def test_clean_category_en(raw: str, want: str) -> None:
    assert clean_category_en(raw) == want
    assert TurnConstraints.build(category_en=raw).category_en == want


async def _query_used(monkeypatch, tmp_path: Path, pt: TurnConstraints) -> str:
    import app.tools.item_picker as ip

    seen: list[str] = []

    class _Fake:
        async def score_detailed(self, query: str, texts: list[str]):
            seen.append(query)
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    with thread_scope("t-cat-en", tmp_path):
        set_turn_constraints(pt)
        cands = [
            ItemCandidate(item_id=i, platform="amazon", title=t, price_usd=50.0, rating=4.5)
            for i, t in (("a", "Ultraboost Running Shoe"), ("b", "Trail Running Shoes"))
        ]
        await item_picker.ainvoke({"candidates": cands})
    assert seen, "没走到 reranker"
    return seen[0]


async def test_rerank_query_prefers_category_en(monkeypatch, tmp_path: Path) -> None:
    pt = TurnConstraints.build(category="跑鞋", category_en="men's running shoes")
    assert await _query_used(monkeypatch, tmp_path, pt) == "running shoes"


async def test_rerank_query_falls_back_to_chinese(monkeypatch, tmp_path: Path) -> None:
    pt = TurnConstraints.build(category="跑鞋")
    assert await _query_used(monkeypatch, tmp_path, pt) == "跑鞋"


async def test_title_and_path_scored_with_separate_queries(monkeypatch, tmp_path: Path) -> None:
    """标题分用 planner 检索词（判具体属性），路径分用英文品类名（品牌颜色会压塌路径分）。"""
    import app.tools.item_picker as ip

    calls: dict[str, list[str]] = {}

    class _Fake:
        async def score_detailed(self, query: str, texts: list[str]):
            calls[query] = list(texts)
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    pt = TurnConstraints.build(
        category="跑鞋", category_en="running shoes", keywords=["adidas", "black running shoes"]
    )
    with thread_scope("t-split", tmp_path):
        set_turn_constraints(pt)
        cand = ItemCandidate(
            item_id="a",
            platform="amazon",
            title="Ultraboost Running Shoe",
            price_usd=50.0,
            fine_category="Shoes > Athletic > Running > Road Running",
        )
        await item_picker.ainvoke({"candidates": [cand]})
    assert calls["running shoes"] == ["shoes > athletic > running > road running"]
    assert len(calls["adidas black running shoes"]) == 1  # 只有标题


async def test_rerank_score_is_a_continuous_term(monkeypatch, tmp_path: Path) -> None:
    """过了门的候选之间，精排分高的排前：便宜、评分高也压不过 0.9 对 0.5 的相关性差距。"""
    import app.tools.item_picker as ip

    # 0.5 过得了展示相对门（0.35×0.9），两件都展示，比的是顺序；去掉精排项时短裤靠评分排前。
    rr = {"Cheap Running Shorts": 0.5, "Road Running Shoe": 0.9}

    class _Fake:
        async def score_detailed(self, query: str, texts: list[str]):
            return [next(v for k, v in rr.items() if k.lower() in t) for t in texts], True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    with thread_scope("t-wrel", tmp_path):
        set_turn_constraints(TurnConstraints.build(category="跑鞋", category_en="running shoes"))
        cands = [
            ItemCandidate(
                item_id="shorts",
                platform="amazon",
                title="Cheap Running Shorts",
                price_usd=9.0,
                rating=4.9,
            ),
            ItemCandidate(
                item_id="shoe",
                platform="amazon",
                title="Road Running Shoe",
                price_usd=120.0,
                rating=4.1,
            ),
        ]
        out = await item_picker.ainvoke({"candidates": cands})
    assert [p["item_id"] for p in json.loads(str(out))["picks"]] == ["shoe", "shorts"]
