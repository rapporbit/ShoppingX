"""品类门的 rerank query 必须**可复现**——它是执法判据，不能随 LLM 每轮的自由发挥变。

`_category_relevance` 的契约第①条一直写着「绝不拼 prefer 偏好词，拼了实测排序反转」，但实现
从另一条路违反了它：`assemble` 把 `pt.like_terms()`（P_t 的**软偏好**，polarity="like"、不分
blocking）装进了名为 `mem.must` 的字段，picker 再把 `mem.must` 当硬约束拼进 query。

2026-09-09 实测的后果（orchab 三遍同一条 query，见 docs/plans/baseline-artifacts/）：
偏好词每轮不同 → 同一候选 rerank 分数 0.783 vs 0.254 → 品类门判定翻转（oncat 17/26 vs 1/30）
→ 一遍出 8 件、一遍触发补搜回退只出 3 件。

**契约写在 docstring 里、实现悄悄偏离、没有任何测试会红**——这与 P0-1（reasoning_boost 空操作）
是同一种失败。所以这里断言的是**关系式**而不是某个具体 query 字符串：软偏好换一批，query 必须
逐字不变。谁再把偏好词接回 query，这条就红。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.context import set_session_pt
from app.memory.session_state import SessionConstraint, SessionPrefState, save_pt
from app.tools.item_picker import item_picker
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


def _cand(item_id: str, title: str) -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, platform="amazon", title=title, price_usd=29.9, rating=4.5
    )


def _pt(like_terms: list[str]) -> SessionPrefState:
    """带一条 like 约束的 P_t —— 正是 planner 每轮现生成、每轮都不同的那种软偏好。"""
    return SessionPrefState(
        category="旅行收纳",
        constraints=[
            SessionConstraint(
                id="c1",
                content="偏好" + "、".join(like_terms),
                polarity="like",
                keywords=like_terms,
                category="other",
                source_quote="想买便宜又抗造的旅行收纳三件套",
                turn_added=1,
                blocking=False,
            )
        ],
    )


async def _query_used(monkeypatch, tmp_path: Path, tag: str, like_terms: list[str]) -> str:
    """跑一次 picker，返回它实际发给 reranker 的 query。"""
    import app.tools.item_picker as ip

    seen: list[str] = []

    class _Fake:
        async def score_detailed(self, query: str, texts: list[str]):
            seen.append(query)
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    sd = tmp_path / tag
    sd.mkdir(parents=True, exist_ok=True)
    with thread_scope(f"t-rrq-{tag}", sd):
        # **两处都要设**：同一份 P_t 有两个读法——``assemble`` 读 ContextVar（get_session_pt），
        # 而 ``_category_relevance`` 的普通轮读的是落盘那份（load_pt(sd).category）。
        # 只设 ContextVar → 品类门因「P_t 无品类」整个停用，压根走不到 reranker；
        # 只 save_pt → mem.must 恒为空，这条测试**假绿**。两个坑都实测踩过。
        pt = _pt(like_terms)
        set_session_pt(pt)
        save_pt(sd, pt)
        await item_picker.ainvoke(
            {
                "candidates": [
                    _cand("a", "Travel Packing Cubes Set"),
                    _cand("b", "Canvas Luggage Organizer"),
                ]
            }
        )
    assert seen, "本轮没走到 reranker，测不到 query（检查品类门是否被别的条件停用了）"
    return seen[0]


async def test_rerank_query_ignores_soft_likes(monkeypatch, tmp_path: Path) -> None:
    """软偏好换一批 → rerank query **逐字不变**。

    两组 like 词取自线上实测的两遍（同一条 query、同一个 planner，产出就是不一样）。
    """
    q1 = await _query_used(
        monkeypatch, tmp_path, "run1", ["durable", "niche brand", "小众", "canvas", "nylon"]
    )
    q2 = await _query_used(
        monkeypatch, tmp_path, "run2", ["niche brand", "durable", "canvas", "waterproof"]
    )
    assert q1 == q2, f"软偏好污染了执法判据：{q1!r} != {q2!r}"


async def test_rerank_query_keeps_category_anchor(monkeypatch, tmp_path: Path) -> None:
    """护栏：别为了稳定把 query 修成空串——品类锚必须还在，否则门就没判据了。"""
    q = await _query_used(monkeypatch, tmp_path, "anchor", ["durable", "canvas"])
    assert "旅行收纳" in q
    for term in ("durable", "canvas", "niche", "小众", "nylon", "waterproof"):
        assert term not in q, f"偏好词 {term!r} 漏进了 rerank query：{q!r}"
