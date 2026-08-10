"""域反证与品类门解锚（第四病根：LLM 结构化输出「合法但错」无核验、后果静默反转）。

三层防线各测一块：
- ``infer_domains_from_text``：用户原文词面 → 域投票（确定性、宁漏勿错）。
- ``reconcile_domains``：planner 域漏判时词面证据**并入不替换**（手表 badcase 的主修）。
- ``_category_relevance`` 锚核验：category 锚与原文词面分歧 → 门 fail-open 不执法——
  「合法但错」的锚会让品类门反着杀（把真手表沉底、留西装），不执法比反向执法安全。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.memory.domains import infer_domains_from_text, reconcile_domains
from app.utils.thread_ctx import thread_scope


class TestInferDomains:
    def test_watch_query_votes_jewelry_watches(self) -> None:
        """手表 badcase 原句：正装场景词齐飞，词表仍稳判 jewelry_watches。"""
        assert "jewelry_watches" in infer_domains_from_text("formal dress watch men business")
        assert "jewelry_watches" in infer_domains_from_text("想买一块正装手表，商务场合戴")

    def test_word_boundary_reuses_term_hits(self) -> None:
        """命中口径复用 term_hits：watching 不算 watch（裸 in 会误判）。"""
        assert infer_domains_from_text("i enjoy watching movies") == set()

    def test_no_coverage_returns_empty_not_other(self) -> None:
        """词表覆盖不到 → 空集 =「无证据」，不是「不属于任何域」。"""
        assert infer_domains_from_text("") == set()
        assert infer_domains_from_text("送长辈的礼物，有什么推荐") == set()


class TestReconcileDomains:
    def test_union_not_replace(self) -> None:
        """漂移主修：planner 判 apparel、原文词面判 jewelry_watches → 并入，不丢 planner 的。"""
        out = reconcile_domains(["apparel"], "formal dress watch men business")
        assert "jewelry_watches" in out and "apparel" in out

    def test_no_evidence_keeps_planner_verdict(self) -> None:
        """词面无证据时原样返回（返回同一列表对象，零开销）。"""
        domains = ["apparel"]
        assert reconcile_domains(domains, "送长辈的礼物") is domains


@pytest.mark.asyncio
async def test_anchor_conflict_fails_open(monkeypatch) -> None:
    """category 锚（planner 输出）与用户原文词面分歧 → 门不执法且**不触发 reranker**。"""
    import app.tools.item_picker as ip
    from app.api.context import set_original_query
    from app.memory.session_state import SessionPrefState, save_pt
    from app.tools.schemas import ItemCandidate

    monkeypatch.setattr(ip, "get_reranker", lambda: pytest.fail("锚分歧下不该走到打分"))
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-anchor-conf", sd):
        save_pt(sd, SessionPrefState(category="dress shoes"))  # 锚漂成鞋履
        set_original_query("想买一块正装手表")  # 原文明明在买表
        cands = [ItemCandidate(item_id="W1", platform="amazon", title="Quartz Watch")]
        scores, gate_on, conflict = await ip._category_relevance(cands)
    assert (scores, gate_on, conflict) == ({}, False, True)


@pytest.mark.asyncio
async def test_anchor_agreement_enforces_gate(monkeypatch) -> None:
    """锚与原文一致 → 照旧执法（对照组，防解锚把门整个焊死）。"""
    import app.tools.item_picker as ip
    from app.api.context import set_original_query
    from app.memory.session_state import SessionPrefState, save_pt
    from app.tools.schemas import ItemCandidate

    class _Fake:
        async def score_detailed(self, query, texts):
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-anchor-ok", sd):
        save_pt(sd, SessionPrefState(category="watch"))
        set_original_query("想买一块正装手表")
        cands = [ItemCandidate(item_id="W1", platform="amazon", title="Quartz Watch")]
        scores, gate_on, conflict = await ip._category_relevance(cands)
    assert gate_on is True and conflict is False and scores["W1"] == 0.9


@pytest.mark.asyncio
async def test_must_terms_join_rerank_query(monkeypatch) -> None:
    """普通轮的 rerank query = 品类 + must 硬约束词（M22）。

    单个品类词把 cross-encoder 降级成品类分类器——离线在 ESCI 上同分母实测，「品类 + 约束
    关键词」比「纯品类词」多 +3.10pt vs +0.59pt（recall@8）。这条锁住拼接行为，也锁住
    **prefer 软偏好词绝不能进**（背包 badcase：偏好词字面命中把跨品类垃圾抬到真品之上）。
    """
    import app.tools.item_picker as ip
    from app.api.context import set_original_query
    from app.memory.session_state import SessionPrefState, save_pt
    from app.tools.schemas import ItemCandidate

    seen: list[str] = []

    class _Fake:
        async def score_detailed(self, query, texts):
            seen.append(query)
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-must-join", sd):
        save_pt(sd, SessionPrefState(category="backpack"))
        set_original_query("想买一个防水的双肩包")
        cands = [ItemCandidate(item_id="B1", platform="amazon", title="Waterproof Backpack")]
        await ip._category_relevance(cands, ["waterproof"])
    assert seen == ["backpack waterproof"]


@pytest.mark.asyncio
async def test_no_must_terms_falls_back_to_category(monkeypatch) -> None:
    """没有 must（多数轮次的常态）→ 退回纯品类词，行为与 M22 之前一致。"""
    import app.tools.item_picker as ip
    from app.api.context import set_original_query
    from app.memory.session_state import SessionPrefState, save_pt
    from app.tools.schemas import ItemCandidate

    seen: list[str] = []

    class _Fake:
        async def score_detailed(self, query, texts):
            seen.append(query)
            return [0.9] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-must-none", sd):
        save_pt(sd, SessionPrefState(category="backpack"))
        set_original_query("想买一个双肩包")
        cands = [ItemCandidate(item_id="B1", platform="amazon", title="Backpack")]
        await ip._category_relevance(cands, [])
    assert seen == ["backpack"]
