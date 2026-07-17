"""RAG 知识库基础设施单测：归一 / 本地 hybrid 召回 / reranker / 入库门禁（全离线）。"""

from __future__ import annotations

import pytest

from app.recall.category_kb import CategoryCard, normalize_category
from app.recall.kb_client import KBClient, should_disable_bm25
from app.recall.reranker import RerankerClient


def test_normalize_category() -> None:
    # 别名归一到标准品类。
    assert normalize_category("旅行收纳") == "travel accessories"
    assert normalize_category("Packing Cubes") == "travel accessories"
    # 大小写/空白清洗。
    assert normalize_category("  Coffee   Mug ") == "coffee mugs"
    # 开放词表：未知品类原样（清洗后）返回，不强行映射。
    assert normalize_category("Garden Hose") == "garden hose"


def test_search_text_boosts_category() -> None:
    card = CategoryCard(
        card_id="c1", category="luggage", card_type="attribute", summary="评分分布：4.5★+ 90%"
    )
    # 品类名重复提权，保住属性/价格卡的品类信号。
    assert card.search_text().count("luggage") == 2


def test_should_disable_bm25() -> None:
    assert should_disable_bm25("中性气质的咖啡杯") is True
    assert should_disable_bm25("minimal aesthetic mug") is True
    assert should_disable_bm25("luggage") is False


def _cards() -> list[CategoryCard]:
    return [
        CategoryCard(
            card_id="luggage_bs",
            category="luggage",
            card_type="bestseller",
            summary="luggage: Big Roller Suitcase / Carry On Spinner",
            confidence=0.9,
        ),
        CategoryCard(
            card_id="mug_bs",
            category="coffee mugs",
            card_type="bestseller",
            summary="coffee mugs: Ceramic Mug / Travel Tumbler",
            confidence=0.9,
        ),
        CategoryCard(
            card_id="shoe_bs",
            category="running shoes",
            card_type="bestseller",
            summary="running shoes: Trail Runner / Road Racer",
            confidence=0.9,
        ),
    ]


async def test_local_kb_search_ranks_relevant_first() -> None:
    """本地 hybrid 后端：query 应把最相关品类卡排在首位（KNN+token 融合）。"""
    kb = KBClient(cards=_cards(), host=None)
    assert kb.remote is False
    hits = await kb.search("luggage suitcase", coarse_k=3)
    assert hits, "应有召回"
    top_card, top_score = hits[0]
    assert top_card.card_id == "luggage_bs"
    # 融合分按降序。
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


async def test_local_kb_empty_when_no_cards() -> None:
    kb = KBClient(cards=[], host=None)
    assert await kb.search("anything", coarse_k=5) == []


async def test_reranker_local_scoring() -> None:
    """无 endpoint → 确定性本地打分：token 重叠多的候选分更高，且同输入同输出。"""
    rr = RerankerClient(endpoint=None)
    assert rr.remote is False
    cands = ["luggage suitcase roller", "ceramic coffee mug", "running shoes trail"]
    scores = await rr.score("luggage suitcase", cands)
    assert len(scores) == 3
    # 最相关候选得分最高。
    assert scores[0] == max(scores)
    # 确定性：重算一致。
    assert scores == await rr.score("luggage suitcase", cands)


async def test_reranker_empty_candidates() -> None:
    rr = RerankerClient(endpoint=None)
    assert await rr.score("q", []) == []


async def test_reranker_remote_failure_degrades_with_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B1：配了 endpoint 但远程精排抛错时——降级为本地打分且**大声记 warning**，

    而不是静默把本地分当真精排返回（remote 仍是 True，日志是上层唯一的降级线索）。
    """
    import httpx

    rr = RerankerClient(endpoint="http://reranker.invalid/score")
    assert rr.remote is True

    async def _boom(query: str, candidates: list[str]) -> list[float]:
        raise httpx.ConnectError("connection refused")

    rr._score_remote = _boom  # type: ignore[method-assign]
    cands = ["luggage suitcase roller", "ceramic coffee mug"]
    with caplog.at_level("WARNING", logger="shoppingx.reranker"):
        scores = await rr.score("luggage suitcase", cands)
    # 降级成本地打分：仍返回与候选同长的可排序结果，不抛。
    assert len(scores) == len(cands)
    assert scores == rr._score_local("luggage suitcase", cands)
    # 且留下了可观测的降级日志。
    assert any("降级" in r.message for r in caplog.records)


async def test_resolve_category_votes_by_category() -> None:
    """两段式第一段：hybrid 命中按品类投票——相关品类胜出，置信度是得分占比且降序。"""
    kb = KBClient(cards=_cards(), host=None)
    cands = await kb.resolve_category("luggage suitcase", top_n=2)
    assert cands and cands[0][0] == "luggage"
    confs = [c for _, c in cands]
    assert confs == sorted(confs, reverse=True)
    assert sum(confs) <= 1.0 + 1e-9


async def test_fetch_cards_exact_category_sorted_by_confidence() -> None:
    """两段式第二段：按品类精确取全该品类卡片（零跨品类污染），同品类按 confidence 降序。"""
    extra = CategoryCard(
        card_id="luggage_attr",
        category="luggage",
        card_type="attribute",
        summary="评分分布：4.5★+ 60% / 4.0–4.5★ 30% / 3.0–4.0★ 8% / <3.0★ 2%",
        confidence=0.95,
    )
    kb = KBClient(cards=[*_cards(), extra], host=None)
    cards = await kb.fetch_cards("luggage")
    assert {c.category for c in cards} == {"luggage"}
    assert len(cards) == 2
    # confidence 降序：0.95 的 attribute 卡排在 0.9 的爆款卡前。
    assert [c.card_id for c in cards] == ["luggage_attr", "luggage_bs"]


async def test_resolve_and_fetch_disambiguates_with_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """品类定位歧义（top2/top1 ≥ AMBIGUITY_RATIO）时，cross-encoder 对品类描述消歧定胜者。"""
    import app.tools.category_insight as ci

    sets = {
        "travel accessories": [
            CategoryCard(
                card_id="ta_bs",
                category="travel accessories",
                card_type="bestseller",
                summary="travel accessories: Packing Cubes",
                aliases=["旅行收纳", "packing cubes"],
            )
        ],
        "travel duffel bags": [
            CategoryCard(
                card_id="td_bs",
                category="travel duffel bags",
                card_type="bestseller",
                summary="travel duffel bags: Big Duffel",
                aliases=["旅行袋"],
            )
        ],
    }

    class FakeKB:
        async def resolve_category(self, query: str, top_n: int = 2):
            # 票仓接近 → 歧义（0.40/0.45 > AMBIGUITY_RATIO），且错的品类占先。
            return [("travel duffel bags", 0.45), ("travel accessories", 0.40)]

        async def fetch_cards(self, category: str):
            return sets[category]

    class FakeReranker:
        async def score_detailed(
            self, query: str, candidates: list[str]
        ) -> tuple[list[float], bool]:
            # 消歧文本 = 品类名+别名；按别名命中把 travel accessories 顶上来。
            return [1.0 if "旅行收纳" in c else 0.0 for c in candidates], True

    monkeypatch.setattr(ci, "get_kb_client", lambda: FakeKB())
    monkeypatch.setattr(ci, "get_reranker", lambda: FakeReranker())

    matched, conf, cards = await ci._resolve_and_fetch("旅行收纳")
    assert matched == "travel accessories"
    assert cards[0].card_id == "ta_bs"
    assert conf == 0.40  # 置信度如实取该候选的票仓占比，不因消歧胜出而虚增


def _gate_env(monkeypatch, votes, rerank_score, used_remote):
    """核验闸测试的公共桩：单候选低置信 + 可控的 rerank 分/远程标志。"""
    import app.tools.category_insight as ci

    card = CategoryCard(
        card_id="cc_bs", category=votes[0][0], card_type="bestseller",
        summary=f"{votes[0][0]}: Cleaner", aliases=["清洁剂"],
    )

    class FakeKB:
        async def resolve_category(self, query: str, top_n: int = 2):
            return votes

        async def fetch_cards(self, category: str):
            return [card]

    class FakeReranker:
        async def score_detailed(self, query, candidates):
            return [rerank_score] * len(candidates), used_remote

    monkeypatch.setattr(ci, "get_kb_client", lambda: FakeKB())
    monkeypatch.setattr(ci, "get_reranker", lambda: FakeReranker())
    return ci


async def test_offkb_low_relevance_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """库外品类：低置信 + rerank 分趋零 → 判未收录，返回空卡（不喂跨品类垃圾）。"""
    ci = _gate_env(monkeypatch, [("car care", 0.37)], rerank_score=0.0001, used_remote=True)
    matched, conf, cards = await ci._resolve_and_fetch("40oz 保温杯")
    assert matched == "" and cards == []
    assert conf == 0.37  # 如实保留占比，供上层上报


async def test_low_confidence_but_relevant_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """低置信但 rerank 判相关（金标里正确定位最低 0.014）→ 正常返回卡片。"""
    ci = _gate_env(monkeypatch, [("car care", 0.45)], rerank_score=0.014, used_remote=True)
    matched, _, cards = await ci._resolve_and_fetch("汽车内饰清洁")
    assert matched == "car care" and len(cards) == 1


async def test_gate_skipped_on_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """远程精排降级为本地 token 重叠打分时闸不生效——量纲不同，启用会误杀一切中文 query。"""
    ci = _gate_env(monkeypatch, [("car care", 0.37)], rerank_score=0.0, used_remote=False)
    matched, _, cards = await ci._resolve_and_fetch("汽车清洁")
    assert matched == "car care" and len(cards) == 1


async def test_high_confidence_skips_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """置信 ≥ GATE 不花 rerank 调用——rerank 桩若被调到会返回 0 分砍空，据此反证没调。"""
    ci = _gate_env(monkeypatch, [("car care", 0.80)], rerank_score=0.0, used_remote=True)
    matched, _, cards = await ci._resolve_and_fetch("car interior cleaner")
    assert matched == "car care" and len(cards) == 1


def test_thin_data_note_thresholds() -> None:
    """薄数据自报：卡少或置信低 → 非空文案；量足 → 空串；空卡不报（另有未收录硬路径）。"""
    from app.tools.category_insight import _thin_data_note

    assert _thin_data_note(2, 0.9) != ""  # 卡片少
    assert _thin_data_note(6, 0.3) != ""  # 卡片自评置信低
    assert _thin_data_note(6, 0.8) == ""  # 数据量正常
    assert _thin_data_note(0, 0.0) == ""  # 空卡走「未收录」路径，不重复报


def test_insight_result_renders_data_note_first() -> None:
    """data_note 渲染在人读摘要首行——主 loop 与前端先看到折扣提示，再看大盘数字。"""
    from app.tools.category_insight import CategoryInsightOutput, _insight_result

    out = CategoryInsightOutput(
        category="cat bed", matched_category="cat bed", resolution_confidence=0.8,
        components=[], bestsellers=[], attributes=[], attribute_schema=[],
        price_tiers=[], card_count=1, confidence=0.4, source="opensearch",
        data_note="库内该品类数据薄",
    )
    assert _insight_result(out).splitlines()[0].startswith("⚠ 库内该品类数据薄")


async def test_score_detailed_reports_remote_mode() -> None:
    """G2：score_detailed 如实返回「本次是否真走了远程」，供上层诚实上报 reranked。"""
    import httpx

    # 没配 endpoint → 本地打分，used_remote=False。
    local = RerankerClient(endpoint=None)
    _, used = await local.score_detailed("q", ["a", "b"])
    assert used is False

    # 配了 endpoint 但远程失败 → 降级本地，used_remote=False（不再谎报已精排）。
    rr = RerankerClient(endpoint="http://reranker.invalid/score")

    async def _boom(query: str, candidates: list[str]) -> list[float]:
        raise httpx.ConnectError("refused")

    rr._score_remote = _boom  # type: ignore[method-assign]
    scores, used = await rr.score_detailed("q", ["a", "b"])
    assert used is False and len(scores) == 2

    # 远程成功 → used_remote=True。
    async def _ok(query: str, candidates: list[str]) -> list[float]:
        return [0.9, 0.1]

    rr._score_remote = _ok  # type: ignore[method-assign]
    scores, used = await rr.score_detailed("q", ["a", "b"])
    assert used is True and scores == [0.9, 0.1]


def test_admit_gate() -> None:
    from scripts.etl.admit import admit

    good = {
        "card_id": "x_bestseller",
        "category": "x",
        "card_type": "bestseller",
        "summary": "x: a / b / c",
        "confidence": 0.8,
    }
    assert admit(good)[0] is True
    # 置信度不足被拦。
    assert admit({**good, "confidence": 0.3})[0] is False
    # bestseller 缺冒号前缀被拦。
    assert admit({**good, "summary": "a / b / c"})[0] is False
    # attribute 缺百分号被拦。
    attr = {**good, "card_id": "x_attr", "card_type": "attribute", "summary": "评分都很高"}
    assert admit(attr)[0] is False
