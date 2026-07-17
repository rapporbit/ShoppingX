"""M4 验收：九大工具可独立 ainvoke + 注册表完整。

分两类：
- 确定性工具（item_search / price_compare / shipping_calc / category_insight / item_picker）
  纯算力，直接断言数值与排序；需索引的两个用自建临时小索引，不依赖 data/。
- LLM 工具（planner / chat_fallback / shopping_summary）把 get_llm monkeypatch 成假模型，
  离线断言结构与数据流，不打真网络。
- web_search 断言「没 key 时优雅降级」。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.agent.platform_scope import platform_scope
from app.recall.qdrant_store import QdrantRecall
from app.recall.schemas import ItemRecord
from app.recall.towers import TowerClient
from app.tools.schemas import ItemCandidate


# --------------------------------------------------------------------------
# 注册表完整性
# --------------------------------------------------------------------------
def test_registry_complete() -> None:
    from app.agent.tool_registry import FULL_TOOL_SET, TERMINAL_TOOLS

    names = [t.name for t in FULL_TOOL_SET]
    # 九大业务工具 + 两个 dispatch 元工具。
    for expected in [
        "planner",
        "item_search",
        "price_compare",
        "shipping_calc",
        "category_insight",
        "item_picker",
        "web_search",
        "chat_fallback",
        "shopping_summary",
        "dispatch_tool",
        "parallel_dispatch_tool",
    ]:
        assert expected in names, f"{expected} 未注册"
    # 无重名（同质 fork 共用唯一集合，重名会让子 Agent 拿到歧义工具）。
    assert len(names) == len(set(names))
    # 终结性工具确实在工具集中。
    assert TERMINAL_TOOLS <= set(names)


# --------------------------------------------------------------------------
# price_compare：汇率归一 + 排序 + 未知币种降级
# --------------------------------------------------------------------------
def _cand(item_id: str, platform: str, price: float, currency: str, **kw: Any) -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, platform=platform, title=item_id, price=price, currency=currency, **kw
    )


async def test_price_compare_normalizes_ranks_and_skips() -> None:
    from app.tools.price_compare import price_compare

    cands = [
        _cand("A1", "amazon", 100, "USD"),  # 100 USD
        _cand("E1", "shein", 50, "EUR"),  # 54 USD（1.08）
        _cand("M1", "shopee", 1000, "MXN"),  # 58 USD（0.058）
        _cand("X1", "shopee", 999, "XYZ"),  # 未知币种 → skipped
    ]
    out = await price_compare.ainvoke({"candidates": [c.model_dump() for c in cands]})

    assert out.base_currency == "USD"
    assert "X1" in out.skipped
    # 升序：EUR(54) < MXN(58) < USD(100)，未知币种沉底。
    priced = [c for c in out.ranked if c.price_usd is not None]
    assert [c.item_id for c in priced] == ["E1", "M1", "A1"]
    assert priced[0].price_usd == pytest.approx(54.0)
    # 每平台最便宜：shopee 两条里 M1 能折算、X1 不能，故取 M1。
    assert out.cheapest_per_platform["shopee"] == "M1"


async def test_price_compare_fills_landed_cost_in_one_step() -> None:
    """刀3（延迟归因 round2）：price_compare 一步产出到手价，主链路无需再调 shipping_calc。"""
    from app.tools.price_compare import price_compare

    cands = [
        _cand("A1", "amazon", 100, "USD"),
        _cand("E1", "shein", 50, "EUR"),
    ]
    out = await price_compare.ainvoke({"candidates": [c.model_dump() for c in cands]})

    assert out.dest_country  # 收货国由机制兜底（默认国）
    for c in out.ranked:
        assert c.landed_usd is not None
        assert c.landed_usd == pytest.approx(c.price_usd + c.shipping_usd + c.duty_usd)
    # 排序口径已是到手价而非货价。
    landed = [c.landed_usd for c in out.ranked]
    assert landed == sorted(landed)
    # 喂给模型的紧凑形态里带「无需再调 shipping_calc」的直接指令。
    assert "shipping_calc" in str(out)


# --------------------------------------------------------------------------
# shipping_calc：到手价 = 货价 + 运费 + 关税，按到手价升序
# --------------------------------------------------------------------------
async def test_shipping_calc_landed_cost_and_sort() -> None:
    from app.tools.shipping_calc import shipping_calc

    cands = [
        ItemCandidate(
            item_id="A1",
            platform="amazon",
            title="shoes",
            category="shoes",
            price_usd=20.0,
            weight_kg=1.0,
        ),  # ship 10, duty 0(<免征额) → 30
        ItemCandidate(
            item_id="S1",
            platform="shopee",
            title="bag",
            category="bags",
            price_usd=40.0,
            weight_kg=1.0,
        ),  # 满50免邮? 40<50 → ship 12; duty 0 → 52
        ItemCandidate(item_id="N1", platform="amazon", title="no price"),  # 无 price_usd → uncosted
    ]
    # 显式指定收货国 JP（免征额 $130）：本例要验的是运费 + 排序 + uncosted 沉底，让两件都落在
    # 免征额以下、关税恒 0，把变量收敛到运费上。关税随国家变化另见 tests/test_landed_cost.py。
    out = await shipping_calc.ainvoke(
        {"dest_country": "JP", "candidates": [c.model_dump() for c in cands]}
    )

    assert "N1" in out.uncosted
    costed = [c for c in out.items if c.landed_usd is not None]
    # 升序：A1(30) < S1(52)，uncosted 沉底。
    assert [c.item_id for c in costed] == ["A1", "S1"]
    a1 = costed[0]
    assert a1.shipping_usd == pytest.approx(10.0)
    assert a1.duty_usd == pytest.approx(0.0)  # 20 USD < 日本免征额 130
    assert a1.landed_usd == pytest.approx(30.0)


# --------------------------------------------------------------------------
# item_picker：硬约束淘汰 + 预算筛 + 软偏好加分排序
# --------------------------------------------------------------------------
async def test_item_picker_filters_and_scores() -> None:
    from app.tools.item_picker import item_picker

    cands = [
        ItemCandidate(
            item_id="P1", platform="a", title="plastic travel bottle set", landed_usd=10, rating=4.0
        ),  # 命中排除词 plastic → 淘汰
        ItemCandidate(
            item_id="P2", platform="a", title="canvas niche travel pouch", landed_usd=30, rating=4.5
        ),  # 命中 canvas+niche，软偏好两连
        ItemCandidate(
            item_id="P3", platform="a", title="basic nylon travel pouch", landed_usd=25, rating=4.2
        ),  # 无软偏好命中
        ItemCandidate(
            item_id="P4", platform="a", title="canvas luxury bag", landed_usd=500, rating=5.0
        ),  # 超预算 → 淘汰
    ]
    out = await item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "budget_usd": 100,
            "exclude_keywords": ["plastic"],
            "prefer_keywords": ["canvas", "niche"],
            "top_k": 5,
        }
    )

    assert "P1" in out.excluded
    assert "P4" in out.over_budget
    ids = [c.item_id for c in out.picks]
    assert set(ids) == {"P2", "P3"}
    # P2 命中两个软偏好，应排在 P3 前。
    assert ids[0] == "P2"
    assert out.picks[0].pick_reason  # 有理由文案
    # 结构化偏好命中判据——summary 的 prompt 拿它判「有没有真对上偏好」，不再靠匹配
    # pick_reason 的措辞片段（措辞一改判据就静默失效）。
    assert out.picks[0].pref_matched is True  # P2 命中 canvas+niche
    assert out.picks[1].pref_matched is False  # P3 什么都没命中


async def test_item_picker_display_gate_drops_low_relevance(monkeypatch: Any) -> None:
    """展示相对门：cross-encoder 品类相关分显著低于池内头部的候选，宁缺毋滥不凑数展示。

    单平台池小时的防线——老的「按分填满 8」会把跨品类蹭词货塞进末位卡。门的降级路径
    （rerank_on=False 退化回填满）由别处覆盖，这里测门**生效**时：低分被挡 + 保底留头部。
    """
    import app.tools.item_picker as mod

    cands = [
        ItemCandidate(item_id="G1", platform="a", title="laptop backpack", landed_usd=30, rating=4.5),
        ItemCandidate(item_id="G2", platform="a", title="laptop sleeve bag", landed_usd=25, rating=4.3),
        ItemCandidate(
            item_id="G3", platform="a", title="backpack keychain sticker", landed_usd=5, rating=4.0
        ),  # 品类蹭词货：标题含 backpack 但其实是钥匙扣贴纸
    ]
    scores = {"G1": 0.9, "G2": 0.5, "G3": 0.05}  # peak 0.9 → floor 0.315，G3 低于门

    async def fake_rel(survivors: Any) -> Any:
        return {c.item_id: scores[c.item_id] for c in survivors}, True, False

    monkeypatch.setattr(mod, "_category_relevance", fake_rel)
    out = await mod.item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 8})
    ids = [c.item_id for c in out.picks]
    assert "G3" not in ids  # 相关分低于头部 35%，不凑数展示（哪怕没填满 top_k）
    assert "G1" in ids and "G2" in ids  # 品类相符的照常展示
    assert ids  # 保底：至少留头部一件，展示门不主动产空清单


async def test_item_picker_enforces_session_pt() -> None:
    """会话级 P_t 机制性强制：调用**不传** exclude_keywords / budget_usd，仍按 P_t 累积约束淘汰。

    这是把「续聊里上一轮说过的不要塑料 / 预算≤X」从 prompt 建议升为硬保证——item_picker 从
    ContextVar 读 P_t，硬 dislike 并入 exclude、预算兜底，不靠模型每轮把旧约束转述进本次调用。
    """
    from app.api.context import set_session_pt
    from app.memory.session_state import SessionConstraint, SessionPrefState
    from app.tools.item_picker import item_picker

    pt = SessionPrefState(
        budget_usd=100.0,
        constraints=[
            SessionConstraint(
                id="c1",
                category="material",
                content="不要塑料",
                polarity="dislike",
                keywords=["plastic"],
            ),
        ],
    )
    cands = [
        ItemCandidate(
            item_id="X1", platform="a", title="plastic bottle set", landed_usd=10, rating=4.0
        ),
        ItemCandidate(item_id="X2", platform="a", title="canvas pouch", landed_usd=30, rating=4.5),
        ItemCandidate(item_id="X3", platform="a", title="leather bag", landed_usd=500, rating=5.0),
    ]
    # P_t 按 session_dir 聚合（planner 要跨 context 写它，裸 ContextVar 传不出工具边界），
    # 故注入 P_t 必须在会话作用域内——真实链路本来就总有 session_dir。
    from app.utils.thread_ctx import thread_scope

    with thread_scope("t-pt-enforce", Path(tempfile.mkdtemp())):
        set_session_pt(pt)
        # 关键：调用不传 exclude_keywords / budget_usd，全靠 P_t 强制。
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 5})

    assert "X1" in out.excluded  # P_t 硬 dislike「塑料」淘汰（调用没传 exclude）
    assert "X3" in out.over_budget  # P_t 预算 100 淘汰（调用没传 budget_usd）
    assert [c.item_id for c in out.picks] == ["X2"]


class _FakeTower:
    """假三塔编码器：把负向意图词 + 其「语义近邻」映到同一单位向量，其余映到正交向量。

    确定性地测语义 Attenuator（论文式8）：``ugly``（负向意图）与 ``plasticky``（语义近邻、不含字面）
    都编成 [1,0]（cosine=1 → 减分）；``clean`` 编成 [0,1]（cosine=0）。
    """

    async def encode_texts(self, texts: list[str]) -> Any:
        import numpy as np

        rows = [[1.0, 0.0] if ("ugly" in t or "plasticky" in t) else [0.0, 1.0] for t in texts]
        return np.array(rows, dtype=np.float32)


async def test_item_picker_semantic_attenuator_penalizes_neighbor(monkeypatch: Any) -> None:
    """语义 Attenuator：与负向意图**语义近邻**但**不含字面关键词**的候选也被减分下沉。

    两候选价格/评分/关键词全同，只差语义：A 标题含 ``plasticky``（近邻，关键词 ``ugly`` 匹配不到）,
    B 含 ``clean``。纯关键词减分对两者都是 0；靠语义路 A 被减分压到 B 之后。
    """
    import app.tools.item_picker as mod

    monkeypatch.setattr(mod, "get_tower_client", lambda: _FakeTower())
    cands = [
        ItemCandidate(
            item_id="A", platform="p", title="metal case plasticky", landed_usd=20, rating=4.5
        ),
        ItemCandidate(
            item_id="B", platform="p", title="metal case clean", landed_usd=20, rating=4.5
        ),
    ]
    out = await mod.item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "deprioritize_keywords": ["ugly"],  # 关键词 ugly 两个标题都匹配不到 → 隔离出纯语义效应
            "top_k": 5,
        }
    )
    ids = [c.item_id for c in out.picks]
    assert ids == ["B", "A"], f"语义近邻 A 应被减分排到 B 之后，实际 {ids}"


async def test_item_picker_semantic_attenuator_degrades_on_encode_failure(monkeypatch: Any) -> None:
    """语义 Attenuator 编码失败 → 降级为纯关键词减分，不反噬主链路（照常出清单）。"""
    import app.tools.item_picker as mod

    class _BoomTower:
        async def encode_texts(self, texts: list[str]) -> Any:
            raise RuntimeError("encode boom")

    monkeypatch.setattr(mod, "get_tower_client", lambda: _BoomTower())
    cands = [
        ItemCandidate(item_id="A", platform="p", title="canvas pouch", landed_usd=20, rating=4.5)
    ]
    out = await mod.item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "deprioritize_keywords": ["塑料感"],
            "top_k": 5,
        }
    )
    assert [c.item_id for c in out.picks] == ["A"]  # 编码炸了也照常收敛


async def test_item_picker_semantic_matcher_boosts_neighbor(monkeypatch: Any) -> None:
    """语义 Matcher（论文式6）：与正向意图**语义近邻**但**不含字面关键词**的候选被加分上浮。

    正向意图 ``metalfeel``（复用 _FakeTower：含 ``ugly``/``plasticky`` → [1,0]，否则 [0,1]），故
    含 ``clean`` 的候选与 [0,1] 意图 cosine=1 加分。构造：A 含 ``clean``（近）、B 含 ``plasticky``
    （远）；prefer 关键词两者都匹配不到 → 纯语义把 A 顶到 B 前。
    """
    import app.tools.item_picker as mod

    monkeypatch.setattr(mod, "get_tower_client", lambda: _FakeTower())
    cands = [
        ItemCandidate(
            item_id="A", platform="p", title="metal case clean", landed_usd=20, rating=4.5
        ),
        ItemCandidate(
            item_id="B", platform="p", title="metal case plasticky", landed_usd=20, rating=4.5
        ),
    ]
    out = await mod.item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            # 关键词 metalfeel 两标题都匹配不到 → 隔离纯语义；_FakeTower 把正向意图映 [0,1]
            "prefer_keywords": ["metalfeel"],
            "top_k": 5,
        }
    )
    # A(clean)→[0,1] 与意图 [0,1] cosine=1 加分，B(plasticky)→[1,0] cosine=0 → A 顶到 B 前。
    assert [c.item_id for c in out.picks] == ["A", "B"]


async def test_item_picker_semantic_gated_off_without_soft(monkeypatch: Any) -> None:
    """无任何软意图（正/负都空）时**不触发**语义路（连编码都不做），验证门控省延迟。

    用调用标记检测（异常会被语义块 except 吞掉、测不到），候选只带硬 exclude、无软意图词。
    """
    import app.tools.item_picker as mod

    called = {"n": 0}

    class _SpyTower:
        async def encode_texts(self, texts: list[str]) -> Any:
            called["n"] += 1
            import numpy as np

            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(mod, "get_tower_client", lambda: _SpyTower())
    cands = [
        ItemCandidate(item_id="A", platform="p", title="canvas pouch", landed_usd=20, rating=4.5)
    ]
    out = await mod.item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "exclude_keywords": ["garbageword"],
            "top_k": 5,
        }
    )
    assert [c.item_id for c in out.picks] == ["A"]
    assert called["n"] == 0  # 无正/负软意图 → 编码器一次都没调


async def test_item_picker_must_have_high_weight_no_drop(monkeypatch: Any) -> None:
    """正向硬 must_have：比软偏好权重更高、强力上浮匹配项，但**不淘汰不匹配的**（无空结果）。

    关掉语义两路隔离关键词效应：A 命中 must「metal」（+W_MATCH_HARD=2）> B 命中 prefer「slim」（+1）
    > C 啥都不命中（0）；三者全保留——must 不做二值淘汰。
    """
    import app.tools.item_picker as mod

    monkeypatch.setattr(mod, "_W_MATCH_SEM", 0.0)
    monkeypatch.setattr(mod, "_W_MATCH_HARD_SEM", 0.0)  # 关语义，隔离关键词
    cands = [
        ItemCandidate(item_id="A", platform="p", title="metal case", landed_usd=20, rating=4.0),
        ItemCandidate(item_id="B", platform="p", title="slim case", landed_usd=20, rating=4.0),
        ItemCandidate(item_id="C", platform="p", title="basic case", landed_usd=20, rating=4.0),
    ]
    out = await mod.item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "must_have": ["metal"],
            "prefer_keywords": ["slim"],
            "top_k": 5,
        }
    )
    ids = [c.item_id for c in out.picks]
    assert ids[0] == "A"  # 正硬命中 +2 > 正软命中 +1
    assert set(ids) == {"A", "B", "C"}  # 不命中 must 的 C 未被淘汰（不空结果）
    assert out.must_have_hits == 1  # 池内命中 must 的只有 A


async def test_item_picker_reports_zero_must_hits(monkeypatch: Any) -> None:
    """must_have 池内全不命中 → picks 原样返回（不淘汰）但 must_have_hits=0。

    这是 refine_backfill 质量触发的信号源：复用轮「8 件素色对必须刺绣」件数看着正常，
    件数闸抓不住——靠这个字段抓。没传 must_have 时为 None（不适用，闸不触发）。
    """
    import app.tools.item_picker as mod

    monkeypatch.setattr(mod, "_W_MATCH_SEM", 0.0)
    monkeypatch.setattr(mod, "_W_MATCH_HARD_SEM", 0.0)
    cands = [
        ItemCandidate(
            item_id="A", platform="p", title="solid white set", landed_usd=20, rating=4.0
        ),
        ItemCandidate(item_id="B", platform="p", title="plain grey set", landed_usd=20, rating=4.0),
    ]
    payload = {"candidates": [c.model_dump() for c in cands], "top_k": 5}

    out = await mod.item_picker.ainvoke({**payload, "must_have": ["embroidery", "floral"]})
    assert [c.item_id for c in out.picks] and len(out.picks) == 2  # 全保留
    assert out.must_have_hits == 0  # 但质量信号如实报 0
    assert str(out).startswith('{"must_have_hits": 0')  # 序列化在头部，截断砍尾也抠得到

    out = await mod.item_picker.ainvoke(payload)
    assert out.must_have_hits is None  # 没传 must_have → 不适用
    assert '"must_have_hits"' not in str(out)  # None 不进模型可见文本


async def test_item_picker_pt_like_hard_routes_to_must(monkeypatch: Any) -> None:
    """会话级 P_t 的 like-hard 约束机制性路由到 must_have（正向硬强上浮，不淘汰不匹配的）。"""
    import app.tools.item_picker as mod
    from app.api.context import set_session_pt
    from app.memory.session_state import SessionConstraint, SessionPrefState

    monkeypatch.setattr(mod, "_W_MATCH_SEM", 0.0)
    monkeypatch.setattr(mod, "_W_MATCH_HARD_SEM", 0.0)
    pt = SessionPrefState(
        constraints=[
            SessionConstraint(
                id="c1",
                category="material",
                content="必须金属",
                polarity="like",
                keywords=["metal"],
            )
        ]
    )
    cands = [
        ItemCandidate(item_id="A", platform="p", title="plastic case", landed_usd=20, rating=4.0),
        ItemCandidate(item_id="B", platform="p", title="metal case", landed_usd=20, rating=4.0),
    ]
    from app.utils.thread_ctx import thread_scope

    with thread_scope("t-pt-must", Path(tempfile.mkdtemp())):
        set_session_pt(pt)  # P_t 按 session_dir 聚合 → 必须在会话作用域内注入
        out = await mod.item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "top_k": 5}
        )
    ids = [c.item_id for c in out.picks]
    assert ids == ["B", "A"]  # P_t like-hard「metal」把 B 强上浮
    assert "A" in ids  # 不含 metal 的 A 仍保留（不淘汰）


# item_picker：跳过 shipping_calc（无到手价）时优雅降级——按货价 price_usd 做预算筛与便宜度，
# 不崩、不把「没算到手价」误判成无价。对治意图化改造后「纯推荐不算运费」的常见路径。
async def test_item_picker_degrades_without_landed_price() -> None:
    from app.tools.item_picker import item_picker

    cands = [
        # 全部只有货价 price_usd、无 landed_usd（未跑 shipping_calc 的典型形态）。
        ItemCandidate(item_id="Q1", platform="a", title="canvas pouch", price_usd=30, rating=4.5),
        ItemCandidate(item_id="Q2", platform="a", title="nylon pouch", price_usd=25, rating=4.0),
        ItemCandidate(item_id="Q3", platform="a", title="leather bag", price_usd=200, rating=4.8),
    ]
    out = await item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "budget_usd": 100,  # 按货价硬筛：Q3(200) 超预算
            "prefer_keywords": ["canvas"],
            "top_k": 5,
        }
    )
    assert "Q3" in out.over_budget  # 缺到手价时用 price_usd 判超预算
    ids = {c.item_id for c in out.picks}
    assert ids == {"Q1", "Q2"}
    # 理由里用「售价」口径（非到手价），且不抛异常。「售价」是给用户看的说法，内部的「货价 /
    # price_usd」口径词不该出现在商品卡上。
    assert any("售价" in (c.pick_reason or "") for c in out.picks)
    assert not any("货价" in (c.pick_reason or "") for c in out.picks)


# item_picker：合适的候选全部展示，但封顶 PICK_DISPLAY_CAP（默认 20）
async def test_item_picker_caps_display_at_limit() -> None:
    from app.tools.item_picker import PICK_DISPLAY_CAP, item_picker

    # 30 件全合适（无排除词、无预算）→ 应只返回上限件数，不是全 30。
    cands = [
        ItemCandidate(item_id=f"C{i}", platform="a", title=f"canvas pouch {i}", landed_usd=10 + i)
        for i in range(30)
    ]
    # 不传 top_k：默认应等于展示上限（20→8：token 审计后收紧渲染件数，登记表仍全量）。
    out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands]})
    assert len(out.picks) == PICK_DISPLAY_CAP == 8
    # 模型即便传一个超大的 top_k，机制也封顶。
    out2 = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 999})
    assert len(out2.picks) == PICK_DISPLAY_CAP


async def test_item_picker_merges_near_duplicates() -> None:
    """同价 + 标题几乎全同的变体（颜色款）只留一件，名额顺延；不同卡口的同规格镜头**不**合并。

    后者用相机 bad case 的两支真实 Meike 35mm（Fuji X 口 vs Sony E 口）：各自兼容机型列表
    完全不同（Jaccard≈0.24），是不同商品——确定性层绝不能把它们当重复杀掉。
    """
    from app.tools.item_picker import item_picker

    meike_fuji = (
        "Meike 35mm f1.7 Large Aperture Manual Focus APSC Lens Compatible with Fujifilm X "
        "Mount Mirrorless Camera X-T3 X-H1 X-Pro2 X-E3 X-T1 X-T2 X-T4 X-T5 X-T10 X-T20"
    )
    meike_sony = (
        "Meike 35mm F1.7 Large Aperture Manual Focus Prime Fixed Lens APS-C Compatible with "
        "Sony E-Mount Mirrorless Cameras NEX 3 3N NEX 5R NEX 6 7 A6600 A6400 A5000 A5100"
    )
    cands = [
        # 同一款包的红 / 蓝配色：同价、标题只差颜色词 → 应合并（留 rating 高的红色款）。
        ItemCandidate(
            item_id="RED",
            platform="a",
            title="Travel Packing Cubes 3pc Set Lightweight Luggage Organizer Red",
            price_usd=19.99,
            rating=4.7,
        ),
        ItemCandidate(
            item_id="BLUE",
            platform="a",
            title="Travel Packing Cubes 3pc Set Lightweight Luggage Organizer Blue",
            price_usd=19.99,
            rating=4.5,
        ),
        ItemCandidate(item_id="FUJI", platform="a", title=meike_fuji, price_usd=69.99, rating=4.5),
        ItemCandidate(item_id="SONY", platform="a", title=meike_sony, price_usd=69.99, rating=4.3),
    ]
    out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands]})
    ids = {c.item_id for c in out.picks}
    assert "RED" in ids and "BLUE" not in ids  # 颜色变体合并，留分高的
    assert {"FUJI", "SONY"} <= ids  # 不同卡口是不同商品，不许合并


def test_item_search_output_collapses_known_candidates() -> None:
    """重试检索召回的「已入池」候选折叠成 id 列表：全量字段模型已看过，重复回显纯烧 token。"""
    import json as _json

    from app.tools.item_search import ItemSearchOutput

    cands = [
        ItemCandidate(item_id="NEW1", platform="amazon", title="Sony a6400 body", price_usd=898.0),
        ItemCandidate(item_id="OLD1", platform="amazon", title="Sony a6400 kit", price_usd=998.0),
    ]
    out = ItemSearchOutput(
        platform="amazon", candidates=cands, total_recall=2, truncated=False, known_ids=["OLD1"]
    )
    rendered = _json.loads(str(out))
    assert [c["item_id"] for c in rendered["candidates"]] == ["NEW1"]  # 只全文渲染新增
    assert rendered["already_in_pool"] == ["OLD1"]
    assert rendered["total_recall"] == 2  # 召回统计口径不变

    # 无重复时不带 already_in_pool 键（常态零成本）。
    out2 = ItemSearchOutput(platform="amazon", candidates=cands, total_recall=2, truncated=False)
    assert "already_in_pool" not in _json.loads(str(out2))


def test_item_picker_output_truncates_title_echo() -> None:
    """picks 回显标题截短成 handle（完整标题模型在检索结果里已看过；下游按 id hydrate）。"""
    import json as _json

    from app.tools.item_picker import ItemPickerOutput

    long_title = "Meike 35mm F1.7 Large Aperture Manual Focus Prime Fixed Lens " * 4
    out = ItemPickerOutput(
        picks=[ItemCandidate(item_id="X1", platform="a", title=long_title, price_usd=9.9)],
        excluded=[],
        over_budget=[],
    )
    rendered = _json.loads(str(out))
    assert len(rendered["picks"][0]["title"]) <= 61  # 60 + 省略号
    assert rendered["picks"][0]["title"].endswith("…")


async def test_item_picker_emits_items_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """先出货、后出文案：picker 一定稿就推 items_preview，卡片字段与收尾同构、且已含理由。

    这条事件是感知延迟的落点——收尾那轮解码 + summary 生成期间，用户已经在看卡片了。
    """
    import app.tools.item_picker as picker_mod

    seen: list[list[dict[str, object]]] = []

    async def fake_preview(items: list[dict[str, object]]) -> None:
        seen.append(items)

    monkeypatch.setattr(picker_mod.monitor, "report_items_preview", fake_preview)

    cands = [
        ItemCandidate(
            item_id="P1",
            platform="amazon",
            title="canvas travel pouch",
            landed_usd=30,
            price_usd=25,
            rating=4.6,
            url="https://amazon.example/P1",
            image_url="https://img.example/P1.jpg",
        ),
    ]
    out = await picker_mod.item_picker.ainvoke(
        {"candidates": [c.model_dump() for c in cands], "prefer_keywords": ["canvas"]}
    )

    assert len(seen) == 1
    (card,) = seen[0]
    assert card["item_id"] == "P1"
    assert card["platform"] == "amazon"
    assert card["landed_usd"] == 30  # 到手价与货价照实透传、互不兜底
    assert card["price_usd"] == 25
    assert card["url"] and card["image_url"]  # 图 / 链接直出，卡片能显能点
    assert card["reason"] == out.picks[0].pick_reason  # 理由 = picker 的确定性理由，不等 LLM


# --------------------------------------------------------------------------
# G1：item_picker 把当前用户的长期 dislike 黑名单确定性并入排除词
# --------------------------------------------------------------------------
async def test_item_picker_auto_excludes_user_dislikes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.memory.injector as injector_mod
    from app.memory.store import PreferenceEntry, get_store
    from app.tools.item_picker import item_picker
    from app.utils.thread_ctx import thread_scope

    # 用户在偏好页面亲手勾了「绝不推荐塑料」（source=user + blocking）——**只有这样**才有硬淘汰权。
    store = get_store()
    await store.write(
        "user-x",
        PreferenceEntry(
            slug="plastic",
            content="不要塑料",
            category="material",
            domain="global",
            polarity="dislike",
            keywords=["plastic"],
            source="user",
            blocking=True,
        ),
    )
    monkeypatch.setattr(injector_mod, "get_store", lambda: store)

    cands = [
        ItemCandidate(
            item_id="P1", platform="a", title="plastic travel bottle", landed_usd=10, rating=4.0
        ),
        ItemCandidate(
            item_id="P2", platform="a", title="canvas travel pouch", landed_usd=20, rating=4.5
        ),
    ]
    # 注意：调用方**没有**传 exclude_keywords=["plastic"]——黑名单应由长期偏好确定性兜住。
    with thread_scope("t-g1", tmp_path, user_id="user-x"):
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 5})
    assert "P1" in out.excluded  # 塑料候选被长期黑名单自动淘汰
    assert [p.item_id for p in out.picks] == ["P2"]

    # 匿名用户（无 user_id）则不应自动排除任何东西。
    with thread_scope("t-g1b", tmp_path):
        out2 = await item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "top_k": 5}
        )
    assert out2.excluded == []


# --------------------------------------------------------------------------
# A1：item_picker 的 Attenuator——软性避讳命中减分、但不淘汰（区别于 exclude 硬淘汰）
# --------------------------------------------------------------------------
async def test_item_picker_attenuates_soft_dislikes() -> None:
    from app.tools.item_picker import item_picker

    # 两件评分/价格相同，A 命中软避讳「花哨」、B 不命中 → B 排 A 前，但 A **仍在** picks。
    cands = [
        ItemCandidate(item_id="A", platform="a", title="flashy 花哨 pouch", landed_usd=20),
        ItemCandidate(item_id="B", platform="a", title="plain canvas pouch", landed_usd=20),
    ]
    out = await item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in cands],
            "deprioritize_keywords": ["花哨"],
            "top_k": 5,
        }
    )
    ids = [c.item_id for c in out.picks]
    assert out.excluded == []  # 软避讳不淘汰
    assert set(ids) == {"A", "B"}
    assert ids == ["B", "A"]  # 命中软避讳的 A 被减分、排到 B 后


# 长期 dislike 按 strength 分流：soft → Attenuator 减分不淘汰（对比 hard → 淘汰）
async def test_item_picker_auto_attenuates_user_soft_dislikes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.memory.injector as injector_mod
    from app.memory.store import PreferenceEntry, get_store
    from app.tools.item_picker import item_picker
    from app.utils.thread_ctx import thread_scope

    store = get_store()
    # curator 从对话里学到的 dislike（source=agent）——**永远只减分不淘汰**，无论它多确信。
    # 硬淘汰权只由用户在偏好页面显式授予（见上一条测试）。
    await store.write(
        "user-s",
        PreferenceEntry(
            slug="plastic",
            content="不太喜欢塑料感",
            category="material",
            domain="global",
            polarity="dislike",
            keywords=["plastic"],
            source="agent",
        ),
    )
    monkeypatch.setattr(injector_mod, "get_store", lambda: store)

    cands = [
        ItemCandidate(item_id="P1", platform="a", title="plastic pouch", landed_usd=20, rating=4.5),
        ItemCandidate(item_id="P2", platform="a", title="canvas pouch", landed_usd=20, rating=4.5),
    ]
    with thread_scope("t-soft", tmp_path, user_id="user-s"):
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 5})
    ids = [c.item_id for c in out.picks]
    assert out.excluded == []  # agent 学到的 dislike 不淘汰（只有用户勾的 blocking 才淘汰）
    assert set(ids) == {"P1", "P2"}
    assert ids == ["P2", "P1"]  # 命中软避讳的 P1 减分、排后


# --------------------------------------------------------------------------
# item_search / category_insight：自建临时小索引（不依赖 data/）
# --------------------------------------------------------------------------
async def _build_tiny_recall(dim: int = 32) -> QdrantRecall:
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="A1",
            platform="amazon",
            title="canvas travel bag",
            brand="Nomad",
            price=20.0,
            rating=4.6,
            reviews_count=900,
            category="bags",
            embed_text="canvas travel bag",
        ),
        ItemRecord(
            item_id="A2",
            platform="amazon",
            title="running shoe",
            brand="Saucony",
            price=60.0,
            rating=4.4,
            reviews_count=300,
            category="shoes",
            embed_text="running shoe",
        ),
        ItemRecord(
            item_id="S1",
            platform="shopee",
            title="canvas storage pouch",
            brand="Local",
            price=200.0,
            currency="MXN",
            rating=4.1,
            reviews_count=50,
            category="bags",
            embed_text="canvas storage pouch",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)
    return recall


async def test_item_search_returns_candidates(monkeypatch: Any) -> None:
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    out = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "amazon", "top_k": 5}
    )
    assert out.platform == "amazon"
    assert out.total_recall >= 1
    top = out.candidates[0]
    assert top.title == "canvas travel bag"
    assert top.platform == "amazon"
    assert top.score > 0
    # amazon 临时库只有 2 件，top_k=5 取不满 → 不算截断（honest 信号，非「恰好取满」）。
    assert out.truncated is False
    # top_k=1 而「all」（此处启用 amazon+shopee）有 2 件**相关**候选（两件 canvas，running shoe 被
    # 相关性下限滤掉）→ 确实还有更多相关项 → 截断。
    with platform_scope(["amazon", "shopee"]):
        one = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "all", "top_k": 1}
        )
    assert one.truncated is True and one.total_recall == 1


async def test_item_search_auto_relaxes_rating_when_recall_too_thin(monkeypatch: Any) -> None:
    """召回不足时工具**自己**摘掉评分门槛重搜，不把这个判断甩给模型（省掉一整轮 Think 的解码）。

    min_rating=4.5 会把 canvas storage pouch（4.1 分）挡在外面，只剩 1 件 → 低于 RETRY_MIN_HITS
    → 自动摘掉评分门槛重搜，把它捞回来，并置 relaxed=True 告诉模型「已放宽过，别再自己重搜一轮」。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    with platform_scope(["amazon", "shopee"]):
        strict = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "all", "top_k": 5, "min_rating": 4.5}
        )
    titles = [c.title for c in strict.candidates]
    assert strict.relaxed is True
    assert "canvas storage pouch" in titles  # 4.1 分的那件被放宽档捞了回来
    # 放宽这件事要让模型看见（投影里带 relaxed），否则它不知道这批候选是松过评分档的。
    assert '"relaxed": true' in str(strict)


async def test_item_search_relax_never_loosens_hard_constraints_or_relevance(
    monkeypatch: Any,
) -> None:
    """放宽只摘评分门槛，**不动用户硬约束、也不动相关度红线**。

    这是本机制最危险的失败模式：为了「凑够条数」，把用户明说不要的东西、或压根不沾边的跑题货
    塞回来充数——那比空手而归更糟（后者只是没找到，前者是幻觉）。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    # 预算硬约束：A1 售价 $20、S1 折 USD 后约 $10+，预算卡 $5 → 召回必然贫瘠，触发放宽重搜；
    # 但放宽档里预算照旧硬卡，绝不为了凑条数把超预算的塞回来。
    with platform_scope(["amazon", "shopee"]):
        out = await mod.item_search.ainvoke(
            {
                "query": "canvas travel bag",
                "platform": "all",
                "top_k": 5,
                "min_rating": 4.5,  # 逼出放宽路径
                "price_usd_max": 5.0,
            }
        )
    assert all((c.price_usd or 0) <= 5.0 for c in out.candidates)

    # 相关度红线：跑题的 running shoe 在放宽档里也不许出现（RELEVANCE_FLOOR 不参与放宽）。
    with platform_scope(["amazon", "shopee"]):
        out2 = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "all", "top_k": 5, "min_rating": 4.5}
        )
    assert "running shoe" not in [c.title for c in out2.candidates]

    # 品牌黑名单：Nomad 被排除后，放宽重搜也不许把它捞回来。
    with platform_scope(["amazon", "shopee"]):
        out3 = await mod.item_search.ainvoke(
            {
                "query": "canvas travel bag",
                "platform": "all",
                "top_k": 5,
                "min_rating": 4.5,
                "brand_exclude": ["Nomad"],
            }
        )
    assert all(c.brand.lower() != "nomad" for c in out3.candidates)


async def test_item_search_memory_exclusion_at_recall_stage(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """blocking 黑名单在 **item_search 内**就过滤（不等 item_picker 事后杀），且 total_recall
    反映排除后的真实召回数——web_search 兜底闸靠它判「召回全空」，排除前的假数字会把兜底拦死
    （候选全被黑名单杀光时，用户拿到空清单还无处补货）。memory_excluded 把「杀了几条」摆上台面。"""
    import app.tools.item_search as mod
    from app.memory.store import PreferenceEntry, get_store
    from app.utils.thread_ctx import thread_scope

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    # 用户亲手勾的「绝不推荐帆布」。domain=global：单测直调没有 planner，域为空时 fail-closed
    # 只放行 global——测的是排除机制本身，不是域闸。
    await get_store().write(
        "user-ms-recall",
        PreferenceEntry(
            slug="canvas",
            content="绝不推荐帆布",
            category="material",
            domain="global",
            polarity="dislike",
            keywords=["canvas"],
            source="user",
            blocking=True,
        ),
    )
    with thread_scope("t-ms-recall", tmp_path, user_id="user-ms-recall"):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "amazon", "top_k": 5}
        )
    assert out.memory_excluded >= 1  # canvas travel bag 在召回阶段就被黑名单杀掉
    assert all("canvas" not in c.title.lower() for c in out.candidates)
    assert out.total_recall == len(out.candidates)  # 计数 = 排除后的真实召回

    # 匿名（无 user_id）：黑名单不生效，canvas 候选原样返回。
    out2 = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "amazon", "top_k": 5}
    )
    assert out2.memory_excluded == 0
    assert any("canvas" in c.title.lower() for c in out2.candidates)


async def test_item_search_caps_model_supplied_top_k(monkeypatch: Any) -> None:
    """机制封顶：模型传的 top_k 只当**上界的候选**，真正生效的是 min(top_k, MAX_TOP_K)。

    实测模型会无视默认值自己传 ``top_k=20``（prompt 里给 item_picker 写的「top_k 缺省即 20」被它
    套到了本工具头上，换成没这句话的 prompt 变体后照样传）。默认值是建议，建议拦不住模型——照本
    仓库口径（边界用机制兜，见 fork 深度闸 / PICK_DISPLAY_CAP），这里必须硬收口，否则「候选体
    砍一半」这个决定会被模型一句 top_k=20 单方面推翻。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))
    monkeypatch.setattr(mod, "MAX_TOP_K", 1)  # 把封顶压到 1，好在小库上观察到收口

    with platform_scope(["amazon", "shopee"]):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "all", "top_k": 20}
        )

    assert out.total_recall == 1  # 模型要 20，机制只给 1
    assert out.truncated is True  # 且如实告诉模型「还有更多，是我截的」


async def test_item_search_clamps_to_enabled_platforms(monkeypatch: Any) -> None:
    """平台收口（机制层）：未启用的平台搜不到——``"all"`` 只等于「全部**启用**平台」，模型点名一个
    没启用的平台也会被拉回启用集合。默认单平台（amazon）时，shopee 的候选一件都不该出现。"""
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    with platform_scope(["amazon"]):
        # "all" 不再是全库：shopee 的 canvas storage pouch 不该被捞进来。
        allsearch = await mod.item_search.ainvoke(
            {"query": "canvas storage pouch", "platform": "all", "top_k": 5}
        )
        assert allsearch.candidates and all(c.platform == "amazon" for c in allsearch.candidates)
        # 模型点名未启用平台 → 回落启用集合（搜 amazon），而不是照搜 shopee。
        named = await mod.item_search.ainvoke(
            {"query": "canvas storage pouch", "platform": "shopee", "top_k": 5}
        )
        assert named.platform == "amazon"
        assert all(c.platform == "amazon" for c in named.candidates)

    # 勾上 shopee 后，同一次检索就能召回到它——收口的是「启用集合」，不是把平台写死。
    with platform_scope(["amazon", "shopee"]):
        opened = await mod.item_search.ainvoke(
            {"query": "canvas storage pouch", "platform": "shopee", "top_k": 5}
        )
    assert opened.platform == "shopee"
    assert [c.platform for c in opened.candidates] == ["shopee"]


async def test_item_search_relevance_floor_drops_irrelevant(monkeypatch: Any) -> None:
    """相关性下限：滤掉「最近邻但不相关」的召回；全无关时如实报空召回（触发兜底/空路径）。"""
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    # floor=0.45：query「canvas travel bag」下 running shoe(≈0.245) 被滤，canvas 类(≥0.45)留下。
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", 0.45)
    out = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "all", "top_k": 5}
    )
    titles = [c.title for c in out.candidates]
    assert "running shoe" not in titles  # 不相关项被下限滤掉
    assert out.candidates and all(c.score >= 0.45 for c in out.candidates)

    # floor 高到没有任何候选达标 → 相关召回为空（total_recall=0），下游据此走兜底/空召回硬路径。
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", 1.01)
    empty = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "all", "top_k": 5}
    )
    assert empty.total_recall == 0 and empty.candidates == []


async def test_item_search_target_name_filters_mismatched_model(monkeypatch: Any) -> None:
    """定点调查传 target_name 时，语义相关但型号不符的候选不算命中——复现真实 bad case：
    搜「Sony WH-1000XM5」却召回到完全不同的廉价型号「WH-CH710N」，型号过滤应把它剔除。
    """
    import app.tools.item_search as mod

    dim = 32
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="XM5",
            platform="amazon",
            title="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            brand="Sony",
            price=399.0,
            rating=4.5,
            reviews_count=1000,
            category="headphones",
            embed_text="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        ),
        ItemRecord(
            item_id="CH710N",
            platform="amazon",
            title="Sony Noise Canceling Headphones WHCH710N Wireless Bluetooth",
            brand="Sony",
            price=143.8,
            rating=4.4,
            reviews_count=6000,
            category="headphones",
            embed_text="Sony Noise Canceling Headphones WHCH710N Wireless Bluetooth",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)

    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: tower)
    # 型号过滤本身要单独测，用极低下限让两条语义候选都先通过 RELEVANCE_FLOOR。
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", -1.0)

    out = await mod.item_search.ainvoke(
        {
            "query": "Sony WH-1000XM5",
            "platform": "amazon",
            "top_k": 5,
            "target_name": "Sony WH-1000XM5",
        }
    )
    assert [c.item_id for c in out.candidates] == ["XM5"]
    assert out.total_recall == 1


async def test_item_search_target_name_no_model_token_no_filter(monkeypatch: Any) -> None:
    """target_name 解析不出型号 token（纯描述性文本，无型号数字）时不过滤——回归保护：
    候选与不传 target_name 时一致，不会误伤没有具体型号的商品名。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", 0.45)

    baseline = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "all", "top_k": 5}
    )
    with_target = await mod.item_search.ainvoke(
        {
            "query": "canvas travel bag",
            "platform": "all",
            "top_k": 5,
            "target_name": "帆布旅行包",
        }
    )
    assert [c.item_id for c in with_target.candidates] == [c.item_id for c in baseline.candidates]


async def _build_accessory_recall(dim: int = 32) -> tuple[QdrantRecall, TowerClient]:
    """定点调查场景常见的真实 bad case：型号 token 对得上，但其中一件是配件而非本体
    （复现 Bose QC45 调查真机命中的充电线——数据源里配件被分进 "Televisions Video Products"，
    跟耳机本体的 "Headphones Earbuds Accessories" 完全不是一类）。
    """
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="XM5",
            platform="amazon",
            title="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            brand="Sony",
            price=399.0,
            rating=4.5,
            reviews_count=1000,
            category="Headphones Earbuds Accessories",
            embed_text="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        ),
        ItemRecord(
            item_id="CABLE",
            platform="amazon",
            title="TPLTECH USB-C Charging Cable Cord for Sony WH-1000XM5 Headphones Charger",
            brand="TPLTECH",
            price=7.98,
            rating=4.5,
            reviews_count=0,
            category="Televisions Video Products",
            embed_text="TPLTECH USB-C Charging Cable Cord for Sony WH-1000XM5 Headphones Charger",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)
    return recall, tower


async def test_item_search_expected_category_drops_matching_model_wrong_category(
    monkeypatch: Any,
) -> None:
    """型号过滤挡不住的假阳性：配件标题带宿主型号，靠 expected_category 兜——
    真实复现过定点调查 Sony WH-1000XM5 / Bose QC45 时各自召回到同型号配件，误判"库内已找到"，
    连带拦掉了本该放行的 web_search 兜底（见 app.agent.retrieval_budget.web_search_allowed）。
    """
    import app.tools.item_search as mod

    recall, tower = await _build_accessory_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: tower)
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", -1.0)
    monkeypatch.setattr(mod, "CATEGORY_MATCH_FLOOR", 0.5)

    out = await mod.item_search.ainvoke(
        {
            "query": "Sony WH-1000XM5",
            "platform": "amazon",
            "top_k": 5,
            "target_name": "Sony WH-1000XM5",
            "expected_category": "headphones",
        }
    )
    assert [c.item_id for c in out.candidates] == ["XM5"]
    assert out.total_recall == 1


async def test_item_search_without_expected_category_keeps_old_behavior(
    monkeypatch: Any,
) -> None:
    """不传 expected_category（旧调用方 / 非定点场景）时跳过品类过滤——回归保护：
    只传 target_name 时，型号相符的配件仍然算命中（跟改动前行为一致），不静默改变旧调用方语义。
    """
    import app.tools.item_search as mod

    recall, tower = await _build_accessory_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: tower)
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", -1.0)

    out = await mod.item_search.ainvoke(
        {
            "query": "Sony WH-1000XM5",
            "platform": "amazon",
            "top_k": 5,
            "target_name": "Sony WH-1000XM5",
        }
    )
    assert {c.item_id for c in out.candidates} == {"XM5", "CABLE"}


async def test_item_search_expected_category_skips_candidates_missing_category(
    monkeypatch: Any,
) -> None:
    """候选自身没有 category 数据时不过滤（跟型号过滤"解析不出型号就不过滤"同一保守口径），
    避免数据缺失把真实候选误伤掉。
    """
    import app.tools.item_search as mod

    dim = 32
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="NOCAT",
            platform="amazon",
            title="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            brand="Sony",
            price=399.0,
            rating=4.5,
            reviews_count=1000,
            category="",
            embed_text="Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)

    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: tower)
    monkeypatch.setattr(mod, "RELEVANCE_FLOOR", -1.0)

    out = await mod.item_search.ainvoke(
        {
            "query": "Sony WH-1000XM5",
            "platform": "amazon",
            "top_k": 5,
            "target_name": "Sony WH-1000XM5",
            "expected_category": "headphones",
        }
    )
    assert [c.item_id for c in out.candidates] == ["NOCAT"]


async def test_item_search_orders_by_recall_similarity(monkeypatch: Any) -> None:
    """item_search 不再做 cross-encoder 精排：候选顺序由召回（融合相似度）决定。

    query="canvas travel bag" 应把同名候选排在第一，而不会被无关的 "running shoe" 挤掉
    ——验证移除 reranker 后仍是召回序直出（质量把关交给下游 item_picker）。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))

    out = await mod.item_search.ainvoke(
        {"query": "canvas travel bag", "platform": "all", "top_k": 5}
    )
    assert out.candidates[0].title == "canvas travel bag"


async def _recall_tower_with_query_spy(
    monkeypatch: Any,
) -> tuple[Any, list[str]]:
    """装好 item_search 的 recall/tower 依赖，并在 tower 上挂一个 encode_query 探针。

    返回 (item_search 模块, 记录 encode_query 入参的列表)——个性化现在是「把 like 偏好词拼进
    检索词」，所以断言的对象就是**最终进了编码器的那串文本**。这正是这条通路的好处：它可断言、
    可上报、可解释；换成原来的 user 向量画像，你只能断言「encode_user 被调用了」，至于它对召回
    做了什么，测试和用户都看不见。
    """
    import app.tools.item_search as mod

    recall = await _build_tiny_recall()
    tower = TowerClient(model=None, local_dim=32)
    seen: list[str] = []
    orig = tower.encode_query

    async def _spy(text: str) -> Any:
        seen.append(text)
        return await orig(text)

    monkeypatch.setattr(tower, "encode_query", _spy)
    monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
    monkeypatch.setattr(mod, "get_tower_client", lambda: tower)
    return mod, seen


async def _write_like_pref(uid: str, slug: str, keyword: str, domain: str) -> None:
    from app.memory.store import PreferenceEntry, get_store

    await get_store().write(
        uid,
        PreferenceEntry(
            slug=slug,
            content=f"喜欢{keyword}",
            category="material",
            domain=domain,  # type: ignore[arg-type]
            polarity="like",
            keywords=[keyword],
        ),
    )


async def test_item_picker_reports_memory_applied(monkeypatch: Any) -> None:
    """本轮用到了哪些长期记忆，必须**上报出去**——记忆最危险的失败是静默的。

    改造前 domain 字段「写了不读」的 bug 能长期潜伏，正是因为一条偏好没生效 / 误杀了一批商品，
    前端不会有任何提示；用户只觉得「这破 Agent 老是搜不出东西」，且归因不到记忆头上。
    """
    from app.api import monitor
    from app.memory.store import PreferenceEntry, get_store
    from app.tools.item_picker import item_picker
    from app.utils.thread_ctx import thread_scope

    seen: list[dict[str, Any]] = []

    async def _spy(domains: list[str], excluded: list[str], attenuated: list[str]) -> None:
        seen.append({"domains": domains, "excluded": excluded, "attenuated": attenuated})

    monkeypatch.setattr(monitor, "report_memory_applied", _spy)

    cands = [
        ItemCandidate(item_id="M1", platform="a", title="leather boots", landed_usd=50, rating=4.0),
        ItemCandidate(item_id="M2", platform="a", title="canvas shoes", landed_usd=40, rating=4.0),
    ]
    with thread_scope("t-mem", Path(tempfile.mkdtemp()), user_id="u-mem"):
        await get_store().write(
            "u-mem",
            PreferenceEntry(
                slug="leather",
                content="不要皮革",
                category="material",
                domain="global",
                polarity="dislike",
                keywords=["leather"],
                source="agent",  # curator 学到的 → 只减分
            ),
        )
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands]})

    assert seen and seen[0]["attenuated"] == ["leather"]
    assert seen[0]["excluded"] == []  # agent 学到的拿不到硬淘汰权
    assert out.excluded == []  # 皮靴仍在候选里，只是排后
    assert [p.item_id for p in out.picks] == ["M2", "M1"]


async def test_item_picker_reports_memory_even_if_model_restated_terms(monkeypatch: Any) -> None:
    """模型自己也传了同一个词时，记忆**照报不误**——否则事件恰好在最常见的路径上静默。

    偏好本来就注入了 prompt，模型多半会把 keywords 照抄进 exclude_keywords / deprioritize_keywords。
    若上报口径取「去重后新增的那部分」，这种时候就一条事件都不发：记忆真的杀了商品，用户却什么
    都看不到——正是这套系统要治的那个病。去重只该用于拼匹配列表，不该兼职当上报口径。
    """
    from app.api import monitor
    from app.memory.store import PreferenceEntry, get_store
    from app.tools.item_picker import item_picker
    from app.utils.thread_ctx import thread_scope

    seen: list[dict[str, Any]] = []

    async def _spy(domains: list[str], excluded: list[str], attenuated: list[str]) -> None:
        seen.append({"domains": domains, "excluded": excluded, "attenuated": attenuated})

    monkeypatch.setattr(monitor, "report_memory_applied", _spy)

    cands = [
        ItemCandidate(item_id="M1", platform="a", title="leather boots", landed_usd=50, rating=4.0),
        ItemCandidate(item_id="M2", platform="a", title="canvas shoes", landed_usd=40, rating=4.0),
    ]
    with thread_scope("t-mem2", Path(tempfile.mkdtemp()), user_id="u-mem2"):
        await get_store().write(
            "u-mem2",
            PreferenceEntry(
                slug="leather",
                content="绝对不要皮革",
                category="material",
                domain="global",
                polarity="dislike",
                keywords=["leather"],
                source="user",  # 用户亲手勾的「绝不推荐」 → 有硬淘汰权
                blocking=True,
            ),
        )
        out = await item_picker.ainvoke(
            {
                "candidates": [c.model_dump() for c in cands],
                "exclude_keywords": ["leather"],  # 模型把注入的偏好又转述了一遍
            }
        )

    assert seen and seen[0]["excluded"] == ["leather"]  # 模型重说一遍，不影响上报
    assert out.excluded == ["M1"]  # 皮靴被硬淘汰（用户授过权）
    assert [p.item_id for p in out.picks] == ["M2"]


# --------------------------------------------------------------------------
# 排除词匹配：词边界 + 否定修饰（真实误杀 —— "vegan leather" 正是不要皮革的人想要的）
# --------------------------------------------------------------------------
def test_hits_respects_word_boundary() -> None:
    from app.tools.item_picker import _hits

    assert _hits("pu", "pu leather bag")  # 完整单词 → 命中
    assert not _hits("pu", "canvas pouch")  # pouch 里的 pu 不算（裸子串会误杀）
    assert not _hits("bag", "baggage tag")


def test_hits_skips_negated_mentions() -> None:
    """「不要皮革」不该杀掉 vegan leather / 人造皮革 / leather-free —— 那些**正是**替代品。"""
    from app.tools.item_picker import _hits

    assert _hits("leather", "genuine leather boots")  # 真皮 → 该杀
    assert not _hits("leather", "vegan leather tote")  # 素皮 → 不该杀
    assert not _hits("leather", "faux-leather sneakers")
    assert not _hits("leather", "leather-free walking shoes")
    assert not _hits("皮革", "人造皮革单肩包")
    assert not _hits("plastic", "plastic-free travel kit")


def test_hits_chinese_negation_without_word_boundaries() -> None:
    """中文没有分词——否定词不会恰好是整个前缀，必须做后缀匹配。

    真实误杀（本次 code review 抓到）：``"2024新款人造皮革手提包"`` 的前段是 ``"2024新款人造"``，
    精确相等永远匹配不上 ``"人造"``，于是这件**人造革**包会被「不要皮革」杀掉。
    """
    from app.tools.item_picker import _hits

    assert not _hits("皮革", "2024新款人造皮革手提包")
    assert not _hits("皮革", "时尚仿皮革斜挎包")
    assert _hits("皮革", "头层牛皮革钱包")  # 真皮 → 该杀


def test_hits_english_negation_is_exact_token() -> None:
    """英文反过来：必须精确 token 匹配，不能用后缀——``piano`` 以 ``no`` 结尾。"""
    from app.tools.item_picker import _hits

    assert _hits("leather", "piano leather cover")  # piano 不是否定词，该杀


def test_hits_partially_negated_still_hits() -> None:
    """一处被否定、另一处没有 → 仍算命中（标题里真的有真皮部件）。"""
    from app.tools.item_picker import _hits

    assert _hits("leather", "vegan leather strap with genuine leather trim")


async def test_item_search_appends_like_prefs_to_query(monkeypatch: Any) -> None:
    """本轮域内的 like 偏好词被拼进检索词——个性化召回，且**看得见**。"""
    from app.api.context import set_session_domains
    from app.utils.thread_ctx import thread_scope

    mod, seen = await _recall_tower_with_query_spy(monkeypatch)
    with thread_scope("t-like", Path(tempfile.mkdtemp()), user_id="u-like"):
        await _write_like_pref("u-like", "canvas", "canvas", "bags")
        set_session_domains(["bags"])
        out = await mod.item_search.ainvoke({"query": "travel bag", "platform": "amazon"})
    assert seen == ["travel bag canvas"]  # 偏好词并入检索词
    assert out.total_recall >= 1


async def test_item_search_ignores_out_of_domain_prefs(monkeypatch: Any) -> None:
    """**域隔离**（本次重构的原始目的）：买包时，鞋类的偏好不该掺进检索词。

    改造前这条偏好会无差别地进 user 向量画像，把「买包」的请求向量往「鞋」那边拽。
    """
    from app.api.context import set_session_domains
    from app.utils.thread_ctx import thread_scope

    mod, seen = await _recall_tower_with_query_spy(monkeypatch)
    with thread_scope("t-xdomain", Path(tempfile.mkdtemp()), user_id="u-x"):
        await _write_like_pref("u-x", "suede", "suede", "footwear")  # 鞋类偏好
        set_session_domains(["bags"])  # 但本轮在买包
        await mod.item_search.ainvoke({"query": "travel bag", "platform": "amazon"})
    assert seen == ["travel bag"]  # 跨域偏好完全不参与——不加词、也不减分


async def test_item_search_no_prefs_is_pure_semantic(monkeypatch: Any) -> None:
    """匿名 / 无 like 偏好 → 检索词就是原始 query，退化为纯语义检索。"""
    mod, seen = await _recall_tower_with_query_spy(monkeypatch)
    await mod.item_search.ainvoke({"query": "canvas travel bag", "platform": "amazon"})
    assert seen == ["canvas travel bag"]


async def test_category_insight_rag_pipeline(monkeypatch: Any) -> None:
    """category_insight 走 RAG：注入三类卡片，验证「召回→分组提炼」出结构化结论。"""
    import app.tools.category_insight as mod
    from app.recall.category_kb import CategoryCard
    from app.recall.kb_client import KBClient

    cards = [
        CategoryCard(
            card_id="luggage_bestseller",
            category="luggage",
            card_type="bestseller",
            summary="luggage: Big Roller / Carry On Spinner",
            raw_evidence=[
                "Big Roller | $59.00 | 4.5★ | 月销 1000",
                "Carry On Spinner | $89.00 | 4.3★ | 月销 500",
            ],
            confidence=0.9,
        ),
        CategoryCard(
            card_id="luggage_attribute",
            category="luggage",
            card_type="attribute",
            summary="评分分布：4.5★+ 60% / 4.0–4.5★ 30% / 3.0–4.0★ 5% / <3.0★ 5%",
            confidence=0.9,
        ),
        CategoryCard(
            card_id="luggage_price_range",
            category="luggage",
            card_type="price_range",
            summary="budget $10.00–$50.00 / mid $50.00–$120.00 / premium $120.00–$400.00",
            confidence=0.9,
        ),
        CategoryCard(
            card_id="luggage_attribute_schema",
            category="luggage",
            card_type="attribute_schema",
            summary="属性维度：Bag/Case material / Bag/Case features",
            raw_evidence=[
                "映射自 Shopify 品类：Luggage & Bags > Suitcases（相似度 0.71）",
                "Bag/Case material: Canvas, Leather, Nylon, Polyester, Plastic",
                "Bag/Case features: Anti-theft, Convertible, Dustproof, Four-wheel spinner",
            ],
            confidence=0.71,
        ),
    ]
    # host=None → 走本地后端、source 为 local_kb_fallback（与外部 .env 无关，确定性）。
    kb = KBClient(cards=cards, host=None)
    monkeypatch.setattr(mod, "get_kb_client", lambda: kb)

    out = await mod.category_insight.ainvoke({"category": "luggage", "depth": "deep"})
    assert out.source == "local_kb_fallback"
    assert out.card_count == 4
    # 爆款卡提炼出组件与结构化爆款。
    assert "Big Roller" in out.components
    assert out.bestsellers[0].title == "Big Roller"
    assert out.bestsellers[0].price_usd == 59.0
    assert out.bestsellers[0].rating == 4.5
    # 价格卡解析出三档。
    assert {t.tier for t in out.price_tiers} == {"budget", "mid", "premium"}
    mid = next(t for t in out.price_tiers if t.tier == "mid")
    assert (mid.low_usd, mid.high_usd) == (50.0, 120.0)
    # deep 模式算评分分布。
    assert out.attributes and out.attributes[0].distribution["4.5★+"] == 0.6
    # 属性骨架卡：解析出选购维度 + 取值，provenance 行（全角冒号）被自动跳过。
    schema_names = {a.name for a in out.attribute_schema}
    assert schema_names == {"Bag/Case material", "Bag/Case features"}
    mat = next(a for a in out.attribute_schema if a.name == "Bag/Case material")
    assert "Plastic" in mat.values  # 「不要塑料的」可落到此维度取值
    assert all("映射自" not in n for n in schema_names)


async def test_category_insight_quick_skips_attributes(monkeypatch: Any) -> None:
    """quick 模式不算属性分布（省一次提炼），但爆款/价格仍出。"""
    import app.tools.category_insight as mod
    from app.recall.category_kb import CategoryCard
    from app.recall.kb_client import KBClient

    cards = [
        CategoryCard(
            card_id="mug_attr",
            category="mug",
            card_type="attribute",
            summary="评分分布：4.5★+ 70% / 4.0–4.5★ 20% / 3.0–4.0★ 5% / <3.0★ 5%",
            confidence=0.8,
        ),
    ]
    kb = KBClient(cards=cards, host=None)
    monkeypatch.setattr(mod, "get_kb_client", lambda: kb)

    out = await mod.category_insight.ainvoke({"category": "mug", "depth": "quick"})
    assert out.attributes == []


# --------------------------------------------------------------------------
# web_search：没 key 时优雅降级
# --------------------------------------------------------------------------
async def test_web_search_degrades_without_key(monkeypatch: Any) -> None:
    from app.tools.web_search import web_search

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    out = await web_search.ainvoke({"query": "best travel pouch 2026"})
    assert out.results == []
    assert "TAVILY_API_KEY" in out.note


# --------------------------------------------------------------------------
# LLM 工具：monkeypatch get_llm 成假模型，离线断言结构与数据流
# --------------------------------------------------------------------------
class _FakeStructured:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def ainvoke(self, _messages: Any, config: Any = None) -> Any:
        return self._payload


class _FakeLLM:
    """假模型：with_structured_output 回固定结构；ainvoke 回带 .content 的对象。

    ``config`` 参数不能省：真实工具会传 ``config={"callbacks": [usage_cb]}`` 收集 usage 入账
    （token_budget.charge_tool_llm_usage），假模型签名跟不上会 TypeError。
    """

    def __init__(self, structured_payload: Any = None, content: str = "") -> None:
        self._structured_payload = structured_payload
        self._content = content

    def with_structured_output(self, _schema: Any, **kwargs: Any) -> _FakeStructured:
        # 真实调用钉了 method="function_calling"（见 planner.py 的注释）；替身照单全收并记下来，
        # 免得日后被人悄悄改回默认 method 而测试还是绿的。
        self.structured_kwargs = kwargs
        return _FakeStructured(self._structured_payload)

    async def ainvoke(self, _messages: Any, config: Any = None) -> Any:
        class _Resp:
            content = self._content

        return _Resp()


def test_planner_demotes_weak_exclude_by_evidence() -> None:
    """主闸：evidence 里带弱表达标记 → 该词踢出硬排除、降级为软避讳。

    实测模型约 1/4 概率把「尽量别太花哨」硬归进排除。档位交给机制判（扫原话），不问模型——它转述
    原话是稳的，判档位不稳。**英文词跟着一起降**是这个设计的关键：flashy 自己看不出强弱，但它的
    evidence 同样是那句中文，所以纯字面比对失效的地方，evidence 兜住了。
    """
    from app.tools.planner import ExcludeTerm, PlanOutput

    p = PlanOutput(
        exclude_terms=[
            ExcludeTerm(word="塑料", evidence="不要塑料的"),
            ExcludeTerm(word="plastic", evidence="不要塑料的"),
            ExcludeTerm(word="花哨", evidence="尽量别太花哨"),
            ExcludeTerm(word="flashy", evidence="尽量别太花哨"),
        ],
    )
    assert p.exclude_keywords == ["塑料", "plastic"]
    assert p.soft_dislikes == ["花哨", "flashy"]  # 降级不是丢弃：仍然减分


def test_planner_demotes_keyword_present_in_both_buckets() -> None:
    """兜底闸：evidence 没写好时，靠「词同时出现在软桶里」这个犹豫证据接住。"""
    from app.tools.planner import ExcludeTerm, PlanOutput

    p = PlanOutput(
        exclude_terms=[
            ExcludeTerm(word="塑料", evidence="不要塑料的"),
            ExcludeTerm(word="Flashy", evidence=""),  # evidence 缺失，主闸抓不到
        ],
        soft_dislikes=["flashy"],
    )
    assert p.exclude_keywords == ["塑料"]  # 大小写不敏感地降级
    assert p.soft_dislikes == ["flashy"]  # 已在软桶里，不重复追加


def test_planner_naked_exclude_keywords_are_demoted_not_dropped() -> None:
    """模型无视「不要填」直接填了裸 exclude_keywords（老 schema 惯性）→ 降级，但**不丢**。

    没有 evidence 就不该硬淘汰；可也不能静默扔掉——「不要塑料」是用户明说的，凭空消失比降级更糟。
    """
    from app.tools.planner import PlanOutput

    p = PlanOutput(exclude_keywords=["塑料"])
    assert p.exclude_keywords == []  # 不硬淘汰
    assert p.soft_dislikes == ["塑料"]  # 但保留为减分项


async def test_planner_intent_grounding_passthrough(monkeypatch: Any) -> None:
    """intent_grounding：默认 internal；模型判 web 时原样透传（不被确定性回填逻辑覆盖）。

    web 信号是「检索前先 web_search 做意图翻译」的动机层提示——planner 判、主 loop 消费，
    系统不回填不改写，这里守住它不被 tasks / 币种 / 收货国等回填步骤误伤。
    """
    import app.tools.planner as mod
    from app.tools.planner import PlanOutput

    assert PlanOutput().intent_grounding == "internal"

    payload = PlanOutput(
        category="解压小物", tasks=["recommend"], intent_grounding="web", keywords=["fidget"]
    )
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    out = await mod.planner.ainvoke({"intent": "送女朋友一个今年最流行的那种解压小玩意"})
    assert out.intent_grounding == "web"


async def test_planner_returns_structured(monkeypatch: Any) -> None:
    import app.tools.planner as mod
    from app.tools.planner import ExcludeTerm, PlanOutput

    # 模型只填原始金额 budget_amount，不填币种 / budget_usd（那三个由系统确定性回填）。
    # 意图信号 tasks（本轮要做的事）与 target_refs（点名商品）由模型判、系统原样透传。
    payload = PlanOutput(
        category="旅行收纳",
        budget_amount=300.0,
        tasks=["recommend"],
        exclude_terms=[ExcludeTerm(word="塑料", evidence="不要塑料")],
        prefer_keywords=["小众"],
        keywords=["travel", "pouch"],
    )
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    out = await mod.planner.ainvoke(
        {"intent": "想买便宜抗造的旅行三件套，预算300，不要塑料，喜欢小众"}
    )
    assert out.category == "旅行收纳"
    # 意图里没写币种 → 默认 CNY、标注 assumed，budget_usd 由 fx 静态表确定性折算（300 CNY × 0.14）。
    assert out.currency == "CNY"
    assert out.currency_assumed is True
    assert out.budget_usd == pytest.approx(300 * 0.14)
    assert "塑料" in out.exclude_keywords
    # 意图信号原样透传（无点名商品）——不被货币回填逻辑覆盖。要推荐就自动补 landed_cost：
    # 这句话没提收货国 → 兜底默认国（assumed=True），照样算，由收尾文案讲明「按寄往中国估算」。
    assert out.tasks == ["recommend", "landed_cost"]
    assert out.dest_country_assumed is True
    assert out.target_refs == []


async def test_planner_auto_adds_landed_cost(monkeypatch: Any) -> None:
    """要推荐 → 自动补 landed_cost，用户没开口也把到手价算了。

    跨境购物里用户真正要的数字是「寄到我这儿一共多少」，但他往往不主动问——不能因此只给他一个
    还得自己心算运费关税的平台标价。收货国由四层解析保证永远有值（这里是本轮原话「寄到日本」）。
    """
    import app.tools.planner as mod
    from app.tools.planner import PlanOutput

    # 模型只判了 recommend（它的纪律仍是「只填用户明确表达的」，不许自作主张加 tasks）。
    payload = PlanOutput(category="旅行收纳", tasks=["recommend"], keywords=["packing", "cubes"])
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    out = await mod.planner.ainvoke({"intent": "推荐几个旅行收纳袋，寄到日本"})

    assert out.dest_country == "JP"
    assert out.dest_country_assumed is False  # 本轮原话 → 不是猜的
    assert out.tasks == ["recommend", "landed_cost"]  # 代码补的，不指望模型每轮判对


async def test_planner_skips_landed_cost_for_non_recommend(monkeypatch: Any) -> None:
    """只问品类行情（不挑商品）→ 哪怕收货国已知也不补到手价：没有候选可算，白跑一趟。"""
    import app.tools.planner as mod
    from app.tools.planner import PlanOutput

    payload = PlanOutput(category="旅行收纳", tasks=["category_intel"])
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    out = await mod.planner.ainvoke({"intent": "旅行收纳袋现在什么价位，我在日本"})

    assert out.dest_country == "JP"
    assert out.tasks == ["category_intel"]


def test_planner_atoms_drop_punctuation_only_tokens() -> None:
    """纯标点原子词必须在入口剔除（真实评测抓到 LLM 往 prefer 桶吐 "."）。

    term_hits 对非 ASCII-word 走子串路径——"." 对任何标题永远命中：落 like 桶是全池均匀
    加分（脏），落 exclude 桶就是整池屠杀。这不是美观问题，是失效方向问题。
    """
    from app.tools.planner import _atoms

    assert _atoms([".", "-", "…", "dark color", "深色", " "]) == ["dark color", "深色"]


async def test_planner_writes_session_pt_same_turn(monkeypatch: Any) -> None:
    """planner 识别出的本轮约束**当轮**就落进 P_t —— 短期记忆的机制执行通路。

    改造前 P_t 只由 curator 在会话收尾后写，本轮 item_picker 读到的永远是上一轮的：用户这轮
    亲口说的「不要塑料」机制侧一条没执行，全靠模型自觉转述进 exclude_keywords。
    """
    import app.tools.planner as mod
    from app.api.context import get_session_pt, set_session_pt
    from app.memory.session_state import SessionPrefState, load_pt
    from app.tools.planner import ExcludeTerm, PlanOutput
    from app.utils.thread_ctx import thread_scope

    payload = PlanOutput(
        category="旅行收纳",
        budget_amount=300.0,
        exclude_terms=[
            ExcludeTerm(word="塑料", evidence="不要塑料"),
            ExcludeTerm(word="plastic", evidence="不要塑料"),
        ],
        soft_dislikes=["花哨"],
        prefer_keywords=["帆布", "小众"],
    )
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))

    session_dir = Path(tempfile.mkdtemp())
    with thread_scope("t-pt", session_dir, user_id="u-pt"):
        set_session_pt(SessionPrefState())  # 开局空 P_t（首轮）
        await mod.planner.ainvoke({"intent": "想买旅行三件套，预算300，不要塑料，喜欢小众"})
        pt = get_session_pt()

    assert pt is not None
    # 三个桶 → 三条约束（**一桶一条**，不是一词一条：中英同义词属于同一件事，拆开只会让 P_t
    # 渲染给模型时出现「不要塑料」「不要plastic」两行几乎一样的噪声）
    assert len(pt.constraints) == 3
    # 硬排除词当轮可被 item_picker 机制淘汰（pt.dislike_terms() → exclude）
    assert pt.dislike_terms() == ["塑料", "plastic"]
    # 弱表达只减分、不淘汰
    assert pt.soft_dislike_terms() == ["花哨"]
    # 正向偏好当轮可被强加分（pt.like_terms() → must）
    assert set(pt.like_terms()) == {"帆布", "小众"}
    assert pt.budget_usd == pytest.approx(300 * 0.14)
    # turn 的唯一递增点是 planner 自己（P_t 单写者，递增点随写权一起挪过来；曾归 curator）
    assert pt.turn == 1
    # 落盘了，续聊轮 load_pt 能读回
    assert load_pt(session_dir).dislike_terms() == ["塑料", "plastic"]


async def test_planner_pt_reaches_item_picker_same_turn(monkeypatch: Any) -> None:
    """**接缝测试**：planner 写的 P_t，同一轮的 item_picker 读得到 —— 跨工具、跨 context。

    这是本改动唯一真正危险的地方。planner 和 item_picker 是两个工具、各自在独立 context 里跑，
    P_t 原先是裸 ContextVar：planner 里 set 的值 item_picker **读不到**（同一个坑 context.py 的
    _SESSION_DOMAINS 注释里已记过两次）。改成按 session_dir 聚合后才真正接上。

    验收口径：item_picker 调用**不传** exclude_keywords，塑料候选仍被淘汰——硬约束的执行不再
    依赖主 loop 的模型自觉转述。
    """
    import app.tools.planner as pmod
    from app.tools.item_picker import item_picker
    from app.tools.planner import ExcludeTerm, PlanOutput
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setattr(
        pmod,
        "get_fast_llm",
        lambda: _FakeLLM(
            structured_payload=PlanOutput(
                exclude_terms=[ExcludeTerm(word="塑料", evidence="不要塑料")]
            )
        ),
    )
    cands = [
        ItemCandidate(item_id="P1", platform="a", title="塑料收纳盒", price_usd=10, rating=4.5),
        ItemCandidate(item_id="P2", platform="a", title="帆布收纳袋", price_usd=20, rating=4.0),
    ]
    with thread_scope("t-seam", Path(tempfile.mkdtemp())):
        await pmod.planner.ainvoke({"intent": "买收纳，不要塑料"})
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands], "top_k": 5})

    assert "P1" in out.excluded  # planner 识别的硬排除词，当轮由机制执行
    assert [c.item_id for c in out.picks] == ["P2"]


async def test_planner_soft_dislike_penalizes_not_excludes(monkeypatch: Any) -> None:
    """「尽量别太花哨」→ 减分不淘汰；「不要塑料」→ 淘汰。档位记的是**用户的语气**。

    硬淘汰匹不准的词就是拿误杀去赌，所以弱表达只压排序、仍留在候选里（可能因其它维度胜出）。
    """
    import app.tools.item_picker as imod
    import app.tools.planner as pmod
    from app.tools.planner import PlanOutput
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setattr(imod, "_W_MATCH_SEM", 0.0)  # 关语义路，只看关键词档位
    monkeypatch.setattr(imod, "_W_ATTEN_SEM", 0.0)
    monkeypatch.setattr(
        pmod,
        "get_fast_llm",
        lambda: _FakeLLM(structured_payload=PlanOutput(soft_dislikes=["floral"])),
    )
    cands = [
        ItemCandidate(item_id="S1", platform="a", title="floral pouch", price_usd=10, rating=4.5),
        ItemCandidate(item_id="S2", platform="a", title="plain pouch", price_usd=10, rating=4.5),
    ]
    with thread_scope("t-soft", Path(tempfile.mkdtemp())):
        await pmod.planner.ainvoke({"intent": "买个收纳袋，尽量别太花哨"})
        out = await imod.item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "top_k": 5}
        )

    assert out.excluded == []  # 软避讳不淘汰
    assert [c.item_id for c in out.picks] == ["S2", "S1"]  # 只是被压到后面


@pytest.mark.parametrize(
    ("intent", "code", "explicit"),
    [
        ("预算 500", "CNY", False),  # 无符号 → 默认 CNY、非明示（答案需标注）
        ("预算 500 元", "CNY", True),
        ("预算 ¥500", "CNY", True),
        ("budget $500", "USD", True),
        ("预算 500 美元", "USD", True),  # 「美元」含「元」也不能误判成 CNY
        ("预算 500 欧元", "EUR", True),
        ("预算 €500", "EUR", True),
        ("budget S$500", "SGD", True),  # S$ 必须先于裸 $→USD
        ("预算 500 日元", "JPY", True),
        ("budget ₹500", "INR", True),
    ],
)
def test_resolve_budget_currency_deterministic(intent: str, code: str, explicit: bool) -> None:
    """币种解析纯规则、确定性：同一句话永远同一结果（修「每轮被猜成不同币种」的抖动）。"""
    from app.tools.planner import resolve_budget_currency

    # 连解析 5 次结果必须一致（验收口径：同一「预算 500」连跑 5 次币种一致）。
    results = {resolve_budget_currency(intent) for _ in range(5)}
    assert results == {(code, explicit)}


@pytest.mark.parametrize(
    ("intent", "amount", "grounded"),
    [
        # 追问轮没提任何数字：模型抄上文的 80 → 不落地（P_t 沿用上一轮 USD 预算，不重折）
        ("不要皮革的，最好防泼水", 80.0, False),
        ("预算 80 美元的通勤包", 80.0, True),
        ("预算提到 120", 120.0, True),
        ("预算 1,000 以内", 1000.0, True),  # 逗号千分位
        ("预算 1万 以内", 10000.0, True),  # 中文量级缩写
        ("预算 2k", 2000.0, True),  # 英文量级缩写
        ("预算三百左右", 300.0, True),  # 中文数词无法确定性核对 → 放行，不误删真预算
        ("能装 16 寸笔记本的包", 80.0, False),  # 有数字但对不上 → 16 是尺寸不是预算
    ],
)
def test_budget_amount_grounded(intent: str, amount: float, grounded: bool) -> None:
    """预算落地闸：模型填的 budget_amount 必须真在本轮原话里出现过，防「抄上文再按本轮
    默认币种重折」——真实 e2e 里 80 美元被追问轮当 80 人民币折成 $11.2。"""
    from app.tools.planner import budget_amount_grounded

    assert budget_amount_grounded(intent, amount) is grounded


async def test_chat_fallback_replies(monkeypatch: Any) -> None:
    import app.tools.chat_fallback as mod

    monkeypatch.setattr(mod, "get_llm", lambda: _FakeLLM(content="你好！我可以帮你跨平台找商品。"))
    out = await mod.chat_fallback.ainvoke({"message": "你好"})
    assert "你好" in out.reply


async def test_shopping_summary_returns_list(monkeypatch: Any) -> None:
    import app.tools.shopping_summary as mod
    from app.tools.shopping_summary import ShoppingSummaryOutput, _SummaryDraft

    # LLM 只产一段 summary；title/platform/到手价/图/链接/每件 reason 全由收尾确定性组装。
    payload = _SummaryDraft(summary="为你精选了 1 件：帆布旅行包。")
    # shopping_summary 文案走非推理快模型（perf/model-tiering）→ monkeypatch get_fast_llm。
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    # 候选带真实商品图，但 LLM 的结构化输出 item 里 image_url 为空——验证收尾按 item_id 回填。
    picks = [
        ItemCandidate(
            item_id="A1",
            platform="amazon",
            title="canvas bag",
            landed_usd=30.0,
            image_url="https://m.media-amazon.com/images/I/aaa.jpg",
            url="https://www.amazon.com/dp/A1",
        )
    ]
    # content_and_artifact：传 tool_call 形式 invoke 拿到 ToolMessage，结构化输出在 artifact。
    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {"picks": [c.model_dump() for c in picks], "user_intent": "旅行三件套"},
            "id": "call-1",
            "type": "tool_call",
        }
    )
    assert msg.content == "为你精选了 1 件：帆布旅行包。"  # content = 可读清单文案
    out = msg.artifact
    assert isinstance(out, ShoppingSummaryOutput)
    assert out.items[0].item_id == "A1"
    # 商品图 + 商品页链接按 item_id 从原候选回填（LLM 未生成，防 URL 幻觉），供卡片显图 + 点击跳转。
    assert out.items[0].image_url == "https://m.media-amazon.com/images/I/aaa.jpg"
    assert out.items[0].url == "https://www.amazon.com/dp/A1"
    # summary 已不产偏好（沉淀剥离给 curator）——输出结构里没有 new_preferences 字段。
    assert not hasattr(out, "new_preferences")


def _cam_picks() -> list[ItemCandidate]:
    """相机 bad case 形态的 picks：两台真机身 + 一个支架配件 + 一卷胶卷。"""
    return [
        ItemCandidate(
            item_id="CAM1",
            platform="amazon",
            title="Sony Alpha a6400 Mirrorless Camera 16-50mm Kit",
            price_usd=998.0,
        ),
        ItemCandidate(
            item_id="CAM2",
            platform="amazon",
            title="Sony Alpha a6400 Mirrorless Camera Body (Renewed)",
            price_usd=844.95,
        ),
        ItemCandidate(
            item_id="ACC1",
            platform="amazon",
            title="UURig Vlog Selfie Flip Screen Cold Shoe Bracket",
            price_usd=19.95,
        ),
        ItemCandidate(
            item_id="FILM",
            platform="amazon",
            title="Kodak Professional PORTRA 800 Color Film",
            price_usd=23.49,
        ),
    ]


async def test_shopping_summary_drops_off_intent_items(monkeypatch: Any) -> None:
    """draft LLM 判定 off-intent（配件 / 胶卷混进相机清单）→ 确定性摘除，主推区剩 ≥2 件时生效。"""
    import app.tools.shopping_summary as mod
    from app.tools.shopping_summary import _SummaryDraft

    payload = _SummaryDraft(summary="为你精选了两台相机。", off_intent=["ACC1", "FILM"])
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))

    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {
                "picks": [c.model_dump() for c in _cam_picks()],
                "user_intent": "日本品牌相机",
            },
            "id": "call-oi",
            "type": "tool_call",
        }
    )
    assert [i.item_id for i in msg.artifact.items] == ["CAM1", "CAM2"]


async def test_shopping_summary_off_intent_guard_never_empties_list(monkeypatch: Any) -> None:
    """护栏：off-intent 摘完主推区不足 2 件 → 一件不摘（LLM 判断绝不能把清单摘空）。"""
    import app.tools.shopping_summary as mod
    from app.tools.shopping_summary import _SummaryDraft

    payload = _SummaryDraft(summary="……", off_intent=["CAM1", "CAM2", "ACC1"])
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))

    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {"picks": [c.model_dump() for c in _cam_picks()], "user_intent": "相机"},
            "id": "call-oi2",
            "type": "tool_call",
        }
    )
    assert len(msg.artifact.items) == 4  # 全保留


async def test_shopping_summary_defaults_to_all_picker_picks(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """收尾不传 item_ids → 用 item_picker 定稿的**全部** picks，模型没机会再砍一遍。

    改前模型得把 id 逐个抄进入参，抄的时候会顺手「再精选几件最好的」——picker 挑了 10 件、
    收尾只进去 4 件（实测）。而它此刻既没有打分也没有排序依据；筛选是 picker 的职责。
    """
    import app.tools.shopping_summary as mod
    from app.tools._candidates import register
    from app.tools.item_picker import item_picker
    from app.tools.shopping_summary import ShoppingSummaryOutput, _SummaryDraft
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setattr(
        mod,
        "get_fast_llm",
        lambda: _FakeLLM(structured_payload=_SummaryDraft(summary="清单如下。")),
    )
    cands = [
        ItemCandidate(
            item_id=f"C{i}", platform="amazon", title=f"canvas pouch {i}", landed_usd=10 + i
        )
        for i in range(5)
    ]
    with thread_scope("t-picks", tmp_path):
        register(cands)  # 真实链路里由 item_search 登记；收尾按 id hydrate 全靠这张表
        picked = await item_picker.ainvoke({"item_ids": [c.item_id for c in cands]})
        assert len(picked.picks) == 5
        # pref_matched 已随 register_updates 回写登记表：收尾 hydrate 拿到的是判过的值。
        from app.tools._candidates import hydrate

        assert all(c.pref_matched is not None for c in hydrate([p.item_id for p in picked.picks]))
        # 模型什么都不传（连 item_ids 都没有）——收尾照样拿到全部 5 件。
        msg = await mod.shopping_summary.ainvoke(
            {
                "name": "shopping_summary",
                "args": {"user_intent": "旅行收纳"},
                "id": "call-1",
                "type": "tool_call",
            }
        )
    out = msg.artifact
    assert isinstance(out, ShoppingSummaryOutput)
    assert [i.item_id for i in out.items] == [c.item_id for c in picked.picks]  # 全部、且同序


async def test_shopping_summary_tripwire_on_empty_picks_with_candidates(
    tmp_path: Any, monkeypatch: Any, caplog: Any
) -> None:
    """绊线：候选池非空但本轮没精挑就收尾 → 记 ERROR（fail-loud），行为不变。

    机制上不该到达（phase_check 底线 3 已拦），触发即说明出现了新的绕闸路径——报警的意义
    是让下一条这样的路径从告警里现身，而不是等用户拿着自相矛盾的答案来问（badcase 63093a85）。
    对照组：picker 跑过但定稿为空（诚实空）不报警。
    """
    import app.tools.shopping_summary as mod
    from app.tools._candidates import register, set_last_picks
    from app.tools.shopping_summary import _SummaryDraft
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setattr(
        mod,
        "get_fast_llm",
        lambda: _FakeLLM(structured_payload=_SummaryDraft(summary="没找到合适的。")),
    )
    with thread_scope("t-tripwire", tmp_path):
        register([ItemCandidate(item_id="C1", platform="amazon", title="canvas pouch")])
        with caplog.at_level("ERROR", logger="shoppingx.tools.shopping_summary"):
            msg = await mod.shopping_summary.ainvoke(
                {"name": "shopping_summary", "args": {}, "id": "c1", "type": "tool_call"}
            )
        assert msg.artifact.items == []  # 绊线只报警，不改行为
        assert any("绕过本轮精挑" in r.message for r in caplog.records)

        caplog.clear()
        set_last_picks([])  # picker 定稿为空 = 诚实空清单，合法结论
        with caplog.at_level("ERROR", logger="shoppingx.tools.shopping_summary"):
            await mod.shopping_summary.ainvoke(
                {"name": "shopping_summary", "args": {}, "id": "c2", "type": "tool_call"}
            )
        assert not [r for r in caplog.records if "绕过本轮精挑" in r.message]


async def test_shopping_summary_never_passes_price_off_as_landed(monkeypatch: Any) -> None:
    """没跑 shipping_calc 的候选，landed_usd 必须留空——货价照实走 price_usd。

    曾经 landed 缺失时拿 price_usd 顶上，前端于是把没算过税运的货价标成「到手价（含税运）」。
    跨境单里这两个数差得远，等于骗用户。
    """
    import app.tools.shopping_summary as mod
    from app.tools.shopping_summary import _SummaryDraft

    payload = _SummaryDraft(summary="为你精选了 1 件。")
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    # 只有货价、没有到手价（本轮没调 shipping_calc）——真实场景里这是最常见的一种候选。
    picks = [ItemCandidate(item_id="A1", platform="amazon", title="canvas bag", price_usd=36.99)]
    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {"picks": [c.model_dump() for c in picks], "user_intent": "旅行三件套"},
            "id": "call-1",
            "type": "tool_call",
        }
    )
    item = msg.artifact.items[0]
    assert item.landed_usd is None  # 没算过税运 → 不谎报到手价
    assert item.price_usd == 36.99  # 货价照实透传，前端标「货价（未含税运）」


async def test_shopping_summary_backfills_url_from_registry(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """prod 主路径：picks 已不带 url（被紧凑序列化丢掉），url 从会话登记表按 item_id 回填。"""
    import app.tools.shopping_summary as mod
    from app.tools._candidates import register, reset_candidates
    from app.tools.shopping_summary import _SummaryDraft
    from app.utils.thread_ctx import thread_scope

    # LLM 只产 summary；url/image/reason 由收尾按 item_id 从登记表回填，不经模型。
    payload = _SummaryDraft(summary="精选 1 件。")
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))

    with thread_scope("t-reg", tmp_path):
        # item_search 阶段：把全量候选（含真实 url）登记到会话。
        register(
            [
                ItemCandidate(
                    item_id="A1",
                    platform="amazon",
                    title="canvas bag",
                    image_url="https://img/A1.jpg",
                    url="https://amazon.com/dp/A1",
                )
            ]
        )
        # 收尾时模型传回的 picks **不含 url**（已被紧凑序列化丢掉）——只有登记表能补。
        picks_no_url = [ItemCandidate(item_id="A1", platform="amazon", title="canvas bag")]
        msg = await mod.shopping_summary.ainvoke(
            {
                "name": "shopping_summary",
                "args": {"picks": [c.model_dump() for c in picks_no_url]},
                "id": "call-2",
                "type": "tool_call",
            }
        )
        out = msg.artifact
        assert out.items[0].url == "https://amazon.com/dp/A1"  # 来自登记表，非 picks
        assert out.items[0].image_url == "https://img/A1.jpg"
        reset_candidates()


# --------------------------------------------------------------------------
# 收尾文案流式生成（杠杆3）：增量提取 / 流式路径 / 降级路径
# --------------------------------------------------------------------------
class _FakeStreamChunk:
    def __init__(self, args: str) -> None:
        self.tool_call_chunks = [{"args": args}]


class _FakeStreamingLLM:
    """假流式模型：bind_tools 后 astream 逐片吐 tool-call args 分片。"""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    def bind_tools(self, _tools: Any, tool_choice: Any = None) -> _FakeStreamingLLM:
        return self

    async def astream(self, _messages: Any, config: Any = None, **_kw: Any) -> Any:
        for p in self._pieces:
            yield _FakeStreamChunk(p)


def test_strip_item_ids_replaces_with_names_not_holes() -> None:
    """文案里的商品 ID 一律换成商品名——**换名不是删字**，删了会把句子撕烂。

    ID 是内部主键，用户不认也用不上（卡片上本就有名字和链接）。prompt 已明令禁止，但模型只是
    「大多数时候」遵守（实测同一句 query 跑三次，一次写出了 ASIN），所以在出口处确定性兜死。
    """
    from app.tools.shopping_summary import strip_item_ids

    m = {"B098QCGR94": "Travel Packing Cubes 3pc Set", "B0765B6TZ1": "3-Piece Packing Cubes"}

    # ① 名字已在，括号里的 ID 是纯冗余 → 整块删掉，不留空括号。
    assert strip_item_ids("最推荐 **Cubes 3pc**（B098QCGR94）——到手 ¥130。", m) == (
        "最推荐 **Cubes 3pc**——到手 ¥130。"
    )
    # ② ID 占了名字的位置、后面括号里才是名字 → 用名字顶上（而不是留下 "** （名字）**"）。
    assert strip_item_ids("**B0765B6TZ1（3-Piece Packing Cubes）** 评分 4.8。", m) == (
        "**3-Piece Packing Cubes** 评分 4.8。"
    )
    # ③ 裸 ID → 换成商品名。裸删会得到「看 和 这两件」这种残句。
    assert strip_item_ids("看 B098QCGR94 和 B0765B6TZ1 这两件。", m) == (
        "看 Travel Packing Cubes 3pc Set 和 3-Piece Packing Cubes 这两件。"
    )
    # ④ 没有 ID 的正常文案原样放行（不该被后处理啃掉任何字）。
    assert strip_item_ids("这几件都在预算内，放心挑。", m) == "这几件都在预算内，放心挑。"
    assert strip_item_ids("空映射也不炸。", {}) == "空映射也不炸。"


def test_partial_summary_extracts_growing_text() -> None:
    """从未闭合的 JSON 缓冲里抠 summary 当前文本：转义还原、缺字段与悬垂转义不炸。"""
    from app.tools.shopping_summary import _partial_summary

    assert _partial_summary('{"summary": "为你精') == "为你精"
    assert _partial_summary('{"summary": "带\\"引号\\"和换行\\n') == '带"引号"和换行\n'
    assert _partial_summary('{"reasons": [') == ""  # summary 未出现 → 空串跳过本 tick
    assert _partial_summary('{"summary": "abc\\') == "abc"  # 悬垂半个转义不进组


async def test_stream_draft_emits_prefix_deltas(monkeypatch: Any) -> None:
    """流式路径：增量是最终文案的前缀（累计全文、幂等），收齐后结构化校验通过。"""
    import app.tools.shopping_summary as mod

    pieces = [
        '{"summary": "这批候选主打耐用',
        '与低调，都在预算内，可放心选。"}',
    ]
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeStreamingLLM(pieces))
    sent: list[str] = []

    async def _fake_delta(text: str) -> None:
        sent.append(text)

    monkeypatch.setattr(mod.monitor, "report_summary_delta", _fake_delta)
    draft = await mod._generate_draft([("system", "s"), ("user", "u")], {})
    assert draft.summary.endswith("放心选。")
    assert sent, "至少推送一条流式增量"
    for s in sent:
        assert draft.summary.startswith(s)  # 每条都是定稿的前缀


async def test_generate_draft_falls_back_without_streaming(monkeypatch: Any) -> None:
    """供应商 / 假模型不支持流式（无 bind_tools）→ 静默降级阻塞结构化，产物不变。"""
    import app.tools.shopping_summary as mod
    from app.tools.shopping_summary import _SummaryDraft

    payload = _SummaryDraft(summary="打底文案")
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    draft = await mod._generate_draft([("system", "s"), ("user", "u")], {})
    assert draft.summary == "打底文案"


class TestStringifiedListCoercion:
    """StrListArg（app/tools/_args.py）：模型把 list 参数吐成 JSON 字符串时就地解回。

    gcjp 会话 d0724e95（2026-07-16）：qwen3.5-flash 传 exclude_keywords='["塑料","plastic"]'
    （字符串不是数组），item_picker 校验连挂 4 次到被强制收尾。字符串形如 JSON 数组无歧义，
    机制层 loads 回来比指望模型自纠可靠（同 planner raw_decode 的教训）。
    """

    def test_json_array_string_coerced(self) -> None:
        from app.tools._args import coerce_stringified_list

        assert coerce_stringified_list('["塑料", "plastic"]') == ["塑料", "plastic"]
        assert coerce_stringified_list(' ["a"] ') == ["a"]  # 容首尾空白

    def test_ambiguous_and_normal_inputs_untouched(self) -> None:
        """裸字符串不猜（包单元素对空格分隔多 id 就是错的）；正常 list / None 原样。"""
        from app.tools._args import coerce_stringified_list

        assert coerce_stringified_list("塑料") == "塑料"  # 交回 Pydantic 照常报错
        assert coerce_stringified_list("[broken") == "[broken"
        assert coerce_stringified_list('{"a": 1}') == '{"a": 1}'
        assert coerce_stringified_list(["a"]) == ["a"]
        assert coerce_stringified_list(None) is None

    def test_online_badcase_args_now_validate(self) -> None:
        """线上实锤的那组参数原样过校验——修好的就是这一条。"""
        from app.tools.item_picker import item_picker

        m = item_picker.args_schema.model_validate(
            {
                "budget_usd": "56",
                "exclude_keywords": '["塑料", "plastic"]',
                "prefer_keywords": '["water resistant", "water repellent"]',
                "must_have": '["16 inch laptop", "commuter"]',
                "top_k": 8,
            }
        )
        assert m.exclude_keywords == ["塑料", "plastic"]
        assert m.must_have == ["16 inch laptop", "commuter"]

    def test_model_visible_schema_still_array(self) -> None:
        """给模型的 JSON Schema 仍是 array——容错不反向诱导模型传字符串。"""
        from app.tools.item_picker import item_picker

        prop = item_picker.args_schema.model_json_schema()["properties"]["exclude_keywords"]
        assert {"items": {"type": "string"}, "type": "array"} in prop["anyOf"]

class TestNullIsAbsent:
    """drop_none_values（app/tools/_args.py）：结构化输出的显式 null 归一为缺席。

    gcjp 会话 cdee1d6d（2026-07-17，童装追问「放开预算」）：deepseek-v4-flash（关思考 +
    function_calling）把 PlanOutput 的 5 个 list 字段吐成显式 null，default_factory 不接显式
    null，planner 连挂 2 次——报错还被 langgraph 包成「Error invoking tool with kwargs
    {外层入参}」，看着像入参问题。四处 with_structured_output 顶层 schema 全挂本容错。
    """

    def test_helper_drops_none_keys_only(self) -> None:
        from app.tools._args import drop_none_values

        assert drop_none_values({"a": None, "b": [], "c": 0, "d": False}) == {
            "b": [],
            "c": 0,
            "d": False,
        }
        assert drop_none_values("not a dict") == "not a dict"
        assert drop_none_values(None) is None

    def test_online_badcase_plan_output_now_validates(self) -> None:
        """线上实锤形态：5 个 list 字段显式 null → 默认值接管，不再 ValidationError。"""
        from app.tools.planner import PlanOutput

        plan = PlanOutput.model_validate(
            {
                "tasks": ["recommend"],
                "retrieval": "reuse",
                "category": "童装",
                "target_refs": None,
                "bundle_slots": None,
                "exclude_terms": None,
                "soft_dislikes": None,
                "keywords": None,
                "clear_budget": True,
            }
        )
        assert plan.bundle_slots == []
        assert plan.target_refs == []
        assert plan.clear_budget is True  # 非 None 字段不受影响

    def test_other_structured_output_schemas_covered(self) -> None:
        """同洞面积的另外三处 schema：null 一样归一，不臆造行为。"""
        from app.memory.curator import CurationResult
        from app.memory.parser import _ParseResult
        from app.tools.shopping_summary import _SummaryDraft

        draft = _SummaryDraft.model_validate(
            {"summary": "文案", "reasons": None, "off_intent": None}
        )
        assert draft.reasons == [] and draft.off_intent == []
        assert _ParseResult.model_validate({"preferences": None}).preferences == []
        cur = CurationResult.model_validate({"persistent_preferences": None})
        assert cur.persistent_preferences == []
