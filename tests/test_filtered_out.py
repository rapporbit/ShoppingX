"""批3-1：「库里有货，只是被挡住了」这条信号的全链（探测召回 → 提示语 → 补搜闸改口）。

要治的失败是**沉默的假话**：带预算过滤检索时，「库里没这个品类」和「库里有但都超预算」在返回体
里长得一模一样（total_recall 都是 0），模型只能猜，实测常把后者说成「没找到这类商品」——而用户
可能只差 20 美元。item_search 多打一次不带过滤的探测召回做差集，把被挡的样本如实回给模型；
harness 据此提示别把两者混为一谈；补搜闸拿到「预算内确实没货」的证据后，把「换条件重搜一次」
改成「问用户要不要放宽」——那一轮重搜是必然空手的。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.agent.retrieval_budget import budget_relax_due, note_filtered_probe
from app.recall.qdrant_store import QdrantRecall
from app.recall.schemas import ItemRecord
from app.recall.towers import TowerClient
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


async def _build_priced_recall(dim: int = 32) -> QdrantRecall:
    """三件同品类商品的临时索引：两件有价且超预算，一件**没有 price_usd**。

    最后那件是刻意的：历史上索引里真出现过 price_usd 缺失（见 price-usd-filter-empty-bug），
    而「拿不到价格」不等于「超预算」——探测不许对它下超预算的结论。
    """
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="P1",
            platform="amazon",
            title="canvas travel bag",
            brand="Nomad",
            price=20.0,
            price_usd=20.0,
            rating=4.6,
            category="bags",
            embed_text="canvas travel bag",
        ),
        ItemRecord(
            item_id="P2",
            platform="amazon",
            title="canvas duffel bag",
            brand="Trek",
            price=45.9,
            price_usd=45.9,
            rating=3.2,
            category="bags",
            embed_text="canvas duffel bag",
        ),
        ItemRecord(
            item_id="P3",
            platform="amazon",
            title="canvas tote bag",
            brand="Local",
            price=8.0,
            price_usd=None,  # 缺价：探测不许说它超预算
            rating=4.5,
            category="bags",
            embed_text="canvas tote bag",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)
    return recall


@pytest.fixture
def tiny_search(monkeypatch: pytest.MonkeyPatch) -> Any:
    """把 item_search 的召回 / 编码换成临时小索引。"""

    async def _setup() -> Any:
        import app.tools.item_search as mod

        recall = await _build_priced_recall()
        monkeypatch.setattr(mod, "get_recall_client", lambda: recall)
        monkeypatch.setattr(mod, "get_tower_client", lambda: TowerClient(model=None, local_dim=32))
        return mod

    return _setup


async def test_probe_reports_over_budget_items(tiny_search: Any, tmp_path: Path) -> None:
    """预算把候选杀光时，如实回「库里有这些，只是超预算」，而不是一个空返回体。"""
    mod = await tiny_search()
    with thread_scope("t-probe-budget", tmp_path):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "amazon", "price_usd_max": 5.0}
        )

    assert out.total_recall == 0  # 预算内一件都没有
    ids = {f.item_id for f in out.filtered_out}
    assert ids == {"P1", "P2"}  # 两件有价的被报，缺价的 P3 不下结论
    assert all("超预算" in f.reason for f in out.filtered_out)
    assert any("45.90" in f.reason for f in out.filtered_out)
    assert all("> $5）" in f.reason for f in out.filtered_out)  # 整数预算不拖小数尾
    # 策略可观测 + 证据要真进模型上下文（否则等于没做）。
    assert out.recall_strategy == "dense+price_filter+probe"
    rendered = str(out)
    assert "filtered_out" in rendered and "recall_strategy" in rendered


async def test_probe_never_rounds_the_user_budget(tiny_search: Any, tmp_path: Path) -> None:
    """小数预算原样显示：这句话模型会照抄给用户，$1.5 写成 $2 等于当面改他的硬约束。"""
    mod = await tiny_search()
    with thread_scope("t-probe-round", tmp_path):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "amazon", "price_usd_max": 1.5}
        )
    assert out.filtered_out
    assert all("> $1.50）" in f.reason for f in out.filtered_out)


async def test_probe_skipped_without_hard_filter(tiny_search: Any, tmp_path: Path) -> None:
    """没有任何硬过滤条件时不探测——差集必然为空，那一次查询是纯浪费。"""
    mod = await tiny_search()
    with thread_scope("t-probe-noop", tmp_path):
        out = await mod.item_search.ainvoke({"query": "canvas travel bag", "platform": "amazon"})

    assert out.total_recall > 0
    assert out.filtered_out == []
    assert out.recall_strategy == "dense"
    assert "filtered_out" not in str(out)  # 常态不占 token
    assert budget_relax_due() is False  # 没探测过就没有「预算内没货」的证据


async def test_probe_reason_prefers_structural_exclusion(tiny_search: Any, tmp_path: Path) -> None:
    """既踩品牌黑名单又超预算时报的是**黑名单**——报成超预算会推出「放宽预算就能买」的错结论。"""
    mod = await tiny_search()
    with thread_scope("t-probe-brand", tmp_path):
        out = await mod.item_search.ainvoke(
            {
                "query": "canvas travel bag",
                "platform": "amazon",
                "price_usd_max": 10.0,
                "brand_exclude": ["Nomad"],
            }
        )
        blocked = {f.item_id: f.reason for f in out.filtered_out}
        assert "Nomad" in blocked["P1"] and "超预算" not in blocked["P1"]
        # 混着结构性排除时不给「放宽预算」的结论：放宽了 P1 照样进不来。
        assert budget_relax_due() is False


async def test_filtered_out_never_enters_candidate_registry(
    tiny_search: Any, tmp_path: Path
) -> None:
    """被挡的货只是证据，不是候选：绝不能进登记表，否则会漏进清单 / 商品卡。"""
    from app.tools._candidates import enrich

    mod = await tiny_search()
    with thread_scope("t-probe-registry", tmp_path):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "amazon", "price_usd_max": 5.0}
        )
        assert out.filtered_out
        assert all(enrich(f.item_id) is None for f in out.filtered_out)


async def test_budget_relax_due_needs_price_only_and_zero_hits(tmp_path: Path) -> None:
    """判据的三条缺一不可（保守：它要否掉的是补搜，误判就等于放弃本可捞回的货）。"""
    with thread_scope("t-relax-1", tmp_path):
        note_filtered_probe(hits=0, price_blocked=3, other_blocked=0)
        assert budget_relax_due() is True
    with thread_scope("t-relax-2", tmp_path / "b"):
        note_filtered_probe(hits=0, price_blocked=3, other_blocked=1)
        assert budget_relax_due() is False  # 混着别的原因
    with thread_scope("t-relax-3", tmp_path / "c"):
        note_filtered_probe(hits=2, price_blocked=3, other_blocked=0)
        assert budget_relax_due() is False  # 捞到过预算内的货
    with thread_scope("t-relax-4", tmp_path / "d"):
        note_filtered_probe(hits=0, price_blocked=0, other_blocked=0)
        assert budget_relax_due() is False  # 没有「库里其实有货」的证据


async def test_nudge_tells_model_not_to_claim_nothing_found(tmp_path: Path) -> None:
    """探测有结论时，提示必须缀到 item_search 结果尾部——模型下一次解码就能读到。"""
    from app.harness.hooks.result_guard import append_nudges
    from app.harness.state import GuardState

    with thread_scope("t-nudge", tmp_path):
        ctx: dict[str, Any] = {
            "_guard": GuardState(),
            "tool_name": "item_search",
            "tool_result": '{"platform": "amazon", "total_recall": 0}',
            "call_filtered_out": [
                {"item_id": "P1", "title": "canvas travel bag", "reason": "超预算（$20.00 > $5）"}
            ],
            "call_filtered_price_only": True,
        }
        out = await append_nudges(ctx)

    assert out is not None
    text = out["tool_result"]
    assert "canvas travel bag" in text and "超预算" in text
    assert "ask_user" in text  # 只能问用户，不许自己放宽预算


async def test_nudge_absent_when_nothing_blocked(tmp_path: Path) -> None:
    """没被挡住任何货时不加提示（空 filtered_out 不许变成噪声）。"""
    from app.harness.hooks.result_guard import append_nudges
    from app.harness.state import GuardState

    with thread_scope("t-nudge-none", tmp_path):
        out = await append_nudges(
            {
                "_guard": GuardState(),
                "tool_name": "item_search",
                "tool_result": "{}",
                "call_filtered_out": [],
            }
        )
    assert out is None


async def test_backfill_gate_suggests_relaxing_instead_of_empty_research(tmp_path: Path) -> None:
    """探测已证明「预算内没货」→ 补搜闸不再回退重搜（那一轮必然空手），改口指路问用户。"""
    from app.api.context import get_retrieval_mode, set_retrieval_mode
    from app.harness.hooks.phase_transition import append_transition_notice, check_refine_backfill
    from app.harness.phase_machine import Phase, PhaseStateMachine, set_phase_machine
    from app.harness.state import GuardState

    with thread_scope("t-relax-gate", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")
        note_filtered_probe(hits=0, price_blocked=4, other_blocked=0)

        notice_ctx: dict[str, Any] = {
            "tool_name": "item_picker",
            "tool_result": '{"picks": []}',
            "call_picks": 0,
            "call_excluded": 0,
            "call_over_budget": 7,
            "_guard": GuardState(),
        }
        noticed = await append_transition_notice(notice_ctx)
        assert noticed is not None
        text = noticed["tool_result"]
        assert "ask_user" in text and "不要再重复检索" in text
        assert "price_usd_max" not in text  # 不许再指路「带预算重搜」

        await check_refine_backfill(
            {
                "picker_attempted": True,
                "picks_count": 0,
                "excluded_count": 0,
                "over_budget_count": 7,
                "_guard": GuardState(),
            }
        )
        # 阶段不退、mode 不改：这一轮补搜被证伪，留给模型去问用户。
        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "search"


async def test_diagnostics_side_channel_reaches_adapter(tiny_search: Any, tmp_path: Path) -> None:
    """接缝测试：工具登记的探测诊断，被 adapter 原样取成 call_filtered_out 交给 Hook。

    信号走侧信道而非模型可见文本——文本会被截断 Hook 砍尾，正则抠字段是历史上反复失效的老路。
    """
    from types import SimpleNamespace

    from app.harness.adapter import _collect_call_signals

    mod = await tiny_search()
    with thread_scope("t-probe-signal", tmp_path):
        out = await mod.item_search.ainvoke(
            {"query": "canvas travel bag", "platform": "amazon", "price_usd_max": 5.0}
        )
        signals = _collect_call_signals(
            SimpleNamespace(fresh_candidates=0),  # type: ignore[arg-type]
            "item_search",
            str(out),
        )

    assert len(signals["call_filtered_out"]) == len(out.filtered_out) == 2
    assert signals["call_filtered_price_only"] is True
