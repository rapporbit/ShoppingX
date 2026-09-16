"""候选池跨轮存活：追问（「我只要防水的」）能拿到上一轮搜过的商品，不必重新检索一遍。

候选池原本每轮 reset，而跨轮只回喂 (role, content) 文本、工具消息全丢——第二轮的 Agent 手里既没有
候选登记表也没有 item_id 列表，物理上没法「在上一步基础上过滤」。落盘 + 开局读回后，下游工具按
item_id hydrate 得到上一轮的候选。

注意：「有候选」≠「本轮是追问轮」——用户完全可能换品类。本模块只保证候选**可用**，「这轮要不要
重搜」是 planner 的判断，不由候选池的有无来猜。
"""

from pathlib import Path

import pytest

from app.harness.hooks.progress import BACKFILL_LATCH, check_refine_backfill
from app.harness.phase_machine import Phase, PhaseStateMachine, set_phase_machine
from app.harness.signals import _count_candidates, candidate_count
from app.harness.state import GuardState
from app.tools._candidates import register
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio


def _cand(item_id: str = "A1") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id,
        platform="amazon",
        title=f"Waterproof Travel Bag {item_id}",
        price=19.9,
        currency="USD",
        price_usd=19.9,
        rating=4.5,
        url=f"https://example.com/{item_id}",
        image_url=f"https://img.example.com/{item_id}.jpg",
    )


# ---------- 阶段推进只认工具返回，不数登记表 ----------


def test_candidate_signal_counts_tool_return_not_registry(tmp_path: Path) -> None:
    """本轮新召回数从 item_search 的返回里数。

    登记表是累积容器（跨轮候选也在里面），拿它的总数当「本轮搜到了东西」的进展信号，换品类那轮
    就会被旧候选骗过去——阶段机直接推进 COMPARING，模型想搜新品类却发现 item_search 不放行。
    """
    with thread_scope("t-signal", tmp_path):
        register([_cand("OLD1"), _cand("OLD2")])  # 上一轮读回的旧候选

        assert candidate_count() == 2  # 仓库里确实有货（供 item_picker hydrate）
        assert _count_candidates('{"platform":"amazon","candidates":[]}') == 0  # 但本轮啥也没搜到
        assert _count_candidates('{"candidates":[{"item_id":"NEW1"}]}') == 1
        assert _count_candidates("子 Agent 回传的自然语言总结") == 0  # 数不出来就不算，宁可少算


# ---------- 补搜闸：污染 / 硬淘汰杀池，一轮只补一次 ----------


# ---------- 污染分支：首搜轮品类门吃空池子 → 补搜（手表 badcase 6718ed65） ----------


async def test_polluted_first_search_triggers_backfill(tmp_path: Path) -> None:
    """首搜轮（search，非 reuse）：品类门判 10 条里 8 条跨品类混入、相符只剩 2 → 退回补搜。

    重现手表 badcase：检索词 formal dress watch men business 召回一堆西装皮鞋，品类一致性门
    正确沉底 8 条，但旧闸只认 reuse 轮，没人补货，径直收尾出了 2 件的清单。
    """

    with thread_scope("t-polluted", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        guard = GuardState()

        ctx = await check_refine_backfill(
            {
                "picker_attempted": True,
                "picks_count": 8,  # 沉底的垃圾也占 picks 名额——件数信号对污染是瞎的
                "oncat_count": 2,
                "offcat_count": 8,
                "total_candidates": 10,
                "_guard": guard,
            }
        )

        assert machine.phase is Phase.SEARCHING
        assert BACKFILL_LATCH in guard.notified_transitions  # 只触发一次，不会无限回退
        # 污染批不再算「本轮已搜到货」：同轮 40 号钩子不得凭它把 SEARCHING 立刻推回 COMPARING。
        assert ctx is not None and ctx["total_candidates"] == 0
        assert ctx["reset_fresh_candidates"] is True


async def test_sparse_but_clean_pool_no_backfill(tmp_path: Path) -> None:
    """池子小但干净（oncat=2、offcat=0）是库存稀疏，不是检索词的错——重搜同样的词只会拿回
    同一池货，不触发。"""
    with thread_scope("t-sparse", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 2, "oncat_count": 2, "offcat_count": 0}
        )

        assert machine.phase is Phase.COMPARING


# ---------- 硬淘汰杀池分支：预算/排除把干净池杀空 → 补搜（交接遗留洞 #1） ----------


async def test_hard_cull_first_search_triggers_backfill(tmp_path: Path) -> None:
    """首搜轮：10 条召回被预算杀 7 件、排除词杀 1 件，只剩 2 件 → 退回补搜。

    池子是按相关性召回的 top-k，不是按「预算内的相关性」——库里预算内的货可能排在
    k 名开外，带 price_usd_max 补搜捞得回来。旧闸对这形态完全不触发（手表 badcase 的
    同型未爆洞）。"""

    with thread_scope("t-hard-cull", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        guard = GuardState()

        ctx = await check_refine_backfill(
            {
                "picker_attempted": True,
                "picks_count": 2,
                "excluded_count": 1,
                "over_budget_count": 7,
                "total_candidates": 10,
                "_guard": guard,
            }
        )

        assert machine.phase is Phase.SEARCHING
        assert BACKFILL_LATCH in guard.notified_transitions  # 只触发一次
        assert ctx is not None and ctx["total_candidates"] == 0
        assert ctx["reset_fresh_candidates"] is True


async def test_sparse_pool_without_cull_no_backfill(tmp_path: Path) -> None:
    """池子小但淘汰为 0（库存稀疏）不触发——与污染分支「小但干净不触发」同一纪律；
    诊断缺席（None）同样不触发（失效方向中性）。"""
    with thread_scope("t-cull-sparse", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)

        await check_refine_backfill(
            {
                "picker_attempted": True,
                "picks_count": 2,
                "excluded_count": 0,
                "over_budget_count": 0,
            }
        )
        assert machine.phase is Phase.COMPARING

        await check_refine_backfill({"picker_attempted": True, "picks_count": 2})
        assert machine.phase is Phase.COMPARING


async def test_hard_cull_notice_points_to_price_filter(tmp_path: Path) -> None:
    """杀池通告必须指到实处：超预算为主 → 带 price_usd_max 重搜（召回期过滤），
    照原样重搜只会拿回同一批超预算的货。判据与闸共用 _hard_cull_backfill_due。"""
    from app.harness.hooks.progress import append_transition_notice

    with thread_scope("t-cull-notice", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)

        ctx = {
            "tool_name": "item_picker",
            "tool_result": '{"picks": []}',
            "call_picks": 0,  # 杀得最狠的形态（全灭）也必须指路，不许沉默
            "call_excluded": 1,
            "call_over_budget": 7,
            "_guard": GuardState(),
        }
        out = await append_transition_notice(ctx)
        assert out is not None
        assert "[阶段回退]" in out["tool_result"]
        assert "price_usd_max" in out["tool_result"]


async def test_pollution_backfill_fires_only_once(tmp_path: Path) -> None:
    """已补搜过一次（闩已写）后即使仍污染也不再回退——防「重搜还是脏 → 无限回退」。"""
    with thread_scope("t-once", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        guard = GuardState()
        guard.notified_transitions.add(BACKFILL_LATCH)  # 已补搜过一次

        await check_refine_backfill(
            {
                "picker_attempted": True,
                "picks_count": 2,
                "oncat_count": 1,
                "offcat_count": 9,
                "_guard": guard,
            }
        )

        assert machine.phase is Phase.COMPARING
        assert machine.phase is Phase.COMPARING  # 没有第二次回退


async def test_no_rerank_signal_no_pollution_judgement(tmp_path: Path) -> None:
    """本轮没跑相关性门（oncat=None）→ 判不了污染就不判；失效方向 = 维持现状。"""
    with thread_scope("t-no-rerank", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 2, "oncat_count": None, "offcat_count": None}
        )

        assert machine.phase is Phase.COMPARING


def test_picker_head_counts_visible_to_model_only_when_polluted() -> None:
    """oncat/offcat 在**模型可见文本**里仍放头部（截断后模型至少看得到计数）；
    常态（offcat=0）不带字段不烧 token。harness 一侧已不读这段文本（走诊断侧信道）。"""
    from app.tools.item_picker import ItemPickerOutput

    out = ItemPickerOutput(picks=[], excluded=[], over_budget=[], oncat_count=2, offcat_count=8)
    text = str(out)
    assert text.index('"offcat_count"') < text.index('"picks"')

    clean = str(
        ItemPickerOutput(picks=[], excluded=[], over_budget=[], oncat_count=10, offcat_count=0)
    )
    assert '"offcat_count"' not in clean


def test_diagnostics_channel_roundtrip_and_isolation(tmp_path) -> None:
    """侧信道契约：FIFO 配对、消费即删除、thread 间隔离、无 thread 作用域静默降级。"""
    from app.tools._diagnostics import consume_diagnostics, report_diagnostics
    from app.utils.thread_ctx import thread_scope

    # 无 thread 作用域：两个方向都是 no-op / None（单测环境不炸）
    report_diagnostics("item_picker", {"picks": 1})
    assert consume_diagnostics("item_picker") is None

    with thread_scope("t-diag-a", tmp_path):
        report_diagnostics("item_picker", {"picks": 3, "oncat_count": 2})
        report_diagnostics("item_picker", {"picks": 7})
        with thread_scope("t-diag-b", tmp_path):
            assert consume_diagnostics("item_picker") is None  # 别的 thread 看不见
        assert consume_diagnostics("item_picker") == {"picks": 3, "oncat_count": 2}  # FIFO
        assert consume_diagnostics("item_picker") == {"picks": 7}
        assert consume_diagnostics("item_picker") is None  # 消费即删除
