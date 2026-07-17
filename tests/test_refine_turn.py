"""候选池跨轮存活：追问（「我只要防水的」）能拿到上一轮搜过的商品，不必重新检索一遍。

候选池原本每轮 reset，而跨轮只回喂 (role, content) 文本、工具消息全丢——第二轮的 Agent 手里既没有
候选登记表也没有 item_id 列表，物理上没法「在上一步基础上过滤」。落盘 + 开局读回后，下游工具按
item_id hydrate 得到上一轮的候选。

注意：「有候选」≠「本轮是追问轮」——用户完全可能换品类。本模块只保证候选**可用**，「这轮要不要
重搜」是 planner 的判断，不由候选池的有无来猜。
"""

from pathlib import Path

import pytest

from app.api.context import get_retrieval_mode, reset_retrieval_mode, set_retrieval_mode
from app.harness.agent_middleware import _count_candidates
from app.harness.hooks.phase_transition import check_refine_backfill, try_phase_transition
from app.harness.phase_machine import Phase, PhaseStateMachine, set_phase_machine
from app.harness.signals import candidate_count
from app.tools._candidates import (
    load_candidates,
    persist_candidates,
    register,
    registry_snapshot,
    render_prior_candidates,
    reset_candidates,
)
from app.tools.planner import PlanOutput
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


def test_candidates_survive_across_turns(tmp_path: Path) -> None:
    """收尾落盘 → 内存清空 → 下一轮读回：追问轮才拿得到上一轮的 item_id。"""
    with thread_scope("t-refine", tmp_path):
        register([_cand("A1"), _cand("A2")])
        persist_candidates(tmp_path)
        reset_candidates()  # 模拟本轮收尾清内存

        recovered = load_candidates(tmp_path)

    assert [c.item_id for c in recovered] == ["A1", "A2"]
    # url/image_url 也要活过来——收尾回填商品卡靠它们，模型侧则永远看不到（见 compact 投影）。
    assert recovered[0].url == "https://example.com/A1"
    assert "example.com" not in render_prior_candidates(recovered)


def test_load_candidates_missing_file_is_silent(tmp_path: Path) -> None:
    """首轮（无 candidates.json）读空，不报错——普通轮照常走 planner → item_search。"""
    with thread_scope("t-fresh", tmp_path):
        assert load_candidates(tmp_path) == []
        assert render_prior_candidates([]) == ""


# ---------- planner 判 retrieval：谁来决定「这轮要不要重搜」 ----------


def test_no_prior_candidates_forces_search(tmp_path: Path) -> None:
    """手上没有既有候选时，模型填什么都强制 search——首轮判 reuse 会让主 loop 去精挑一个空池子。"""
    with thread_scope("t-first", tmp_path):
        plan = PlanOutput(retrieval="reuse", tasks=["recommend"])
        if not registry_snapshot():  # planner 工具体里的那道确定性兜底
            plan.retrieval = "search"

        assert plan.retrieval == "search"


def test_retrieval_mode_survives_tool_context(tmp_path: Path) -> None:
    """判定必须能被主 loop 的 hook 读到。

    工具在独立 context 里执行，其中对 ContextVar 的 set **不回传**主 loop——planner 判了 reuse、
    阶段机却读不到，模型照样重搜一遍（实测 200 秒）。故按 session_dir 聚合，不用裸 ContextVar。
    """
    with thread_scope("t-mode", tmp_path):
        assert get_retrieval_mode() == "search"  # planner 没跑时的安全默认

        set_retrieval_mode("reuse")
        assert get_retrieval_mode() == "reuse"

        reset_retrieval_mode()
        assert get_retrieval_mode() == "search"


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


# ---------- planner 的判定驱动阶段 + 补搜降级 ----------


async def test_reuse_sends_planner_straight_to_comparing(tmp_path: Path) -> None:
    """planner 判 reuse → 阶段（遥测）直接跳 COMPARING，一次 item_search 都不调。"""
    with thread_scope("t-reuse", tmp_path):
        machine = PhaseStateMachine()
        set_phase_machine(machine)
        set_retrieval_mode("reuse")

        await try_phase_transition({"planner_output_ready": True})

        assert machine.phase is Phase.COMPARING


async def test_search_mode_goes_to_searching(tmp_path: Path) -> None:
    """换品类（search）→ 照常进 SEARCHING；不被上一轮的旧候选卡在 COMPARING。"""
    with thread_scope("t-search", tmp_path):
        machine = PhaseStateMachine()
        set_phase_machine(machine)
        set_retrieval_mode("search")

        await try_phase_transition({"planner_output_ready": True})

        assert machine.phase is Phase.SEARCHING


async def test_thin_reuse_result_triggers_backfill(tmp_path: Path) -> None:
    """复用轮精挑后只剩 1 件 → 退回 SEARCHING 补搜，并把 retrieval 改写成 augment（只触发一次）。"""
    with thread_scope("t-thin", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("reuse")

        ctx = await check_refine_backfill({"picker_attempted": True, "picks_count": 1})

        assert machine.phase is Phase.SEARCHING
        assert get_retrieval_mode() == "augment"  # 已在补搜 → 本闸不会再触发，不会无限回退
        # 「请重新检索」的指路不再走 inject（晚一轮）——由 transition_notice 缀在 picker 结果上
        assert ctx is not None and not ctx.get("inject_messages")


async def test_enough_reuse_picks_no_backfill(tmp_path: Path) -> None:
    """复用轮精挑够多（≥3 件）且 must_have 有命中 → 不补搜，省下那近百秒的重新检索。"""
    with thread_scope("t-enough", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("reuse")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 5, "must_have_hits": 2}
        )

        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "reuse"


async def test_reuse_zero_must_hits_triggers_backfill(tmp_path: Path) -> None:
    """复用轮 picks 件数正常、但 must_have 池内 0 命中 → 照样退回补搜。

    重现 bad case「三件套 → 要中式刺绣」：picker 的 must_have 是加分不淘汰，8 件素色候选
    对「必须刺绣」原样返回 8 件——件数闸（<3）对这种质量假阳性是瞎的，径直推进 CONCLUDING
    后模型想补搜已被阶段哨兵拦死，只能收尾承认失败。
    """
    with thread_scope("t-zero-hits", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("reuse")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 8, "must_have_hits": 0}
        )

        assert machine.phase is Phase.SEARCHING
        assert get_retrieval_mode() == "augment"  # 只触发一次，不会无限回退


async def test_reuse_without_must_have_no_quality_backfill(tmp_path: Path) -> None:
    """本轮没传 must_have（hits=None）→ 没有「质」可判，件数够就不补搜。"""
    with thread_scope("t-no-must", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("reuse")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 5, "must_have_hits": None}
        )

        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "reuse"


# ---------- 污染分支：首搜轮品类门吃空池子 → 补搜（手表 badcase 6718ed65） ----------


async def test_polluted_first_search_triggers_backfill(tmp_path: Path) -> None:
    """首搜轮（search，非 reuse）：品类门判 10 条里 8 条跨品类混入、相符只剩 2 → 退回补搜。

    重现手表 badcase：检索词 formal dress watch men business 召回一堆西装皮鞋，品类一致性门
    正确沉底 8 条，但旧闸只认 reuse 轮，没人补货，径直收尾出了 2 件的清单。
    """
    from app.harness.state import GuardState

    with thread_scope("t-polluted", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")
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
        assert get_retrieval_mode() == "augment"  # 只触发一次，不会无限回退
        # 污染批不再算「本轮已搜到货」：同轮 40 号钩子不得凭它把 SEARCHING 立刻推回 COMPARING。
        assert ctx is not None and ctx["total_candidates"] == 0
        assert ctx["reset_fresh_candidates"] is True
        # 初搜若走了并行 fork，postfork 棘轮闸会拦直搜——必须随回退发一次授权。
        assert guard.postfork_search_grants == 1


async def test_sparse_but_clean_pool_no_backfill(tmp_path: Path) -> None:
    """池子小但干净（oncat=2、offcat=0）是库存稀疏，不是检索词的错——重搜同样的词只会拿回
    同一池货，不触发。"""
    with thread_scope("t-sparse", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 2, "oncat_count": 2, "offcat_count": 0}
        )

        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "search"


# ---------- 硬淘汰杀池分支：预算/排除把干净池杀空 → 补搜（交接遗留洞 #1） ----------


async def test_hard_cull_first_search_triggers_backfill(tmp_path: Path) -> None:
    """首搜轮：10 条召回被预算杀 7 件、排除词杀 1 件，只剩 2 件 → 退回补搜。

    池子是按相关性召回的 top-k，不是按「预算内的相关性」——库里预算内的货可能排在
    k 名开外，带 price_usd_max 补搜捞得回来。旧闸对这形态完全不触发（手表 badcase 的
    同型未爆洞）。"""
    from app.harness.state import GuardState

    with thread_scope("t-hard-cull", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")
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
        assert get_retrieval_mode() == "augment"  # 只触发一次
        assert ctx is not None and ctx["total_candidates"] == 0
        assert ctx["reset_fresh_candidates"] is True
        assert guard.postfork_search_grants == 1


async def test_sparse_pool_without_cull_no_backfill(tmp_path: Path) -> None:
    """池子小但淘汰为 0（库存稀疏）不触发——与污染分支「小但干净不触发」同一纪律；
    诊断缺席（None）同样不触发（失效方向中性）。"""
    with thread_scope("t-cull-sparse", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")

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
        assert get_retrieval_mode() == "search"


async def test_hard_cull_notice_points_to_price_filter(tmp_path: Path) -> None:
    """杀池通告必须指到实处：超预算为主 → 带 price_usd_max 重搜（召回期过滤），
    照原样重搜只会拿回同一批超预算的货。判据与闸共用 _hard_cull_backfill_due。"""
    from app.harness.hooks.phase_transition import append_transition_notice
    from app.harness.state import GuardState

    with thread_scope("t-cull-notice", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")

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
    """已在补搜（augment）后即使仍污染也不再回退——防「重搜还是脏 → 无限回退」。"""
    with thread_scope("t-once", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("augment")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 2, "oncat_count": 1, "offcat_count": 9}
        )

        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "augment"


async def test_no_rerank_signal_no_pollution_judgement(tmp_path: Path) -> None:
    """本轮没跑相关性门（oncat=None）→ 判不了污染就不判；失效方向 = 维持现状。"""
    with thread_scope("t-no-rerank", tmp_path):
        machine = PhaseStateMachine(initial=Phase.COMPARING)
        set_phase_machine(machine)
        set_retrieval_mode("search")

        await check_refine_backfill(
            {"picker_attempted": True, "picks_count": 2, "oncat_count": None, "offcat_count": None}
        )

        assert machine.phase is Phase.COMPARING
        assert get_retrieval_mode() == "search"


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


# ---------- 换品类那轮：旧候选必须被清掉 ----------
#
# 「登记表里该有什么」只有 planner 判得了（就是 retrieval 判定本身）。清干净之后，登记表**自身**
# 即「本轮该精挑的全集」——下游因此不需要任何「哪些是本轮新召回的」额外账本，item_picker 直接
# 吃全集（它的 item_ids 参数已删：让模型抄 id 是纯搬运，还给它开了「再自己筛一遍」的口子）。


class _FakePlannerLLM:
    """假模型：with_structured_output(...).ainvoke(...) 回一个固定的 PlanOutput。"""

    def __init__(self, plan: PlanOutput) -> None:
        self._plan = plan

    def with_structured_output(self, _schema: object, **kwargs: object) -> "_FakePlannerLLM":
        self.structured_kwargs = kwargs
        return self

    async def ainvoke(self, _messages: object, config: object = None) -> PlanOutput:
        return self._plan


async def _run_planner(monkeypatch: pytest.MonkeyPatch, retrieval: str) -> None:
    import app.tools.planner as mod

    plan = PlanOutput(category="沙发", tasks=["recommend"], retrieval=retrieval)  # type: ignore[arg-type]
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakePlannerLLM(plan))
    await mod.planner.ainvoke({"intent": "换个方向，想看真皮沙发"})


async def test_search_turn_drops_prior_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """planner 判 search（换品类）→ 上一轮读回的旧候选清空。

    否则「上一轮的鞋」会混进「这一轮的沙发」：item_picker 吃的是登记表全集，旧货留着就是拿别的
    品类的商品去答本轮的问题。
    """
    with thread_scope("t-search", tmp_path):
        load_candidates(tmp_path)  # 首轮无文件；显式登记两件旧候选模拟读回
        register([_cand("OLD1"), _cand("OLD2")])
        assert len(registry_snapshot()) == 2

        await _run_planner(monkeypatch, "search")

        assert registry_snapshot() == []  # 旧品类的候选不许留下
        reset_retrieval_mode()
        reset_candidates()


async def test_reuse_turn_keeps_prior_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """planner 判 reuse（只收紧条件）→ 旧候选原样留着：本轮不检索，它们就是精挑的全集。"""
    with thread_scope("t-reuse", tmp_path):
        register([_cand("A1"), _cand("A2")])

        await _run_planner(monkeypatch, "reuse")

        assert [c.item_id for c in registry_snapshot()] == ["A1", "A2"]
        reset_retrieval_mode()
        reset_candidates()
