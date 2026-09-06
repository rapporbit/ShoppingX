"""「多类并列」形态验收：形态登记 / 每类各取 top N / 一类都不砍 / 文案与报告口径。

并列形态与「一套齐」共用槽位机制（登记、打标、分组精排、报告结构），差别只在最后那步选择
规则。这里断言的正是那点差别——**MCKP 不许在并列轮里砍掉某一类**，以及由此派生的报告 /
渲染 / 刷新口径（预算是每件上限，不是几类的总额）。纯确定性，无 LLM 无网络。
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.tools._bundle import (
    SLOT_MODE_BUNDLE,
    SLOT_MODE_PARALLEL,
    BundleSlot,
    combine_parallel,
    detect_slot,
    drop_pick_from_report,
    get_session_mode,
    note_slot_searched,
    refresh_report_prices,
    render_allocation,
    reset_session_bundle,
    set_session_bundle,
)
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope


@contextmanager
def _parallel_session(name: str, slots: list[BundleSlot]):
    with thread_scope(name, Path(tempfile.mkdtemp())):
        set_session_bundle(slots, mode=SLOT_MODE_PARALLEL)
        try:
            yield
        finally:
            reset_session_bundle(clear_file=True)


def _slot(name: str, *, keywords: list[str] | None = None) -> BundleSlot:
    return BundleSlot(name=name, essential=True, keywords=keywords or [])


def _c(item_id: str, title: str, price: float, slot: str = "") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, platform="amazon", title=title, price_usd=price, slot=slot, rating=4.0
    )


def _scores(*ids: str) -> dict[str, float]:
    """按传入顺序递减的 base 分（第一个最高），让「取 top N」的结果可断言。"""
    return {i: float(len(ids) - n) for n, i in enumerate(ids)}


# --------------------------------------------------------------------------
# 选择规则：每类各取 top N，一类都不砍
# --------------------------------------------------------------------------
def test_parallel_keeps_every_slot_even_when_sum_exceeds_budget() -> None:
    """这是并列形态存在的全部理由：几类价格加起来远超「预算」，也不许砍掉任何一类。

    同一组数据交给 combine_bundle 会为了凑总预算整槽放弃 optional 槽——那在「一套齐」里对，
    在「跑鞋 + 耳机」上就是把用户明说要看的一类弄丢了。
    """
    cands = [
        _c("R1", "running shoes pro", 120, "跑鞋"),
        _c("R2", "running shoes lite", 80, "跑鞋"),
        _c("H1", "noise cancelling headphones", 150, "耳机"),
        _c("H2", "budget anc headphones", 90, "耳机"),
    ]
    ids = [c.item_id for c in cands]
    with _parallel_session("t-par-1", [_slot("跑鞋"), _slot("耳机")]):
        out = combine_parallel(cands, _scores(*ids), {}, 200.0, w_cheap=0.2, w_slot_pref=0.5)
    assert out is not None
    picked = {p.slot.name for p in out.chosen}
    assert picked == {"跑鞋", "耳机"}  # 两类都在，一类都没砍
    assert out.report["skipped_optional"] == []
    assert out.report["feasible"] is True  # 并列形态没有「凑不齐这一套」这回事
    # 每类都给了多件（不是 bundle 的每槽一件）
    assert len([p for p in out.chosen if p.slot.name == "跑鞋"]) == 2


def test_parallel_takes_top_n_per_slot_by_score() -> None:
    """每类内按分排序取前 N（默认 3）：第 4 名不进，且进的是分高的那几个。"""
    cands = [_c(f"R{i}", f"running shoes {i}", 50 + i, "跑鞋") for i in range(4)]
    cands += [_c("H1", "anc headphones", 90, "耳机")]
    with _parallel_session("t-par-2", [_slot("跑鞋"), _slot("耳机")]):
        out = combine_parallel(
            cands,
            _scores("R0", "R1", "R2", "R3", "H1"),
            {},
            None,
            w_cheap=0.0,
            w_slot_pref=0.0,
        )
    assert out is not None
    shoes = [p.cand.item_id for p in out.chosen if p.slot.name == "跑鞋"]
    assert shoes == ["R0", "R1", "R2"]  # 封顶 3 件，按分取前三


def test_parallel_inactive_with_single_stocked_slot() -> None:
    """只有一类有货 → 退化普通精挑（分组展示无意义），失效方向与 bundle 一致。"""
    with _parallel_session("t-par-3", [_slot("跑鞋"), _slot("耳机")]):
        out = combine_parallel(
            [_c("R1", "running shoes", 50, "跑鞋")],
            _scores("R1"),
            {},
            None,
            w_cheap=0.0,
            w_slot_pref=0.0,
        )
    assert out is None


# --------------------------------------------------------------------------
# 形态登记：落盘 / 懒读回 / 旧文件兼容
# --------------------------------------------------------------------------
def test_mode_persists_and_lazy_reloads() -> None:
    """形态与槽表同生命周期：内存清掉后从 bundle.json 一起读回。

    只落内存的话，续聊轮（内存已清）会把并列轮当成「一套齐」重新组合——那正是砍类事故的
    发生方式，且用户看不出哪里错了，只觉得「我要的耳机怎么没了」。
    """
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-par-4", sd):
        set_session_bundle([_slot("跑鞋"), _slot("耳机")], mode=SLOT_MODE_PARALLEL)
        reset_session_bundle()  # 只清内存，文件留着
        assert get_session_mode() == SLOT_MODE_PARALLEL
        reset_session_bundle(clear_file=True)


def test_legacy_bundle_file_without_mode_reads_as_bundle() -> None:
    """老会话的 bundle.json 没有 mode 字段——那时只有「一套齐」一种形态，按 bundle 读回。"""
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-par-5", sd):
        (sd / "bundle.json").write_text(
            '{"slots": [{"name": "床品"}, {"name": "台灯"}], "declined": []}', encoding="utf-8"
        )
        assert get_session_mode() == SLOT_MODE_BUNDLE
        reset_session_bundle(clear_file=True)


def test_slot_marker_accepts_parallel_wording() -> None:
    """并列轮的 demand 写「子需求：X」，打标机制照样认——措辞跟着场景走，让模型写
    「套装槽位：跑鞋」这种别扭话，漏写的概率就高一截，而漏写 = 这批候选没盖章。"""
    assert detect_slot("子需求：跑鞋。预算 500 以内") == "跑鞋"
    assert detect_slot("套装槽位：床品，要纯棉") == "床品"


# --------------------------------------------------------------------------
# 报告与文案口径：预算是每件上限，不是几类的总额
# --------------------------------------------------------------------------
def test_parallel_report_never_flags_overshoot_on_refresh() -> None:
    """并列轮各类价格求和超过「预算」是正常的（预算是每件上限），不许报假超支警告。"""
    cands = [_c("R1", "running shoes", 120, "跑鞋"), _c("H1", "anc headphones", 150, "耳机")]
    with _parallel_session("t-par-6", [_slot("跑鞋"), _slot("耳机")]):
        out = combine_parallel(cands, _scores("R1", "H1"), {}, 200.0, w_cheap=0.0, w_slot_pref=0.0)
        assert out is not None
        refreshed = refresh_report_prices(out.report, cands)
    assert refreshed["feasible"] is True
    assert refreshed["over_usd"] == 0


def test_parallel_render_avoids_total_and_bundle_wording() -> None:
    """注入给收尾模型的分配文本不能出现「一套」「合计」——喂了它就会把跑鞋和耳机加总。"""
    cands = [_c("R1", "running shoes", 120, "跑鞋"), _c("H1", "anc headphones", 150, "耳机")]
    with _parallel_session("t-par-7", [_slot("跑鞋"), _slot("耳机"), _slot("水壶")]):
        note_slot_searched("水壶")  # 搜了但没货 → 如实列出，不静默消失
        out = combine_parallel(cands, _scores("R1", "H1"), {}, 500.0, w_cheap=0.0, w_slot_pref=0.0)
        assert out is not None
        text = render_allocation(out.report)
    assert "一套" not in text and "合计" not in text
    assert "每件" in text  # 预算口径写明白
    assert "水壶" in text  # 缺货的那类如实交代


# --------------------------------------------------------------------------
# planner 收口 + picker 端到端
# --------------------------------------------------------------------------
def test_planner_validator_normalizes_parallel_mode() -> None:
    """两道机制收口：非法形态落 bundle（保持既有行为）；parallel 下 essential 强制 true。

    并列需求里「可选」这个概念不存在——用户点名的每一类都得给交代。留着 False 不会砍类
    （combine_parallel 压根不看它），但报告会把那一类讲成「已放弃的可选项」，等于对用户撒谎。
    """
    from app.tools.planner import PlanOutput

    slots = [
        BundleSlot(name="跑鞋", essential=True),
        BundleSlot(name="耳机", essential=False),
    ]
    plan = PlanOutput(bundle_slots=slots, slot_mode=SLOT_MODE_PARALLEL)
    assert [s.essential for s in plan.bundle_slots] == [True, True]
    assert PlanOutput(bundle_slots=slots, slot_mode="whatever").slot_mode == SLOT_MODE_BUNDLE
    # bundle 形态不动 essential——「一套齐」靠它决定预算紧时砍谁。
    mixed = PlanOutput(
        bundle_slots=[BundleSlot(name="床品"), BundleSlot(name="台灯", essential=False)],
        slot_mode=SLOT_MODE_BUNDLE,
    )
    assert [s.essential for s in mixed.bundle_slots] == [True, False]


async def test_item_picker_parallel_mode_end_to_end() -> None:
    """走真实工具入口：会话登记成 parallel → 每类各给几件、一类不砍、卡片带形态标记。"""
    from app.tools.item_picker import item_picker

    cands = [
        _c("R1", "running shoes pro cushioned", 120, "跑鞋"),
        _c("R2", "running shoes lite mesh", 80, "跑鞋"),
        _c("H1", "noise cancelling headphones over ear", 150, "耳机"),
        _c("H2", "budget anc headphones wireless", 90, "耳机"),
    ]
    with _parallel_session("t-par-8", [_slot("跑鞋"), _slot("耳机")]):
        out = await item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "budget_usd": 200.0}
        )
    assert out.bundle is not None and out.bundle["mode"] == SLOT_MODE_PARALLEL
    slots_of_picks = [c.pick_reason[1 : c.pick_reason.index("】")] for c in out.picks]
    assert set(slots_of_picks) == {"跑鞋", "耳机"}  # 两类都在
    assert len(out.picks) == 4  # 每类两件全给（不是每槽一件）


async def test_parallel_drops_same_slot_near_duplicates() -> None:
    """同一类里的颜色/包装变体只留一件——并列形态一类给好几件，才会撞上这个问题。

    被摘掉的那件同步从分配报告里剔除，但该类还有别的件时**不能**报成「这一类没找到货」。
    """
    from app.tools.item_picker import item_picker

    cands = [
        _c("R1", "running shoes pro cushioned black", 80, "跑鞋"),
        _c("R2", "running shoes pro cushioned blue", 80, "跑鞋"),  # 同价 + 标题几乎全同
        _c("H1", "noise cancelling headphones over ear", 90, "耳机"),
    ]
    with _parallel_session("t-par-9", [_slot("跑鞋"), _slot("耳机")]):
        out = await item_picker.ainvoke({"candidates": [c.model_dump() for c in cands]})
    shoes = [c for c in out.picks if c.pick_reason.startswith("【跑鞋】")]
    assert len(shoes) == 1  # 变体合并
    assert out.bundle is not None
    assert "跑鞋" not in out.bundle["missing_essential"]  # 这一类还有货，不许报缺


def test_drop_pick_keeps_slot_when_other_items_remain() -> None:
    """收尾摘掉某件（slot off-intent）后，该类还剩件就不改报缺货——bundle 每槽只有一件，
    摘掉即空，行为与原来一致；并列形态一类多件，无条件报缺就是假警报。"""
    cands = [
        _c("R1", "running shoes a", 80, "跑鞋"),
        _c("R2", "running shoes b", 70, "跑鞋"),
        _c("H1", "anc headphones", 90, "耳机"),
    ]
    with _parallel_session("t-par-10", [_slot("跑鞋"), _slot("耳机")]):
        out = combine_parallel(
            cands, _scores("R1", "R2", "H1"), {}, None, w_cheap=0.0, w_slot_pref=0.0
        )
        assert out is not None
        drop_pick_from_report("R1")
        assert "跑鞋" not in out.report["missing_essential"]
        drop_pick_from_report("R2")  # 这一类最后一件也没了 → 如实报缺
        assert "跑鞋" in out.report["missing_essential"]


# --------------------------------------------------------------------------
# 派发侧兜底：planner 没拆槽时，按 demand 标记把槽登记回来
# --------------------------------------------------------------------------
def test_dispatch_marker_registers_slot_when_planner_missed_it() -> None:
    """并列形态最脆的一环是 planner 判不判得出拆槽（实测会漏）。漏了就一个章都盖不上，
    精挑退化成全池单一 query 排序、某一类屠版。模型在派发时已明写「子需求：X」——那是确定
    的事实，机制据此把槽登记回来，不必回头指望 planner 那一跳。"""
    from app.tools._bundle import ensure_dispatch_slot, get_session_bundle

    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-par-11", sd):
        assert get_session_bundle() == []  # planner 没拆槽
        s1 = ensure_dispatch_slot("跑鞋")
        s2 = ensure_dispatch_slot("降噪耳机")
        assert s1 and s2 and s1 != s2
        assert [s.name for s in get_session_bundle()] == ["跑鞋", "降噪耳机"]
        assert get_session_mode() == SLOT_MODE_PARALLEL
        # 已有槽表时原样走 register_slot：同名解析回既有 id，不重复建。
        assert ensure_dispatch_slot("跑鞋") == s1
        reset_session_bundle(clear_file=True)


def test_dispatch_fallback_ignores_hallucinated_id_refs() -> None:
    """模型幻觉出的 s9 不代表用户要买一个叫「s9」的东西——纯 id 形状一律不建槽。"""
    from app.tools._bundle import ensure_dispatch_slot, get_session_bundle

    with thread_scope("t-par-12", Path(tempfile.mkdtemp())):
        assert ensure_dispatch_slot("s9") == ""
        assert get_session_bundle() == []
        reset_session_bundle(clear_file=True)
