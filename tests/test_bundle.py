"""「一套齐」套装机制验收：槽位状态 / demand 打标 / MCKP 组合优选 / picker bundle 模式。

组合优选是纯确定性算法（无 LLM、无网络），直接断言选择结果；picker bundle 模式在会话作用域内
注入槽位 + 打标候选走真实工具入口。
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.tools._bundle import (
    BundleSlot,
    combine_bundle,
    detect_slot,
    get_session_bundle,
    note_slot_searched,
    reconcile_slots_from_reply,
    register_slot,
    reset_session_bundle,
    set_session_bundle,
    slot_scope,
)
from app.tools._bundle import current_slot as get_current_slot
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope


@contextmanager
def _bundle_session(name: str, slots: list[BundleSlot]):
    """开一个带套装槽位的会话作用域（组合优选按 session_dir 读槽位定义）。"""
    with thread_scope(name, Path(tempfile.mkdtemp())):
        set_session_bundle(slots)
        try:
            yield
        finally:
            reset_session_bundle(clear_file=True)


def _slot(name: str, *, essential: bool = True, keywords: list[str] | None = None) -> BundleSlot:
    return BundleSlot(name=name, essential=essential, keywords=keywords or [])


def _c(item_id: str, title: str, price: float, slot: str = "") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, platform="amazon", title=title, price_usd=price, slot=slot, rating=4.0
    )


# --------------------------------------------------------------------------
# 组合优选（MCKP）
# --------------------------------------------------------------------------
def test_combine_picks_one_per_slot_within_budget() -> None:
    cands = [
        _c("B1", "bedding set", 50, "床品"),
        _c("B2", "cheap bedding", 30, "床品"),
        _c("L1", "desk lamp", 20, "台灯"),
        _c("L2", "cheap lamp", 10, "台灯"),
    ]
    base = {"B1": 2.0, "B2": 1.0, "L1": 2.0, "L2": 1.0}
    with _bundle_session("t-bundle-1", [_slot("床品"), _slot("台灯")]):
        out = combine_bundle(cands, base, {}, 100.0, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None
    assert [(p.slot.name, p.cand.item_id) for p in out.chosen] == [("床品", "B1"), ("台灯", "L1")]
    assert out.report["feasible"] is True
    assert out.report["total_usd"] == 70.0


def test_combine_downgrades_slot_to_fit_budget() -> None:
    """预算不够顶配组合时，自动在某个槽降级（B2+L1），而不是超预算或整槽放弃。"""
    cands = [
        _c("B1", "bedding set", 50, "床品"),
        _c("B2", "cheap bedding", 30, "床品"),
        _c("L1", "desk lamp", 20, "台灯"),
        _c("L2", "cheap lamp", 10, "台灯"),
    ]
    base = {"B1": 2.0, "B2": 1.0, "L1": 2.0, "L2": 1.0}
    with _bundle_session("t-bundle-2", [_slot("床品"), _slot("台灯")]):
        out = combine_bundle(cands, base, {}, 55.0, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None
    # 可行组合里分数最高的是 B2(1)+L1(2)=3（B1+L2=3 同分但 $60 超预算；B2+L2=2 分更低）。
    assert {p.cand.item_id for p in out.chosen} == {"B2", "L1"}
    assert out.report["feasible"] is True


def test_combine_skips_optional_slot_when_tight() -> None:
    cands = [
        _c("B1", "bedding set", 50, "床品"),
        _c("F1", "cooling fan", 30, "风扇"),
    ]
    base = {"B1": 2.0, "F1": 1.0}
    with _bundle_session("t-bundle-3", [_slot("床品"), _slot("风扇", essential=False)]):
        out = combine_bundle(cands, base, {}, 60.0, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None
    assert [p.cand.item_id for p in out.chosen] == ["B1"]  # 必备保住，可选砍掉
    assert out.report["skipped_optional"] == ["风扇"]
    assert out.report["feasible"] is True


def test_combine_infeasible_reports_overshoot_honestly() -> None:
    """最省组合也超预算 → 如实标 feasible=False + 超支额，仍给出最省组合而非静默超支。"""
    cands = [_c("B1", "bedding", 50, "床品"), _c("L1", "lamp", 10, "台灯")]
    base = {"B1": 1.0, "L1": 1.0}
    with _bundle_session("t-bundle-4", [_slot("床品"), _slot("台灯")]):
        out = combine_bundle(cands, base, {}, 20.0, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None
    assert out.report["feasible"] is False
    assert out.report["over_usd"] == 40.0
    assert {p.cand.item_id for p in out.chosen} == {"B1", "L1"}


def test_refresh_report_prices_uses_landed_and_flags_overshoot() -> None:
    """组合按货价定稿、收尾已补到手价 → 刷新分配表口径；刷新后超预算要如实改标。"""
    from app.tools._bundle import refresh_report_prices

    cands = [_c("B1", "bedding", 40, "床品"), _c("L1", "lamp", 15, "台灯")]
    base = {"B1": 1.0, "L1": 1.0}
    with _bundle_session("t-bundle-r", [_slot("床品"), _slot("台灯")]):
        out = combine_bundle(cands, base, {}, 60.0, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None and out.report["total_usd"] == 55.0
    # 收尾时两件都补上了到手价（货价 + 运费关税），合计 70 > 预算 60。
    hydrated = [
        _c("B1", "bedding", 40, "床品").model_copy(update={"landed_usd": 50.0}),
        _c("L1", "lamp", 15, "台灯").model_copy(update={"landed_usd": 20.0}),
    ]
    refreshed = refresh_report_prices(out.report, hydrated)
    assert refreshed["total_usd"] == 70.0
    assert refreshed["feasible"] is False
    assert refreshed["over_usd"] == 10.0
    assert out.report["total_usd"] == 55.0  # 原 report 不被就地改动


def test_combine_inactive_without_bundle_session() -> None:
    """无会话槽位（普通单品类轮）→ None，picker 走普通精挑。"""
    out = combine_bundle([_c("A", "x", 10)], {"A": 1.0}, {}, None, w_cheap=0.4, w_slot_pref=1.0)
    assert out is None


def test_combine_keyword_fallback_assigns_untagged() -> None:
    """没盖章的候选（主 loop 直搜 / 旧轮读回）按槽 keywords 匹标题兜底归槽。"""
    cands = [
        _c("B1", "bedding set queen", 40, "床品"),
        _c("F1", "portable cooling fan usb", 25),  # 未打标，靠 keywords=fan 归进风扇槽
    ]
    base = {"B1": 1.0, "F1": 1.0}
    slots = [_slot("床品"), _slot("风扇", keywords=["fan"])]
    with _bundle_session("t-bundle-5", slots):
        out = combine_bundle(cands, base, {}, None, w_cheap=0.0, w_slot_pref=1.0)
    assert out is not None
    assert ("风扇", "F1") in [(p.slot.name, p.cand.item_id) for p in out.chosen]


# --------------------------------------------------------------------------
# 槽位状态与 demand 打标
# --------------------------------------------------------------------------
def test_detect_slot_marker_beats_name_and_unknown_names_pass() -> None:
    with _bundle_session("t-bundle-6", [_slot("床品"), _slot("台灯")]):
        # 「套装槽位：X」标记优先——原文透传（新槽名交给 item_search 的 register_slot 统一解析）。
        assert detect_slot("套装槽位：床垫。搜 memory foam mattress，预算内") == "床垫"
        lamp_id = next(s.id for s in get_session_bundle() if s.name == "台灯")
        assert detect_slot("为台灯这个槽找一盏护眼灯") == lamp_id  # 已登记槽名匹配 → 返回 id
        assert detect_slot("在 amazon 搜无线鼠标") is None  # 非套装 demand
    assert detect_slot("为台灯这个槽找一盏护眼灯") is None  # 无会话作用域 → 不打标


def test_slot_scope_and_register_and_searched() -> None:
    assert get_current_slot() == ""
    with slot_scope("床品"):
        assert get_current_slot() == "床品"
    assert get_current_slot() == ""
    with thread_scope("t-bundle-7", Path(tempfile.mkdtemp())):
        register_slot("床垫")  # 套装未激活 → no-op，不给非套装轮开「模型自造槽」的口子
        assert get_session_bundle() == []
        set_session_bundle([_slot("床品"), _slot("台灯")])
        register_slot("床垫")  # 激活后补登（用户确认新增），essential 兜底为 True
        names = [s.name for s in get_session_bundle()]
        assert names == ["床品", "台灯", "床垫"]
        note_slot_searched("床品")
        reset_session_bundle(clear_file=True)
        assert get_session_bundle() == []


def test_register_slot_resolves_drift_to_stable_id() -> None:
    """线上 badcase 75aa84：planner 登记「旅行背包」，检索时模型飘成「旅行背包/双肩包」——
    精确判重会补登一个同品类重复槽（两个 essential 包槽 → 组合优选各配一件）。
    id 化后：任意引用（id / 精确名 / 漂移名）都解析到同一个稳定 id，不再有新槽诞生。"""
    with _bundle_session("t-bundle-drift", [_slot("旅行背包"), _slot("旅行收纳袋")]):
        bag_id = next(s.id for s in get_session_bundle() if s.name == "旅行背包")
        assert register_slot("旅行背包/双肩包") == bag_id  # 新名包含已有槽名 → 归并
        assert register_slot("背包") == bag_id  # 反向包含也归并
        assert register_slot("旅行背包") == bag_id  # 精确名
        assert register_slot(bag_id) == bag_id  # id 直引
        names = [s.name for s in get_session_bundle()]
        assert names == ["旅行背包", "旅行收纳袋"]  # 没有新槽诞生
        assert register_slot("s99") == ""  # 野 id 不创建叫「s99」的幻觉槽
        new_id = register_slot("旅行洗漱包")  # 真新槽照常补登、机制发号
        assert new_id and get_session_bundle()[-1].name == "旅行洗漱包"
        assert get_session_bundle()[-1].id == new_id


def test_bundle_persists_and_lazy_reloads() -> None:
    """内存清掉（run_agent 收尾）后按 session_dir 从 bundle.json 懒读回——续聊轮追问要用。"""
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-bundle-8", sd):
        set_session_bundle([_slot("床品"), _slot("台灯")])
        reset_session_bundle()  # 只清内存，文件留着
        assert [s.name for s in get_session_bundle()] == ["床品", "台灯"]
        reset_session_bundle(clear_file=True)


# --------------------------------------------------------------------------
# planner 收口 + picker bundle 模式
# --------------------------------------------------------------------------
def test_planner_validator_cleans_bundle_slots() -> None:
    from app.tools.planner import PlanOutput

    # 单槽不构成套装 → 清空（下游不进组合模式）。
    assert PlanOutput(bundle_slots=[_slot("床品")]).bundle_slots == []
    # 去重 + 封顶 6。
    many = [_slot(f"槽{i}") for i in range(8)] + [_slot("槽0")]
    assert len(PlanOutput(bundle_slots=many).bundle_slots) == 6


async def test_item_picker_bundle_mode_end_to_end() -> None:
    from app.tools.item_picker import item_picker

    cands = [
        _c("B1", "bedding set queen soft", 40, "床品"),
        _c("B2", "budget bedding twin", 25, "床品"),
        _c("L1", "led desk lamp dimmable", 20, "台灯"),
        _c("L2", "cheap clip lamp", 10, "台灯"),
        _c("F1", "portable cooling fan", 30, "风扇"),
    ]
    slots = [_slot("床品"), _slot("台灯"), _slot("风扇", essential=False)]
    with _bundle_session("t-bundle-9", slots):
        out = await item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "budget_usd": 70.0}
        )
    assert out.bundle is not None
    assert out.bundle["feasible"] is True
    assert out.bundle["total_usd"] <= 70.0
    picked_slots = [c.pick_reason[1 : c.pick_reason.index("】")] for c in out.picks]
    assert len(picked_slots) == len(set(picked_slots))  # 一槽最多一件
    assert {"床品", "台灯"} <= set(picked_slots)  # 必备槽全覆盖
    for c in out.picks:
        assert c.pick_reason.startswith("【")  # 槽名前缀进理由（用户看得出这是哪一格）
    assert "bundle" in str(out)  # 组合报告随工具结果回显给模型


async def test_picker_writes_slot_back_even_for_untagged_picks() -> None:
    """badcase 75aa84 的「其他」组根因：主循环补搜没盖章的候选靠 keywords 兜底归了槽，
    但定稿只把槽名写进 pick_reason 文案、没回写 slot 字段 → 收尾卡片 slot="" → 前端落
    「其他」。断言 picks 的 slot 字段 = 归槽结果，盖章缺失的也不例外。"""
    from app.tools.item_picker import item_picker

    cands = [
        # 补搜的真背包：无盖章（slot=""），标题命中「旅行背包」槽 keywords。
        _c("NEW1", "Large Travel Backpack 17 inch Durable", 30, ""),
        _c("OLD1", "sling crossbody bag", 18, "旅行背包"),  # 上一轮盖过章的次优款
        _c("CUBE", "packing cubes set", 17, "旅行收纳袋"),
    ]
    slots = [
        _slot("旅行背包", keywords=["travel backpack"]),
        _slot("旅行收纳袋", keywords=["packing cubes"], essential=False),
    ]
    with _bundle_session("t-bundle-10", slots):
        table = {s.name: s.id for s in get_session_bundle()}
        out = await item_picker.ainvoke(
            {"candidates": [c.model_dump() for c in cands], "budget_usd": 70.0}
        )
    assert out.bundle is not None
    by_id = {c.item_id: c for c in out.picks}
    # 定稿的每一件都带**槽 id**章，谁也不掉「其他」；名字章（OLD1）也被归一成 id。
    assert all(c.slot in set(table.values()) for c in out.picks)
    bag = by_id.get("NEW1") or by_id.get("OLD1")  # 包槽必有一件入选
    assert bag is not None and bag.slot == table["旅行背包"]
    if "NEW1" in by_id:  # 兜底归槽的那件必须带上归槽结果
        assert by_id["NEW1"].slot == table["旅行背包"]


# --------------------------------------------------------------------------
# id 化审计回归：续聊轮懒读回 / declined 落盘
# --------------------------------------------------------------------------
def test_slot_resolution_survives_continuation_turn() -> None:
    """审计 bug①：run_agent 收尾只清内存，续聊轮第一跳（item_search 直搜）就可能撞上
    register_slot——解析必须内置懒读回，否则漂移归并/补登在续聊轮静默失效。"""
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-bundle-cont", sd):
        set_session_bundle([_slot("旅行背包"), _slot("旅行收纳袋")])
        bag_id = get_session_bundle()[0].id
        reset_session_bundle()  # 只清内存（run_agent 收尾形态），bundle.json 留着
        assert register_slot("旅行背包/双肩包") == bag_id  # 漂移归并在续聊轮第一跳就生效
        reset_session_bundle(clear_file=True)


def test_declined_slot_persists_and_blocks_drifted_resurrection() -> None:
    """审计 bug③：declined 随 bundle.json 落盘；续聊轮清内存后，被删槽的精确名与
    漂移变体（「台灯」→「护眼台灯」）都复活不了。"""
    sd = Path(tempfile.mkdtemp())
    with thread_scope("t-bundle-declined", sd):
        set_session_bundle([_slot("书包"), _slot("文具"), _slot("台灯")])
        removed = reconcile_slots_from_reply("就要书包和文具", offered=["书包", "文具", "台灯"])
        assert removed == ["台灯"]
        reset_session_bundle()  # 续聊轮：内存清空，全靠盘上那份
        assert register_slot("护眼台灯") == ""  # 漂移变体不复活
        assert register_slot("台灯") == ""  # 精确名同样不复活
        assert [s.name for s in get_session_bundle()] == ["书包", "文具"]
        reset_session_bundle(clear_file=True)
