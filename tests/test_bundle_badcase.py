"""线上 badcase 4c0ac682（「新生入学一套，预算 1500」）取材的回归用例。

四个用户可感知的缺陷各一组测试，数据形态照抄线上真实会话：
① 组成确认只增不删 → 用户删掉的槽被说成「没找到、建议再搜」（reconcile_slots_from_reply）；
② 水杯槽召回全是「water bottle stickers」蹭词垃圾，picker 硬填贴纸顶水杯——三层治理：
   槽内 rerank 排序加分（默认启用）+ 逐出门机制（**默认关**，绝对阈值被真实数据标定证伪：
   贴纸 own 0.30~0.60 vs 真笔袋 0.055，任何阈值要么放垃圾要么杀真品）+ summary 的 slot
   off-intent LLM 兜底（drop_pick_from_report 同步报缺）；
③ 总价混加裸价与到手价却声称「含税到手价」（bare_price 口径保险丝 + cost_item 补算）；
④ 同批并行补搜被逃生门按到达顺序放行一半（按槽授权 + 批次原子化，见 test_harness 扩展）。
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.tools._bundle import (
    BundleSlot,
    combine_bundle,
    get_session_bundle,
    note_slot_searched,
    reconcile_slots_from_reply,
    refresh_report_prices,
    register_slot,
    render_allocation,
    reset_session_bundle,
    set_session_bundle,
)
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope


@contextmanager
def _session(name: str, slots: list[BundleSlot]):
    with thread_scope(name, Path(tempfile.mkdtemp())):
        set_session_bundle(slots)
        try:
            yield
        finally:
            reset_session_bundle(clear_file=True)


def _slot(name: str, *, essential: bool = True, kw: list[str] | None = None) -> BundleSlot:
    return BundleSlot(name=name, essential=essential, keywords=kw or [])


def _c(item_id: str, title: str, price: float, slot: str = "") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id, platform="amazon", title=title, price_usd=price, slot=slot, rating=4.6
    )


# 线上会话的槽位定稿（planner 拆解原样）。
def _freshman_slots() -> list[BundleSlot]:
    return [
        _slot("书包", kw=["backpack", "school bag"]),
        _slot("笔记本电脑", kw=["laptop", "notebook computer"]),
        _slot("文具", essential=False, kw=["stationery", "pen", "notebook"]),
        _slot("水杯", essential=False, kw=["water bottle", "flask"]),
        _slot("台灯", essential=False, kw=["lamp", "desk lamp"]),
    ]


# --------------------------------------------------------------------------
# ① 组成确认的「删」通路
# --------------------------------------------------------------------------
def test_reconcile_removes_unpicked_slots_badcase_verbatim() -> None:
    """用户答「书包 + 文具 + 水杯 + 生活用品」→ 笔记本电脑 / 台灯从槽表删除且不再复活。"""
    with _session("t-rec-1", _freshman_slots()):
        removed = reconcile_slots_from_reply("书包 + 文具 + 水杯 + 生活用品")
        assert sorted(removed) == ["台灯", "笔记本电脑"]
        names = [s.name for s in get_session_bundle()]
        assert names == ["书包", "文具", "水杯"]
        register_slot("台灯")  # demand 里再飘出「台灯」也不复活（用户明确不要过）
        assert "台灯" not in [s.name for s in get_session_bundle()]
        register_slot("生活用品")  # 用户新增的槽照常可登记
        assert "生活用品" in [s.name for s in get_session_bundle()]


def test_reconcile_ignores_single_slot_followup() -> None:
    """「水杯换大点的」只点名 1 个槽——不是枚举式确认，一个槽都不许删。"""
    with _session("t-rec-2", _freshman_slots()):
        assert reconcile_slots_from_reply("水杯换大点的") == []
        assert len(get_session_bundle()) == 5


def test_reconcile_only_removes_offered_slots() -> None:
    """带 options 的确认：只删问卷上出现过的槽，没上问卷的不算被拒。"""
    with _session("t-rec-3", _freshman_slots()):
        removed = reconcile_slots_from_reply(
            "书包 和 文具", offered=["书包/双肩包", "文具（笔等）", "台灯/护眼灯"]
        )
        assert removed == ["台灯"]  # 笔记本电脑 / 水杯没上问卷，不删
        assert len(get_session_bundle()) == 4


# --------------------------------------------------------------------------
# ② 槽位相关性：贴纸不许顶水杯。下面三个 test_relevance_floor_* / test_gate_* 验证的是
#    **逐出门机制本身**（显式传 floor=0.2）——机制保留、线上默认关（PICK_SLOT_RERANK_FLOOR=0，
#    标定见文件头）；默认形态走 test_slot_relevance_ranks_* 的排序信号 + summary 兜底。
# --------------------------------------------------------------------------
def _sticker_pool() -> list[ItemCandidate]:
    """水杯槽的线上真实召回形态：全是标题蹭「water bottle」的贴纸，零真水杯。"""
    return [
        _c("BAG1", "Bookbag Large Capacity Lightweight School Backpack", 19.98, "书包"),
        _c("ST1", "50 Pack Coffee Stickers Water Bottles Laptop Hydroflasks", 5.95, "水杯"),
        _c("ST2", "Teens Hero Stickers Waterproof Decal Water Bottle Cup", 5.59, "水杯"),
    ]


def test_relevance_floor_evicts_slot_squatters_and_reports_missing() -> None:
    cands = _sticker_pool()
    base = {c.item_id: 1.0 for c in cands}
    # cross-encoder 实测形态：真品接近 1、蹭词垃圾接近 0（定稿实验 0.97 vs 0.006）。
    relevance = {"BAG1": 0.95, "ST1": 0.01, "ST2": 0.02}
    slots = [
        _slot("书包", kw=["backpack"]),
        _slot("水杯", essential=False, kw=["water bottle"]),
        _slot("文具", essential=False, kw=["pen"]),
    ]
    with _session("t-gate-1", slots):
        note_slot_searched("书包")
        note_slot_searched("水杯")
        cands.append(_c("PEN1", "Pencil Case Stationery Pen Bag", 4.99, "文具"))
        base["PEN1"] = 1.0
        relevance["PEN1"] = 0.9
        out = combine_bundle(
            cands,
            base,
            {},
            210.0,
            w_cheap=0.0,
            w_slot_pref=1.0,
            slot_relevance=relevance,
            relevance_floor=0.2,
        )
    assert out is not None
    picked_slots = [p.slot.name for p in out.chosen]
    assert "水杯" not in picked_slots  # 贴纸被逐出，槽宁缺毋滥
    assert out.report["missing_optional"] == ["水杯"]  # 且如实报缺，不是静默消失
    assert "ST1" not in {p.cand.item_id for p in out.chosen}


def test_relevance_floor_blocks_keyword_fallback_reentry() -> None:
    """未盖章的贴纸（标题真含 water bottle）走 keywords 兜底归槽——同样要被门拦。"""
    stray = _c("ST9", "Cute Stickers for Water Bottle Laptop", 5.0)  # 无盖章
    real = _c("CUP1", "Stainless Steel Water Bottle 750ml", 15.0, "水杯")
    with _session("t-gate-2", [_slot("水杯", kw=["water bottle"]), _slot("书包", kw=["backpack"])]):
        bagc = _c("BAG2", "College Backpack", 25.0, "书包")
        out = combine_bundle(
            [stray, real, bagc],
            {"ST9": 5.0, "CUP1": 1.0, "BAG2": 1.0},
            {},
            100.0,
            w_cheap=0.0,
            w_slot_pref=1.0,
            slot_relevance={"ST9": 0.01, "CUP1": 0.9, "BAG2": 0.9},
            relevance_floor=0.2,
        )
    assert out is not None
    cup = next(p for p in out.chosen if p.slot.name == "水杯")
    assert cup.cand.item_id == "CUP1"  # 真水杯当选；高 base 分的贴纸进不了槽


def test_gate_skipped_for_slot_without_keywords() -> None:
    """keywords 空的槽（用户确认新增）建不出干净 query——不执法，失效方向=维持现状。"""
    daily = _c("D1", "Mesh Shower Caddy Basket for Dorm", 6.99, "生活用品")
    bag = _c("BAG3", "School Backpack", 20.0, "书包")
    with _session("t-gate-3", [_slot("生活用品", kw=[]), _slot("书包", kw=["backpack"])]):
        out = combine_bundle(
            [daily, bag],
            {"D1": 1.0, "BAG3": 1.0},
            {},
            100.0,
            w_cheap=0.0,
            w_slot_pref=1.0,
            slot_relevance={"D1": 0.01, "BAG3": 0.9},
            relevance_floor=0.2,  # D1 低分但无 query
        )
    assert out is not None
    assert {p.slot.name for p in out.chosen} == {"生活用品", "书包"}  # D1 不被误逐


# --------------------------------------------------------------------------
# ③ 到手价口径：混加裸价要标出来，补算要能算
# --------------------------------------------------------------------------
def test_refresh_report_flags_bare_price_mix() -> None:
    """badcase 数字原样：两件有 landed、两件只有裸价 → bare_price=2 且 render 有口径警示。"""
    picks = [
        _c("BAG1", "School Backpack", 19.98, "书包"),
        _c("ST1", "Stickers", 5.95, "水杯"),
        _c("PEN1", "Pencil Case", 4.99, "文具"),
        _c("D1", "Shower Caddy", 6.99, "生活用品"),
    ]
    picks[0].landed_usd = 29.38
    picks[1].landed_usd = 13.95
    report = {
        "budget_usd": 210.0,
        "total_usd": 0.0,
        "feasible": True,
        "over_usd": 0,
        "rows": [
            {
                "slot": c.slot,
                "essential": True,
                "item_id": c.item_id,
                "title": c.title,
                "price_usd": c.price_usd,
            }
            for c in picks
        ],
        "skipped_optional": [],
        "missing_essential": [],
        "missing_optional": [],
        "not_included": [],
        "unslotted": 0,
        "price_unknown": 0,
        "alternatives": {},
    }
    out = refresh_report_prices(report, picks)
    assert out["bare_price"] == 2
    assert out["total_usd"] == 55.31  # 29.38 + 13.95 + 4.99 + 6.99
    text = render_allocation(out)
    assert "未含运费关税" in text and "2 件" in text


def test_cost_item_backfills_landed_deterministically() -> None:
    """summary 收尾补算用的单件口径：有 price_usd 就能把 shipping/duty/landed 填全。"""
    from app.tools.shipping_calc import cost_item

    item = _c("PEN1", "Pencil Case Pen Bag", 4.99, "文具")
    assert item.landed_usd is None
    assert cost_item(item, "CN") is True
    assert item.landed_usd is not None and item.landed_usd >= item.price_usd
    assert item.shipping_usd is not None and item.duty_usd is not None
    bad = ItemCandidate(item_id="X", platform="amazon", title="no price")
    assert cost_item(bad, "CN") is False  # 缺裸价算不了，如实返回 False


# --------------------------------------------------------------------------
# ④ postfork 闸：套装按槽授权（每槽一次），批内到达顺序不再决定谁被拦
# --------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_slot_backfill_grant_passes_once_per_slot(monkeypatch) -> None:
    from types import SimpleNamespace

    import app.harness.hooks.tool_gates as tg
    from app.harness.middleware import HarnessMiddleware
    from app.harness.state import GuardState

    monkeypatch.setattr(tg, "get_fork_budget", lambda: SimpleNamespace(parallel_calls=1))
    monkeypatch.setattr(tg, "candidate_count", lambda: 58)  # 池非空（badcase 形态）
    guard = GuardState()
    mw = HarnessMiddleware()
    mw.register("pre_tool_call", "search_authority_gate", tg.check_search_authority, priority=30)

    async def _call(args: dict) -> bool:
        out = await mw.run(
            "pre_tool_call", {"tool_name": "item_search", "tool_args": args, "_guard": guard}
        )
        return not out.get("_rejected")

    grant_slots = [_slot("书包", kw=["backpack"]), _slot("水杯", kw=["water bottle"])]
    with _session("t-grant-1", grant_slots):
        # 同批 4 个并行补搜（badcase 形态：书包/水杯/生活用品/文具）——已登记槽全放行，
        # 未登记槽名与无 slot 的直搜照拦，且不再受到达顺序影响。
        assert await _call({"slot": "书包", "query": "college backpack"}) is True
        assert await _call({"slot": "水杯", "query": "water bottle"}) is True
        assert await _call({"slot": "生活用品", "query": "caddy"}) is False  # 未登记槽不授权
        assert await _call({"query": "just better stuff"}) is False  # 无 slot 的「找更好」照拦
        assert await _call({"slot": "水杯", "query": "flask again"}) is False  # 每槽仅 1 次


@pytest.mark.asyncio
async def test_slot_backfill_grant_resolves_drifted_refs(monkeypatch) -> None:
    """id 化审计 bug②回归：闸与盖章共用 resolve_slot——模型传漂移槽名（「旅行背包/双肩包」）
    照样解析到已登记槽放行；额度按稳定 id 记账，同一槽换个写法不给第二次。"""
    from types import SimpleNamespace

    import app.harness.hooks.tool_gates as tg
    from app.harness.middleware import HarnessMiddleware
    from app.harness.state import GuardState

    monkeypatch.setattr(tg, "get_fork_budget", lambda: SimpleNamespace(parallel_calls=1))
    monkeypatch.setattr(tg, "candidate_count", lambda: 40)
    guard = GuardState()
    mw = HarnessMiddleware()
    mw.register("pre_tool_call", "search_authority_gate", tg.check_search_authority, priority=30)

    async def _call(args: dict) -> bool:
        out = await mw.run(
            "pre_tool_call", {"tool_name": "item_search", "tool_args": args, "_guard": guard}
        )
        return not out.get("_rejected")

    with _session("t-grant-2", [_slot("旅行背包", kw=["travel backpack"]), _slot("水杯")]):
        bag_id = next(s.id for s in get_session_bundle() if s.name == "旅行背包")
        assert await _call({"slot": "旅行背包/双肩包", "query": "big backpack"}) is True
        assert await _call({"slot": bag_id, "query": "backpack again"}) is False  # 同槽 id 不二次
        assert await _call({"slot": "背包", "query": "third try"}) is False  # 同槽另一漂移写法同样


@pytest.mark.asyncio
async def test_escape_door_is_batch_atomic(monkeypatch) -> None:
    """同一 think_step（同一条 AI 消息的并行调用）里连撞 4 次效率闸：全拦，且只记 1 次连拒——
    不再出现「前 2 个被拒攒计数、后 2 个触发逃生放行」的到达顺序竞态。"""
    import app.harness.hooks.tool_gates as tg
    from app.harness.middleware import HarnessMiddleware
    from app.harness.state import GuardState

    monkeypatch.setattr(tg, "web_search_allowed", lambda: False)
    guard = GuardState()
    guard.think_step = 7  # 同一批
    mw = HarnessMiddleware()
    mw.register("pre_tool_call", "websearch_gate", tg.check_websearch, priority=15)
    for _ in range(4):
        out = await mw.run("pre_tool_call", {"tool_name": "web_search", "_guard": guard})
        assert out.get("_rejected")  # 同批全拦，无一逃生
    assert guard.gate_reject_counts["websearch_gate:web_search"] == 1  # 一批=一次坚持


# --------------------------------------------------------------------------
# ⑤ dispatch 槽位派发的确定性摘要（内心独白不再泄漏给主 loop）
# --------------------------------------------------------------------------
def test_slot_digest_replaces_subagent_narration() -> None:
    from app.agent.dispatch_tool import _slot_digest
    from app.tools._candidates import register, reset_candidates

    with thread_scope("t-digest-1", Path(tempfile.mkdtemp())):
        register(
            [
                _c("BAG1", "Bookbag Large Capacity Lightweight School Backpack", 19.98, "书包"),
                _c("BAG2", "College Backpack With USB Port", 24.99, "书包"),
            ]
        )
        try:
            digest = _slot_digest("书包")
            assert digest is not None
            assert "书包" in digest and "2 件" in digest and "$19.98" in digest
            assert _slot_digest("水杯") is None  # 该槽零候选入库 → 保留子 Agent 原文
        finally:
            reset_candidates()


# --------------------------------------------------------------------------
# ⑥ picker 相关性打分的三条硬约束（mock reranker）
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_category_relevance_disables_on_local_fallback(monkeypatch) -> None:
    """远程精排降级（used_remote=False）→ 整门停用：token 重叠会给蹭词垃圾打高分。"""
    import app.tools.item_picker as ip
    from app.tools._candidates import register, reset_candidates

    class _Fake:
        async def score_detailed(self, query, texts):
            return [0.9] * len(texts), False  # 本地回退

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    with _session("t-rel-1", [_slot("书包", kw=["backpack"]), _slot("水杯", kw=["water bottle"])]):
        cands = [_c("BAG1", "School Backpack", 19.98, "书包")]
        register(cands)
        try:
            scores, ok, _conf = await ip._category_relevance(cands)
            assert ok is False and scores == {}
        finally:
            reset_candidates()


@pytest.mark.asyncio
async def test_category_relevance_scores_and_caches(monkeypatch) -> None:
    """打分回写登记表；补搜轮重跑对 query 未变的候选零重复调用（增量缓存）。"""
    import app.tools.item_picker as ip
    from app.tools._candidates import register, registry_snapshot, reset_candidates

    calls: list[int] = []

    class _Fake:
        async def score_detailed(self, query, texts):
            calls.append(len(texts))
            return [0.95] * len(texts), True

    monkeypatch.setattr(ip, "get_reranker", lambda: _Fake())
    with _session("t-rel-2", [_slot("书包", kw=["backpack"]), _slot("水杯", kw=["water bottle"])]):
        register([_c("BAG1", "School Backpack", 19.98, "书包")])
        try:
            pool = registry_snapshot()
            scores, ok, _conf = await ip._category_relevance(pool)
            assert ok is True and scores["BAG1"] == 0.95
            assert sum(calls) == 1
            # 第二轮（补搜后重跑 picker）：登记表里的分数直接命中缓存，不再调远程。
            scores2, ok2, _conf2 = await ip._category_relevance(registry_snapshot())
            assert ok2 is True and scores2["BAG1"] == 0.95
            assert sum(calls) == 1  # 零新增调用
        finally:
            reset_candidates()


def test_slot_relevance_ranks_real_item_above_squatters() -> None:
    """默认形态（逐出门关，w_relevance 排序加分）：真水杯在场时压过高 base 分的贴纸。

    标定结论（badcase 4c0ac682 真实标题 + 远程 reranker）：贴纸 own 0.30~0.60、真水杯 0.92、
    真笔袋 0.055——绝对阈值门不可行，排序信号可行。
    """
    sticker = _c("ST1", "Aesthetic Stickers Perfect for Water Bottle", 5.95, "水杯")
    cup = _c("CUP1", "Stainless Steel Insulated Water Bottle 750ml", 15.0, "水杯")
    bag = _c("BAG1", "College Backpack", 25.0, "书包")
    with _session("t-rank-1", [_slot("水杯", kw=["water bottle"]), _slot("书包", kw=["backpack"])]):
        out = combine_bundle(
            [sticker, cup, bag],
            {"ST1": 1.3, "CUP1": 1.0, "BAG1": 1.0},  # 贴纸 base 更高（评分好、便宜）
            {},
            100.0,
            w_cheap=0.0,
            w_slot_pref=1.0,
            slot_relevance={"ST1": 0.60, "CUP1": 0.92, "BAG1": 0.9},
            relevance_floor=0.0,  # 门关（默认形态）
            w_relevance=1.0,
        )
    assert out is not None
    cup_pick = next(p for p in out.chosen if p.slot.name == "水杯")
    assert cup_pick.cand.item_id == "CUP1"  # 1.0+0.92 > 1.3+0.60


def test_drop_pick_from_report_marks_slot_missing() -> None:
    """summary 的 slot off-intent 摘除后：分配报告同步删行、该槽改报缺货、总价重算。"""
    from app.tools._bundle import drop_pick_from_report, get_bundle_report, set_bundle_report

    with _session("t-drop-1", [_slot("水杯", essential=False, kw=["water bottle"])]):
        set_bundle_report(
            {
                "budget_usd": 210.0,
                "total_usd": 33.93,
                "feasible": True,
                "over_usd": 0,
                "rows": [
                    {
                        "slot": "水杯",
                        "essential": False,
                        "item_id": "ST1",
                        "title": "Stickers",
                        "price_usd": 13.95,
                    },
                    {
                        "slot": "书包",
                        "essential": True,
                        "item_id": "BAG1",
                        "title": "Backpack",
                        "price_usd": 19.98,
                    },
                ],
                "skipped_optional": [],
                "missing_essential": [],
                "missing_optional": [],
                "not_included": [],
                "unslotted": 0,
                "price_unknown": 0,
                "alternatives": {},
            }
        )
        drop_pick_from_report("ST1")
        report = get_bundle_report()
        assert report is not None
        assert [r["item_id"] for r in report["rows"]] == ["BAG1"]
        assert report["missing_optional"] == ["水杯"]
        assert report["total_usd"] == 19.98
        drop_pick_from_report("GONE")  # 不存在的行：静默跳过不炸
