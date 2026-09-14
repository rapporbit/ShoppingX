"""会话级短期偏好状态 P_t（``app.memory.session_state``，lite 结构）的确定性测试。

覆盖：
- merge_pt_lite：并入去重 / 极性翻转 / 撤回按词核验（宁紧）/ 换域清表保预算 / 预算放开 / 收货国。
- middle_context 读写与容错（读坏、旧格式 → 空开局；无 TTL）。
- 偏好面板：三个词表 → 行 / 按行 id 删。
"""

import json
from datetime import UTC, datetime, timedelta

from app.memory.session_state import (
    SessionPrefState,
    constraint_rows,
    drop_constraint,
    merge_pt_lite,
    pt_from_state,
    pt_into_state,
)


def test_first_turn_fills_three_buckets_and_budget() -> None:
    pt = merge_pt_lite(
        SessionPrefState(),
        exclude=["塑料", "Plastic", "塑料"],
        avoid=["花哨"],
        prefer=["帆布", "小众"],
        category="旅行收纳",
        domains=["bags"],
        budget_usd=42.0,
    )
    assert pt.exclude_terms == ["塑料", "plastic"]  # 小写、去重、保序
    assert pt.avoid_terms == ["花哨"] and pt.prefer_terms == ["帆布", "小众"]
    assert pt.budget_usd == 42.0 and pt.category == "旅行收纳" and pt.domains == ["bags"]
    assert pt.dislike_terms() == ["塑料", "plastic"]
    assert pt.soft_dislike_terms() == ["花哨"] and pt.like_terms() == ["帆布", "小众"]


def test_followup_turn_inherits_and_appends() -> None:
    prev = merge_pt_lite(SessionPrefState(), exclude=["plastic"], budget_usd=42.0, domains=["bags"])
    pt = merge_pt_lite(prev, prefer=["waterproof"], domains=["bags"])  # 本轮没提预算 → 保持
    assert pt.exclude_terms == ["plastic"] and pt.prefer_terms == ["waterproof"]
    assert pt.budget_usd == 42.0


def test_retract_only_when_word_in_utterance() -> None:
    """撤回宁紧：词在本轮原话里出现才删；幻觉词不删（含中→英扩词：撤「塑料」连带 plastic）。"""
    prev = merge_pt_lite(SessionPrefState(), exclude=["塑料", "plastic", "nylon"])
    pt = merge_pt_lite(prev, retract_terms=["塑料", "nylon"], user_utterance="算了，塑料的也行")
    assert pt.exclude_terms == ["nylon"]  # nylon 原话没提 → 留下（宁紧）


def test_polarity_flip_moves_word_between_buckets() -> None:
    prev = merge_pt_lite(SessionPrefState(), exclude=["blue"])
    pt = merge_pt_lite(prev, prefer=["blue"])  # 「还是要蓝色」→ 最新表达为准
    assert pt.exclude_terms == [] and pt.prefer_terms == ["blue"]


def test_domain_switch_clears_terms_keeps_budget() -> None:
    prev = merge_pt_lite(
        SessionPrefState(),
        exclude=["plastic"],
        prefer=["canvas"],
        domains=["bags"],
        budget_usd=80.0,
    )
    pt = merge_pt_lite(prev, category="沙发", domains=["furniture"], exclude=["leather"])
    assert pt.exclude_terms == ["leather"] and pt.prefer_terms == []
    assert pt.budget_usd == 80.0 and pt.domains == ["furniture"]


def test_same_domain_different_wording_does_not_clear() -> None:
    prev = merge_pt_lite(
        SessionPrefState(), exclude=["plastic"], category="旅行包", domains=["bags"]
    )
    pt = merge_pt_lite(prev, category="travel backpack", domains=["bags", "apparel"])
    assert pt.exclude_terms == ["plastic"]  # 品类措辞漂移不算换域


def test_clear_budget_and_dest_country() -> None:
    prev = merge_pt_lite(SessionPrefState(), budget_usd=42.0, dest_country="jp")
    pt = merge_pt_lite(prev, clear_budget=True)
    assert pt.budget_usd is None and pt.dest_country == "JP"  # 收货国本轮没提 → 保持


# ---------- middle_context roundtrip 与容错 ----------
def test_state_roundtrip() -> None:
    s = merge_pt_lite(SessionPrefState(), exclude=["plastic"], budget_usd=42.0, category="旅行收纳")
    ctx: dict = {}
    pt_into_state(ctx, s)
    loaded = pt_from_state(json.loads(json.dumps(ctx)))  # 走一遍 JSON：模拟 session.json 落盘读回
    assert loaded.exclude_terms == ["plastic"] and loaded.budget_usd == 42.0
    assert loaded.category == "旅行收纳" and loaded.updated_at


def test_load_missing_or_corrupt_returns_empty() -> None:
    assert pt_from_state({}).is_empty()
    assert pt_from_state({"pt": {"exclude_terms": "不是列表"}}).is_empty()


def test_load_old_id_format_degrades_to_empty() -> None:
    """旧格式（带 constraints / epoch / next_id）→ extra=forbid 触发 ValidationError → 空开局。"""
    old = {"category": "x", "constraints": [{"id": "c1", "content": "不要塑料"}], "epoch": 1}
    assert pt_from_state({"pt": old}).is_empty()


def test_stale_state_is_not_expired() -> None:
    """P_t 没有 TTL：同一 thread 隔多久回来都接着上次（随 session.json 同生共死）。"""
    stale = SessionPrefState(
        prefer_terms=["蓝色"], updated_at=(datetime.now(UTC) - timedelta(hours=999)).isoformat()
    )
    assert not pt_from_state({"pt": stale.model_dump()}).is_empty()


# ---------- render / 面板 ----------
def test_render_empty_placeholder_and_buckets() -> None:
    assert "尚无累积约束" in SessionPrefState().render()
    text = SessionPrefState(exclude_terms=["plastic"], avoid_terms=["花哨"], budget_usd=42).render()
    assert "硬排除" in text and "plastic" in text and "软避讳" in text and "$42" in text


def test_constraint_rows_and_drop() -> None:
    pt = SessionPrefState(exclude_terms=["plastic"], prefer_terms=["canvas"])
    rows = constraint_rows(pt)
    assert [(r["id"], r["polarity"], r["blocking"]) for r in rows] == [
        ("exclude:plastic", "dislike", True),
        ("prefer:canvas", "like", False),
    ]
    assert drop_constraint(pt, "exclude:plastic") is True and pt.exclude_terms == []
    assert drop_constraint(pt, "exclude:plastic") is False  # 幂等
    assert drop_constraint(pt, "nope:x") is False
