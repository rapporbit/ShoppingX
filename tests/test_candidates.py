"""会话级候选登记表 + 模型可见紧凑序列化（确定性，无 LLM）。

覆盖：url/image_url 从模型可见形态剔除、但经 item_id 旁路登记/回填仍可取回；各工具 Output 的
``__str__`` 是不含 url 的紧凑 JSON；登记表按会话隔离、收尾可清。
"""

import json
from pathlib import Path

from app.tools._candidates import (
    compact_candidates,
    enrich,
    register,
    reset_candidates,
)
from app.tools.item_picker import ItemPickerOutput
from app.tools.item_search import ItemSearchOutput
from app.tools.price_compare import PriceCompareOutput
from app.tools.schemas import ItemCandidate
from app.tools.shipping_calc import ShippingCalcOutput
from app.utils.thread_ctx import thread_scope


def _cand(item_id: str = "A1") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id,
        platform="amazon",
        title="帆布旅行包",
        price_usd=29.9,
        url="https://www.amazon.com/dp/A1",
        image_url="https://m.media-amazon.com/I/aaa.jpg",
    )


# ---------- compact_candidates ----------
def test_compact_drops_url_and_image_keeps_rest() -> None:
    rows = compact_candidates([_cand()])
    assert "url" not in rows[0] and "image_url" not in rows[0]
    # 其余字段（含渐进填充的 price_usd）保留。
    assert rows[0]["item_id"] == "A1"
    assert rows[0]["title"] == "帆布旅行包"
    assert rows[0]["price_usd"] == 29.9


def test_compact_truncates_title_when_asked() -> None:
    """title_chars：回显场景（picker/price_compare）截短标题做 handle；不传则完整保留。"""
    c = _cand()
    c.title = "Sony Alpha a6400 Mirrorless Camera " * 5
    assert len(compact_candidates([c], title_chars=60)[0]["title"]) <= 61  # 60 + 省略号
    assert compact_candidates([c], title_chars=60)[0]["title"].endswith("…")
    assert compact_candidates([c])[0]["title"] == c.title  # 首次呈现（item_search）不截


def test_compact_drops_unfilled_stage_fields() -> None:
    """召回阶段还没填的字段（None / 空串）不喂模型——实测占返回体 27%，纯烧 token。

    「字段是 null」与「字段不在」对模型是同一个意思（这步还没做），但前者要花真金白银的 token。
    """
    row = compact_candidates([_cand()])[0]
    for absent in ("shipping_usd", "duty_usd", "landed_usd", "weight_kg", "pick_reason", "brand"):
        assert absent not in row, f"{absent} 尚未填充，不该出现在喂给模型的候选里"


def test_compact_projects_to_decision_fields_only() -> None:
    """模型可见投影 = 决策要用的字段。score / reviews_count 不喂，原价币种在 price_usd 在时不喂。

    判据是「模型拿它干什么」：下游工具只收 item_ids、自己 hydrate 全量候选，所以模型手里这份
    候选只用来**挑 id** + 讲人话。score 是召回分（候选已按它排序，序位已表达完）；reviews_count
    在本数据集恒为 0（喂了等于告诉模型「零评价」，是负向误导）。
    """
    c = _cand()
    c.score, c.reviews_count, c.rating, c.category = 0.713, 0, 4.3, "Outdoor Recreation"
    c.price, c.currency = 20.99, "USD"  # price_usd=29.9 已在 → 这两个冗余
    c.pref_matched = True  # 精挑内部判据，只给 shopping_summary 的 _compact，主 loop 不喂

    row = compact_candidates([c])[0]
    assert set(row) == {"item_id", "platform", "title", "price_usd", "rating", "category"}


def test_compact_keeps_raw_price_when_usd_missing() -> None:
    """护栏：``price_usd`` 缺失时，原价 + 币种是**唯一**的价格信息，必须留。

    否则模型对这条候选完全瞎判预算。
    """
    c = _cand()
    c.price_usd = None
    c.price, c.currency = 1000.0, "MXN"

    row = compact_candidates([c])[0]
    assert row["price"] == 1000.0
    assert row["currency"] == "MXN"
    assert "price_usd" not in row


def test_compact_drop_param_removes_extra_fields() -> None:
    """``drop`` 供调用方按上下文再砍：单平台检索时 item_search 用它去掉每条冗余的 platform。"""
    row = compact_candidates([_cand()], drop={"platform"})[0]
    assert "platform" not in row
    assert row["item_id"] == "A1"  # 别把该留的一起砍了


def test_item_search_str_keeps_platform_when_merging_all() -> None:
    """护栏：``platform="all"`` 合流时每条的 platform 必须留——模型得知道每件货来自哪个平台。"""
    c = _cand()
    single = json.loads(
        str(ItemSearchOutput(platform="amazon", candidates=[c], total_recall=1, truncated=False))
    )
    merged = json.loads(
        str(ItemSearchOutput(platform="all", candidates=[c], total_recall=1, truncated=False))
    )

    assert "platform" not in single["candidates"][0]  # 顶层已写过一次，逐条重复是纯冗余
    assert merged["candidates"][0]["platform"] == "amazon"


def test_compact_keeps_filled_stage_fields_and_zero() -> None:
    """护栏：填过的阶段字段必须留下，否则比价 / 到手价 / 精挑的结果会被这刀一起切没。

    数值 0 同理要留——0 是「值」不是「缺失」（``duty_usd=0.0`` 是「免征额内，关税为零」这个**结论**，
    不是「还没算」；跟着 None 一起丢掉会让模型以为关税没算过）。
    """
    c = _cand()
    c.shipping_usd, c.duty_usd, c.landed_usd = 5.0, 0.0, 34.9
    c.pick_reason = "轻便耐用，预算内最优"

    row = compact_candidates([c])[0]
    assert row["shipping_usd"] == 5.0
    assert row["duty_usd"] == 0.0  # 0 不是缺失，别跟着 None 一起被丢
    assert row["landed_usd"] == 34.9
    assert row["pick_reason"] == "轻便耐用，预算内最优"


# ---------- 各 Output 的 __str__ 是紧凑 JSON 且无 url ----------
def test_outputs_str_is_json_without_url() -> None:
    c = _cand()
    cases = [
        str(ItemSearchOutput(platform="amazon", candidates=[c], total_recall=1, truncated=False)),
        str(
            PriceCompareOutput(
                base_currency="USD", ranked=[c], cheapest_per_platform={}, skipped=[]
            )
        ),
        str(ShippingCalcOutput(dest_country="US", base_currency="USD", items=[c], uncosted=[])),
        str(ItemPickerOutput(picks=[c], excluded=[], over_budget=[])),
    ]
    for s in cases:
        obj = json.loads(s)  # 必须是合法 JSON（不再是 pydantic repr）
        assert "amazon.com" not in s and "media-amazon" not in s  # url/image 不喂模型
        assert "A1" in s  # item_id 仍在（下游/收尾按它引用）
        assert isinstance(obj, dict)


# ---------- 登记 / 回填 / 会话隔离 ----------
def test_register_and_enrich_roundtrip(tmp_path: Path) -> None:
    with thread_scope("t1", tmp_path):
        register([_cand("A1")])
        got = enrich("A1")
        assert got is not None
        assert got.url == "https://www.amazon.com/dp/A1"
        assert got.image_url == "https://m.media-amazon.com/I/aaa.jpg"
        assert enrich("NOPE") is None  # 未登记 → None
        reset_candidates()
        assert enrich("A1") is None  # 清理后取不到


def test_register_no_session_is_silent_noop() -> None:
    # 无 session 作用域（未进 thread_scope）：register 不抛、enrich 返回 None。
    register([_cand("Z9")])
    assert enrich("Z9") is None


def test_register_does_not_overwrite_real_url_with_empty(tmp_path: Path) -> None:
    """下游传回的候选可能已丢 url（经模型紧凑序列化），不能覆盖 item_search 登记的真值。"""
    with thread_scope("t2", tmp_path):
        register([_cand("A1")])  # 带真实 url
        stripped = _cand("A1")
        stripped.url = ""
        stripped.image_url = ""
        register([stripped])  # 空 url 不应覆盖
        got = enrich("A1")
        assert got is not None and got.url == "https://www.amazon.com/dp/A1"
        reset_candidates()


def test_sessions_are_isolated(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with thread_scope("ta", a):
        register([_cand("A1")])
    with thread_scope("tb", b):
        assert enrich("A1") is None  # 另一会话取不到
    with thread_scope("ta", a):
        reset_candidates()


async def test_item_picker_takes_registry_as_its_candidate_set(tmp_path: Path) -> None:
    """item_picker 不收 item_ids（参数已删）：候选取自会话登记表的**全集**。

    模型因此不必把 id 逐个抄一遍（纯搬运、要它实打实解码一长串 id），也没法再「自己筛一遍」——
    筛选是本工具的职责（它有权重、有排序、有封顶）。「登记表里该有什么」由 planner 保证：判
    search 的那轮它已把上一轮的旧候选清掉。
    """
    from app.tools.item_picker import item_picker

    with thread_scope("t-pick", tmp_path):
        register([_cand("A1"), _cand("A2")])
        out = await item_picker.ainvoke({})  # 一个候选参数都不传
        assert {c.item_id for c in out.picks} == {"A1", "A2"}
        reset_candidates()
