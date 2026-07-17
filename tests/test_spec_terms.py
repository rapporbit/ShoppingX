"""数值规格专道（16寸背包 badcase，2026-07-15）：解析 / 三态裁决 / picker 路由。

badcase：query 要 16 寸笔记本背包，最终清单里混进 14 寸——数值规格在字面路（"16 inch" 匹不中
"16-inch"）与语义路（embedding 对数字失明，14 寸对 "16 inch" 的 cosine 接近满分）上双重失明，
14 寸不但不被罚、还领着硬约束语义加分上浮。测试全部照真实标题形态写（连字符 / 引号 / 大小写 /
up-to 前缀 / 多维尺寸），不写理想化输入。
"""

from __future__ import annotations

from app.tools.schemas import ItemCandidate
from app.utils.terms import parse_spec_term, spec_verdict

# --------------------------------------------------------------------------
# parse_spec_term：原子词 → (值, 归一单位)
# --------------------------------------------------------------------------


def test_parse_spec_term_variants() -> None:
    assert parse_spec_term("16 inch") == (16.0, "inch")
    assert parse_spec_term("16-inch") == (16.0, "inch")
    assert parse_spec_term("16in") == (16.0, "inch")
    assert parse_spec_term('16"') == (16.0, "inch")
    assert parse_spec_term("16寸") == (16.0, "inch")  # 电商语境「寸」= 英寸
    assert parse_spec_term("16英寸") == (16.0, "inch")
    assert parse_spec_term("15.6 inches") == (15.6, "inch")
    assert parse_spec_term("40l") == (40.0, "l")
    assert parse_spec_term("40升") == (40.0, "l")
    assert parse_spec_term("30cm") == (30.0, "cm")


def test_parse_spec_term_non_spec_words_pass_through() -> None:
    """普通词不解析——它们照走字面 term_hits + 语义通路。"""
    assert parse_spec_term("canvas") is None
    assert parse_spec_term("16") is None  # 裸数字没有单位，判不了是什么规格
    assert parse_spec_term("leather 16") is None
    assert parse_spec_term("") is None


# --------------------------------------------------------------------------
# spec_verdict：match / conflict / unknown 三态
# --------------------------------------------------------------------------


def test_spec_verdict_matches_real_title_forms() -> None:
    """真 16 寸的各种标题写法都要拿到 match——badcase 里它们一个加分都拿不到。"""
    for title in [
        "business laptop backpack 16-inch anti theft",
        "16 inch laptop backpack with usb charging port",
        'slim laptop bag 16" computer case',
        "16in travel backpack water resistant",
    ]:
        assert spec_verdict(16.0, "inch", title) == "match", title


def test_spec_verdict_tolerance_covers_screen_size_families() -> None:
    """15.6 寸位的包装 16 寸机器是常态（差 2.5%，容差内）——别把同档位判成冲突。"""
    assert spec_verdict(16.0, "inch", "laptop backpack fits 15.6 inch notebook") == "match"


def test_spec_verdict_up_to_prefix_is_upper_bound() -> None:
    """「fits up to 17.3 inch」对 16 寸要求是兼容（装得下），不是冲突也不是 unknown。"""
    assert spec_verdict(16.0, "inch", "travel backpack fits up to 17.3 inch laptops") == "match"
    assert spec_verdict(16.0, "inch", "holds up to 17 inch laptop") == "match"
    # 上界比要求还小 → 明确装不下，是冲突。
    assert spec_verdict(16.0, "inch", "sleeve fits up to 14 inch laptops only") == "conflict"


def test_spec_verdict_smaller_size_is_conflict() -> None:
    """badcase 本体：标题明晃晃标着 14 寸，对 16 寸要求就是冲突证据，必须减分。"""
    assert spec_verdict(16.0, "inch", "slim laptop backpack 14 inch water resistant") == "conflict"
    assert spec_verdict(16.0, "inch", "14-inch laptop bag for men") == "conflict"


def test_spec_verdict_unknown_cases_do_not_judge() -> None:
    """失效方向安全：判不了的一律 unknown（不奖不罚），漏挡优于误杀。"""
    # 标题没标尺寸。
    assert spec_verdict(16.0, "inch", "travel laptop backpack business anti theft") == "unknown"
    # 更大的标称尺寸：17 寸包装 16 寸机器不是冲突（只是没明说兼容）。
    assert spec_verdict(16.0, "inch", "17 inch laptop backpack") == "unknown"
    # 多维尺寸（18 高 12 低）不算「全部更小」，不冲突。
    assert spec_verdict(16.0, "inch", "duffel bag 18 x 12 x 6 inch") == "unknown"
    # 单位不同不比：40L 是容量，不是屏幕尺寸。
    assert spec_verdict(16.0, "inch", "40l hiking backpack") == "unknown"


def test_spec_verdict_in_unit_false_positives_guarded() -> None:
    """`in` 单位的两类假命中：「2 in 1」（后跟数字）与介词（"3.0 in the side pocket"）。"""
    assert spec_verdict(16.0, "inch", "2 in 1 convertible backpack tote") == "unknown"
    assert spec_verdict(16.0, "inch", "backpack with usb 3.0 in the side pocket") == "unknown"


# --------------------------------------------------------------------------
# picker 路由：must/prefer/exclude 里的数值词走 spec 专道
# --------------------------------------------------------------------------


def _bag(item_id: str, title: str) -> ItemCandidate:
    """同价同评分——隔离掉价格/评分因子，排序差异只能来自规格计分。"""
    return ItemCandidate(item_id=item_id, platform="a", title=title, landed_usd=40, rating=4.5)


_POOL = [
    _bag("B14", "Slim Laptop Backpack 14 Inch Water Resistant"),
    _bag("B16", "Business Laptop Backpack 16-inch Anti Theft"),
    _bag("BNA", "Travel Laptop Backpack with USB Charging Port"),
]


async def test_picker_must_have_spec_ranks_16_over_14() -> None:
    """badcase 复现：must_have=['16 inch'] 时，16 寸置顶、14 寸沉底到无标称的后面。"""
    from app.tools.item_picker import item_picker

    out = await item_picker.ainvoke(
        {"candidates": [c.model_dump() for c in _POOL], "must_have": ["16 inch"]}
    )

    ids = [c.item_id for c in out.picks]
    assert ids[0] == "B16"  # 容差内命中，拿硬约束加分
    assert ids[-1] == "B14"  # 标称冲突减分，沉到「没标尺寸」的后面
    assert ids == ["B16", "BNA", "B14"]
    # 冲突只沉底不淘汰（容解析噪声），三件都还在。
    assert not out.excluded
    # must_have_hits 走 spec 判定：只有 B16 算命中（BNA 是 unknown，不冒充命中）。
    assert out.must_have_hits == 1


async def test_picker_spec_terms_dedupe_zh_en() -> None:
    """「16寸」+「16 inch」中英双给是同一条规格，去重后不重复计分（同分同序）。"""
    from app.tools.item_picker import item_picker

    payload = {"candidates": [c.model_dump() for c in _POOL]}
    single = await item_picker.ainvoke({**payload, "must_have": ["16 inch"]})
    double = await item_picker.ainvoke({**payload, "must_have": ["16寸", "16 inch"]})

    assert [c.item_id for c in single.picks] == [c.item_id for c in double.picks]
    assert single.must_have_hits == double.must_have_hits == 1


async def test_picker_exclude_spec_kills_named_size() -> None:
    """「不要14寸」：exclude 桶里的规格按数值判，"14 Inch" 写法差异挡不住淘汰。"""
    from app.tools.item_picker import item_picker

    out = await item_picker.ainvoke(
        {"candidates": [c.model_dump() for c in _POOL], "exclude_keywords": ["14寸"]}
    )

    assert out.excluded == ["B14"]
    assert {c.item_id for c in out.picks} == {"B16", "BNA"}


async def test_picker_prefer_spec_soft_conflict() -> None:
    """prefer 档的规格：命中加分、冲突按软避讳减分（不淘汰）。"""
    from app.tools.item_picker import item_picker

    out = await item_picker.ainvoke(
        {"candidates": [c.model_dump() for c in _POOL], "prefer_keywords": ["16 inch"]}
    )

    ids = [c.item_id for c in out.picks]
    assert ids[0] == "B16"
    assert ids[-1] == "B14"
    assert not out.excluded
