"""匹配词归一：中文原子词必须能筛掉英文商品。

这层要挡的是一个**静默失效**——商品库是纯英文的，而 item_picker 的硬淘汰靠字符串命中，于是
「不要塑料」这条中文约束在机制层一直是空转的（planner 恰好多给了个 plastic 时才碰巧生效，
记忆那条通路连这个碰巧都没有）。失败的样子不是报错，是「用户说了不要，结果它还在清单里」。
"""

from __future__ import annotations

import pytest

from app.tools.item_picker import _hits, _searchable, item_picker
from app.tools.schemas import ItemCandidate
from app.utils.terms import normalize_terms


def test_chinese_terms_gain_english_without_losing_the_original() -> None:
    """扩展而非替换：中文原词必须留着——库里还掺着中文标题的商品（shein / lazada）。"""
    assert normalize_terms(["塑料"]) == ["塑料", "plastic", "plastics"]
    assert normalize_terms(["防水"]) == ["防水", "waterproof", "water-resistant"]
    # 英文 → 原样小写 + 反向补中文（长期库常只沉淀英文，掺中文标题的平台不能漏挡）
    assert normalize_terms(["Plastic", "NYLON"]) == ["plastic", "塑料", "nylon", "尼龙"]
    # 去重保序（中英各给一份是 planner 的常见产出，不该出现两个 plastic）
    assert normalize_terms(["塑料", "plastic"]) == ["塑料", "plastic", "plastics"]


def test_english_terms_gain_chinese_but_never_single_char() -> None:
    """英→中反向扩词只补 ≥2 字的中文——中文命中是子串语义，「丝」会撞进「螺丝刀」。"""
    assert "真皮" in normalize_terms(["genuine leather"])
    # "silk" 映射的中文是单字「丝」，不补；词本身保留
    assert normalize_terms(["silk"]) == ["silk"]


def test_unmapped_chinese_is_kept_not_dropped() -> None:
    """补不出英文的词**保留**——丢弃等于静默删掉用户的一条约束（失效方向必须安全）。"""
    assert normalize_terms(["小众"]) == ["小众"]


def test_empty_and_blank_are_skipped() -> None:
    assert normalize_terms(["", "  ", None]) == []  # type: ignore[list-item]
    assert normalize_terms(None) == []


@pytest.mark.asyncio
async def test_chinese_exclude_now_kills_english_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归本体：exclude_keywords=["塑料"]（纯中文，无英文变体）必须淘汰英文塑料商品。

    修复前这条会红——中文词匹不上 "Plastic Storage Box"，商品照样进清单。
    """
    import app.tools.item_picker as mod

    async def _no_memory(_user_id: str) -> object:
        from app.memory.assemble import MemoryBundle

        return MemoryBundle()

    monkeypatch.setattr(mod, "assemble", _no_memory)

    plastic = ItemCandidate(
        item_id="P1", platform="amazon", title="6 Set Plastic Storage Box", price_usd=9.99
    )
    canvas = ItemCandidate(
        item_id="C1", platform="amazon", title="Canvas Travel Organizer", price_usd=12.99
    )
    out = await item_picker.ainvoke(
        {
            "candidates": [c.model_dump() for c in (plastic, canvas)],
            "exclude_keywords": ["塑料"],  # 只给中文，故意不给 plastic
        }
    )
    ids = [c.item_id for c in out.picks]
    assert "P1" not in ids, "中文排除词没能淘汰英文塑料商品——归一层失效"
    assert "C1" in ids


def test_hits_still_matches_english_after_normalization() -> None:
    """归一只换词，不改 _hits 的两道防线（词边界 + 否定修饰）。"""
    text = _searchable(
        ItemCandidate(item_id="x", platform="amazon", title="Plastic-Free Bamboo Set", brand="")
    )
    (plastic,) = [t for t in normalize_terms(["塑料"]) if t == "plastic"]
    # "plastic-free" 是替代品、不是要排除的东西——否定修饰过滤仍然生效
    assert _hits(plastic, text) is False


# ---- 标题属性抽取 ----------------------------------------------------------


def test_extract_attrs_reads_real_tokens_only() -> None:
    from app.utils.terms import extract_attrs

    attrs = extract_attrs("7-Piece Waterproof Nylon Packing Cubes Travel Organizer")
    assert attrs == ["7 件套", "尼龙", "防水"]
    # 抽不到就空——绝不臆造属性（这些标签会原样进商品卡的选购理由）
    assert extract_attrs("Travel Organizer") == []
    # 件数的三种常见写法都认
    assert extract_attrs("6 Set Storage Box")[0] == "6 件套"
    assert extract_attrs("8 Pcs Cube Set")[0] == "8 件套"


def test_display_term_shows_chinese_for_english_hits() -> None:
    from app.utils.terms import display_term

    assert display_term("durable") == "耐用"  # 命中的是英文，但理由是给中文用户看的
    assert display_term("canvas") == "帆布"
    assert display_term("小众") == "小众"  # 没有对照就原样


# ---- 商品卡理由：前 N 件走 LLM，其余走确定性 --------------------------------


@pytest.mark.asyncio
async def test_llm_rewrites_only_top_n_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """前 3 件用模型写的叙事句，第 4 件起用 item_picker 的确定性理由——解码量因此与件数无关。"""
    import app.tools.shopping_summary as mod
    from tests.test_tools import _FakeLLM  # 复用现成的假模型

    picks = [
        ItemCandidate(
            item_id=f"I{i}",
            platform="amazon",
            title=f"Canvas Bag {i}",
            price_usd=10.0 + i,
            pick_reason=f"规则理由 {i}",
        )
        for i in range(4)
    ]
    payload = mod._SummaryDraft(
        summary="给你挑了 4 件。",
        reasons=[
            mod.SummaryReason(item_id="I0", reason="帆布耐造，最便宜的一件，正合你说的抗造。"),
            mod.SummaryReason(item_id="I1", reason="同样帆布，评分更高一点。"),
            mod.SummaryReason(item_id="I2", reason="容量更大，适合长途。"),
        ],
    )
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {"picks": [c.model_dump() for c in picks], "user_intent": "抗造的帆布包"},
            "id": "c1",
            "type": "tool_call",
        }
    )
    reasons = [i.reason for i in msg.artifact.items]
    assert reasons[0].startswith("帆布耐造")  # 前 3 件：模型叙事
    assert reasons[2] == "容量更大，适合长途。"
    assert reasons[3] == "规则理由 3"  # 第 4 件：item_picker 的确定性理由


@pytest.mark.asyncio
async def test_truncated_llm_reason_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型把理由写到一半就停（快档实测会发生）→ 退回确定性理由，绝不让半句话进商品卡。"""
    import app.tools.shopping_summary as mod
    from tests.test_tools import _FakeLLM

    picks = [
        ItemCandidate(
            item_id="I0", platform="amazon", title="Canvas Bag", pick_reason="帆布；评分 4.5"
        )
    ]
    payload = mod._SummaryDraft(
        summary="一件。",
        reasons=[mod.SummaryReason(item_id="I0", reason="评分4.1，标题")],  # 悬垂：半句被截
    )
    monkeypatch.setattr(mod, "get_fast_llm", lambda: _FakeLLM(structured_payload=payload))
    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "args": {"picks": [c.model_dump() for c in picks], "user_intent": "帆布包"},
            "id": "c1",
            "type": "tool_call",
        }
    )
    assert msg.artifact.items[0].reason == "帆布；评分 4.5"


def test_extract_attrs_reads_chinese_titles() -> None:
    """shein / lazada 掺中文标题：抽取侧也要认中文变体与「7件套」写法（此前英文单语，恒空）。"""
    from app.utils.terms import extract_attrs

    assert extract_attrs("防水尼龙旅行收纳袋7件套") == ["7 件套", "尼龙", "防水"]
    # 英文标题行为不变：同一件商品两种语言的标题抽出同一组标签
    assert extract_attrs("7-Piece Waterproof Nylon Packing Cubes") == ["7 件套", "尼龙", "防水"]


def test_extract_attrs_respects_negation() -> None:
    """抽取与匹配必须同一套 term_hits：'Vegan Leather' 不该给用户打「皮革」标签。"""
    from app.utils.terms import extract_attrs

    assert "皮革" not in extract_attrs("Vegan Leather Tote Bag")


def test_title_attr_tokens_chinese_title_normalizes_to_english() -> None:
    """中文标题的亲和证据记到英文 token 名下——词频不因语言分裂，也不因一词双 token 虚高。"""
    from app.utils.terms import title_attr_tokens

    assert title_attr_tokens("防水尼龙背包") == ["nylon", "waterproof"]
