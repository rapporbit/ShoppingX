"""McAuley 并入四条规则的确定性单测（离线）。

三条用例就是定稿时演示的真实商品（字段有删减）：
B0179PPE4U 规则① / B07VR8MVLB 规则② / B0B1LPRDVG 规则③。
"""

from __future__ import annotations

from scripts.merge_mcauley import (
    merge_record,
    needs_brand_prefix,
    size_tokens,
    strip_html,
    variant_rule,
)


def _rec(item_id: str, title: str, **kw: object) -> dict:
    return {"item_id": item_id, "title": title, "brand": "", "description": "", "price": 1.0, **kw}


def test_rule1_self_parent_takes_everything() -> None:
    rec = _rec("B0179PPE4U", "Falcon-III Backpack (Khaki)", price=152.1)
    mc = {
        "parent_asin": "B0179PPE4U",
        "store": "Maxpedition",
        "title": "Maxpedition Falcon-III Backpack (Khaki)",
        "features": ["100% Denier", "Overall volume: 2160 cu. in. / 35L", "To be updated."],
        "description": "The Falcon-III is bigger and more comfortable than prior models.",
        "price": 171.67,
        "rating_number": 1234,
    }
    out = merge_record(rec, mc)
    assert out["mcauley_rule"] == 1
    assert out["title"] == "Maxpedition Falcon-III Backpack (Khaki)"
    assert out["brand"] == "Maxpedition"
    assert out["features"] == ["100% Denier", "Overall volume: 2160 cu. in. / 35L"]
    assert out["description"].startswith("The Falcon-III")
    assert out["price"] == 152.1  # 不取 McAuley 历史价
    assert out["parent_rating_count"] == 1234
    assert "35L" in out["attr_line"]


def test_rule2_variant_keeps_our_title_and_filters_junk() -> None:
    rec = _rec("B07VR8MVLB", "Women's Go Walk Lite-15433 Boat Shoe, Black/White, 12")
    mc = {
        "parent_asin": "B07PARENT1",
        "store": "Skechers",
        "title": "Skechers Women's Go Walk Lite-15433 Boat Shoe",
        "features": [
            "100% Textile",
            "Synthetic sole",
            "Shaft measures approximately not_applicable",
        ],
    }
    out = merge_record(rec, mc)
    assert out["mcauley_rule"] == 2
    assert out["title"] == "Skechers Women's Go Walk Lite-15433 Boat Shoe, Black/White, 12"
    assert out["features"] == ["100% Textile", "Synthetic sole"]
    assert out["attr_line"].startswith("Material:")
    assert "Specs" not in out["attr_line"]  # 变体商品不取规格


def test_rule3_size_conflict_takes_brand_only() -> None:
    rec = _rec(
        "B0B1LPRDVG", "20 Inch Carry On Luggage Airline Approved, Carry-On 20-Inch(Ivory White)"
    )
    mc = {
        "parent_asin": "B0PARENT22",
        "store": "Hanke",
        "title": "Hanke 20/24/29 Inch 3 Piece Luggage Sets PC Hardshell Suitcases (Jet Black)",
        "features": ["Nestable storage 3 Piece Set 20/24/29 Inch", "Made of 100% PC hard shell"],
        "description": "3 piece set",
    }
    out = merge_record(rec, mc)
    assert out["mcauley_rule"] == 3
    assert out["brand"] == "Hanke"
    assert out["title"].startswith("Hanke 20 Inch Carry On")
    assert "features" not in out
    assert out["description"] == ""


def test_no_prefix_when_brand_already_in_title() -> None:
    rec = _rec("B0X", "Samsonite Freeform Hardside", brand="Samsonite")
    assert merge_record(rec, None)["title"] == "Samsonite Freeform Hardside"


def test_uncovered_item_still_gets_title_materials() -> None:
    out = merge_record(_rec("B0Y", "Boys' Full Zip Jacket, Made with Lightweight Fleece"), None)
    assert out["materials"] == ["fleece"]
    assert "mcauley_rule" not in out


def test_size_tokens_and_rule_edges() -> None:
    assert size_tokens("Navy, 2-Piece Set (21/28)") == {"2", "21/28"}
    # McAuley 有尺寸、我们没有 → 判不了，按冲突处理
    assert variant_rule("A", "Zip Tote", {"parent_asin": "P", "title": "Zip Tote 14 inch"}) == 3
    # 两边都没尺寸 → 规则②
    assert variant_rule("A", "Zip Tote", {"parent_asin": "P", "title": "Pendleton Zip Tote"}) == 2


# ---------- 品牌前缀（2026-09-30 体检：前 3 个词重复 20,653 条） ----------
def test_prefix_skipped_when_brand_word_already_leads_title() -> None:
    assert not needs_brand_prefix("SAXX Underwear Co.", "SAXX Men's Underwear – Quest Boxer Briefs")
    assert not needs_brand_prefix("Minus33 Merino Wool", "Minus33 100% Merino Wool Katmai Mens")
    assert not needs_brand_prefix("S SMILEFIL", "Smilefil Tufting Roller Brush")  # 单字母 S 不算
    assert not needs_brand_prefix("Sennheiser Consumer Audio", "Sennheiser HD 300 Closed Back")


def test_prefix_skipped_for_placeholder_stores() -> None:
    for store in ("Amazon Renewed", "Generic", "Unknown", "Artist Unknown"):
        assert not needs_brand_prefix(store, "Sony RF400 Wireless Headphones (Renewed)")


def test_prefix_added_when_title_lacks_brand() -> None:
    assert needs_brand_prefix("Skechers", "Women's Go Walk Lite-15433 Boat Shoe")
    assert needs_brand_prefix("The North Face", "Men's Borealis Backpack")  # the 是停用词


def test_placeholder_store_still_fills_brand_field() -> None:
    rec = _rec("B07M6MRQMP", "Sony RF400 Wireless Home Theater Headphones (Renewed)")
    mc = {"parent_asin": "B07M6MRQMP", "store": "Amazon Renewed", "title": "x"}
    out = merge_record(rec, mc)
    assert out["title"] == rec["title"]
    assert out["brand"] == "Amazon Renewed"


# ---------- 网页代码 ----------
def test_strip_html() -> None:
    assert (
        strip_html("Size 34 (Jacket34/Pants31);<br> Size 36")
        == "Size 34 (Jacket34/Pants31); Size 36"
    )
    assert strip_html("Functional &Amp; Practical") == "Functional & Practical"
    assert strip_html("<p>Soft &quot;cotton&quot;</p>") == 'Soft "cotton"'


def test_merge_cleans_html_in_features_and_description() -> None:
    rec = _rec("B0Z", "Wool Hat")
    mc = {
        "parent_asin": "B0Z",
        "store": "Acme",
        "title": "Acme Wool Hat",
        "features": ["<br>", "Soft &amp; warm<br>"],
        "description": "<p>Classic hat</p>",
    }
    out = merge_record(rec, mc)
    assert out["features"] == ["Soft & warm"]
    assert out["description"] == "Classic hat"
