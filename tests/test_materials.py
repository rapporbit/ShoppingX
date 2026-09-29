"""材质 / 规格抽取的确定性单测（离线）。

用例取自 2026-09-29 抽样里的真实标题与 features：前半证明「点名的材质能抽到」，
后半证明原型里出过的误判（用途 / 比较 / 配套 / 全文扫）不再出现。
"""

from __future__ import annotations

import pytest

from app.recall.materials import (
    attr_line,
    clean_features,
    feature_materials,
    feature_specs,
    title_materials,
)


# ---------- 标题：该抽到的 ----------
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Compatible with Oculus Quest 2 Accessories, Silicone Face Cover", ["silicone"]),
        ("Crown 480 Sheets Bulk Pack Black Tissue Paper Gift Wrap", ["paper"]),
        ("Lusofie Tooth Fairy Box 3D Carved Wooden Cute Tooth Box", ["wood"]),
        ("Genuine Leather Women's Wallets, Slim Bifold", ["leather"]),
        ("Women's Faux Leather Moto Jacket", ["faux leather"]),
        ("18K Gold Freshwater Cultured White Pearl Necklace", ["18k gold"]),
        ("304 Stainless Steel U Bolt Saddle Fastener", ["stainless steel"]),
    ],
)
def test_title_materials_hits(title: str, expected: list[str]) -> None:
    assert title_materials(title) == expected


# ---------- 标题：原型里的误判 ----------
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # 材质是用途：for 后隔 2 个词
        ("Myard PBP66 Post Base Plate for 6x6 Inches Wood Post", []),
        # 材质是加工对象：Cutting 后隔 1 个词；Titanium 是钻头本体涂层，保留
        ("3-8mm Titanium HSS Drill & Saw Bit Set Cutting Carpenter Wood Metal", ["titanium"]),
        # 比较句
        ("Black Scrunchies for Women, Premium Satin Softer than Silk", []),
        # 配套搭配：Resin Mold 浇树脂用；品牌 BSRESIN 不算词
        (
            "BSRESIN Resin Mold for Makeup Brush Holder, "
            "Skull Decor Silicone Molds for Epoxy Resin",
            ["silicone"],
        ),
        ("Akfado Canvas Floater Frame for 36x36 Oil Paintings", []),
        # 护理品
        ("Leather Cleaner and Conditioner for Car Seats", []),
    ],
)
def test_title_materials_skips_non_body(title: str, expected: list[str]) -> None:
    assert title_materials(title) == expected


# ---------- features / details ----------
def test_feature_materials_from_context_only() -> None:
    feats = ["100% Cotton", "Rubber sole", "To be updated.", "Machine Wash"]
    assert feature_materials(feats, None) == ["cotton", "rubber"]


def test_feature_materials_ignores_free_text() -> None:
    # v1 全文扫出过 stone/marble/glass（钻头能钻的）和 down（breaks down）
    feats = [
        "Diamond tip drill bits for stone, marble and glass tile",
        "Reduces lead and breaks down chlorine",
    ]
    assert feature_materials(feats, None) == []


def test_feature_materials_reads_detail_material_keys() -> None:
    details = {"Material": "Stainless Steel", "Color": "Glass Blue"}
    assert feature_materials([], details) == ["stainless steel"]


def test_clean_features_drops_placeholders_and_boilerplate() -> None:
    feats = [
        "Shaft measures approximately not_applicable from arch",
        "100% Risk-Free Satisfaction Guarantee",
        "Lightweight cushioning",
    ]
    assert clean_features(feats) == ["Lightweight cushioning"]


def test_feature_specs_and_attr_line() -> None:
    specs = feature_specs(["Overall volume: 2160 cu. in. / 35L", "5000mAh battery"])
    assert "35L" in specs and "5000mAh" in specs
    assert attr_line(["leather", "rubber"], ["35L"]) == "Material: leather, rubber; Specs: 35L"
    assert attr_line([], []) == ""
    assert len(attr_line(["cotton"] * 50, ["1L"] * 50)) <= 150
