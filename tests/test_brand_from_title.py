"""标题首词补品牌的词表与匹配规则单测（离线，不连 Qdrant）。

用例取自 2026-10-03 抽样里的真实标题：AUSDOM 是要补上的那类，Police / Sling Bag 是不能补的那类。
"""

from __future__ import annotations

from scripts.backfill_brand_from_title import brand_from_title, build_vocab, tokens

# 10 万条标题里：costume / bag / home / decor 是常见词（阈值 = 10 万 × 0.05% = 50 次）。
_WORD_FREQ = {"police": 80, "costume": 900, "bag": 5000, "home": 3000, "decor": 2500, "joe": 12}
_N_TITLES = 100_000


def _vocab(**brand_counts: int) -> dict[str, str]:
    return build_vocab(
        {k.replace("_", " "): v for k, v in brand_counts.items()}, _WORD_FREQ, _N_TITLES
    )


def test_tokens_strip_trailing_punctuation() -> None:
    assert tokens("Bucilla, Toy Soldiers Co.") == ["bucilla", "toy", "soldiers", "co"]
    assert tokens("Kate & Milo DIY") == ["kate", "milo", "diy"]
    assert tokens("") == []


def test_known_brand_at_title_head_is_filled() -> None:
    vocab = _vocab(AUSDOM=12)
    title = "AUSDOM Bluetooth Noise Cancelling Headphones: E7 Wireless Over Ear ANC"
    assert brand_from_title(title, vocab) == "AUSDOM"
    assert brand_from_title("ausdom headphones", vocab) == "AUSDOM"  # 不分大小写，写回词表里的写法


def test_brand_must_be_at_head_and_whole_word() -> None:
    vocab = _vocab(AUSDOM=12)
    assert brand_from_title("Headphones compatible with AUSDOM E7", vocab) == ""
    assert brand_from_title("AUSDOMX Headphones", vocab) == ""


def test_longest_match_wins() -> None:
    vocab = _vocab(Under_Armour=40, Under=5)
    assert brand_from_title("UNDER ARMOUR Replacement Rise Lid Black", vocab) == "Under Armour"


def test_spelling_takes_most_frequent_casing() -> None:
    vocab = build_vocab({"LEGO": 300, "Lego": 4}, _WORD_FREQ, _N_TITLES)
    assert brand_from_title("Lego DUPLO Town", vocab) == "LEGO"


def test_rare_brand_and_placeholder_are_dropped() -> None:
    vocab = _vocab(Xiaokeis=2, Generic=5000, Unknown=900)
    assert vocab == {}
    assert brand_from_title("Generic Luggage Cover", vocab) == ""


def test_common_word_brand_is_dropped() -> None:
    # Police 是真品牌（手表），但「Police Costume for Boys」是通用标题：单词本身常见就不进词表。
    vocab = _vocab(Police=30, Home_Decor=25)
    assert brand_from_title("Police Costume for Boys Cop Uniform", vocab) == ""
    assert brand_from_title("Home Decor Wall Art", vocab) == ""


def test_multiword_brand_with_one_rare_word_is_kept() -> None:
    # Genuine Joe：joe 不是常见词，整体保留。
    vocab = _vocab(Genuine_Joe=9)
    assert brand_from_title('Genuine Joe Window, Vehicle & Wall Brush 10"', vocab) == "Genuine Joe"


def test_no_brand_in_title() -> None:
    vocab = _vocab(AUSDOM=12)
    assert brand_from_title("Sling Bag Crossbody Sling Backpack for Women&Men", vocab) == ""
    assert brand_from_title("", vocab) == ""
