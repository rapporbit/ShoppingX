"""清洗 + 检索文本塑形的确定性单测（embedding 前置，离线）。

覆盖 `app/utils/clean.py:clean_text` 与 `app/recall/text.py` 的 tail_category / clip_sentence /
分词 / 稀疏项，对齐执行计划 `docs/plans/recall-hybrid-embedding-plan.md` §2/§3。
"""

from __future__ import annotations

from app.recall.text import clip_sentence, tail_category
from app.utils.clean import clean_text


# ---------- clean_text ----------
def test_clean_text_strips_html_entities_urls() -> None:
    out = clean_text("<p>Tough &amp; Tools</p> see https://x.com/a now")
    assert out == "Tough & Tools see now"


def test_clean_text_removes_emoji_and_decor() -> None:
    assert clean_text("🌟 High Quality ▪ Box ✦") == "High Quality Box"


def test_clean_text_folds_width_and_cjk_punct() -> None:
    # 全角字母数字 → 半角；中文标点 → 半角；重复标点折叠。
    assert clean_text("ＡＢＣ１２３") == "ABC123"
    assert clean_text("很好，质量不错。。。买它（推荐）") == "很好,质量不错.买它(推荐)"


def test_clean_text_collapses_exotic_whitespace() -> None:
    assert clean_text("a\xa0b　c\n\nd") == "a b c d"


def test_clean_text_multilingual_safe() -> None:
    # 泰文 / 变音符不被误删（只动符号，不动文字）。
    assert clean_text("เสื้อยืด ผู้ชาย") == "เสื้อยืด ผู้ชาย"
    assert clean_text("Café Crème Brûlée").startswith("Café Crème Brûlée")
    # 词内连字符保留（只折叠 3+ 连排）。
    assert clean_text("PK–2 grade") == "PK–2 grade"


def test_clean_text_idempotent_and_strip_html_flag() -> None:
    once = clean_text("<b>Hi</b>  &amp; bye 🌟")
    assert clean_text(once) == once  # 幂等
    # strip_html=False 不解标签（标题/品牌用），但仍归一空白/emoji。
    assert clean_text("<b>Hi</b> 🌟", strip_html=False) == "<b>Hi</b>"


def test_clean_text_clips_to_500() -> None:
    assert len(clean_text("x " * 600)) <= 500


# ---------- tail_category ----------
def test_tail_category_takes_last_three() -> None:
    full = "Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Running > Road Running"
    assert tail_category(full) == "Athletic > Running > Road Running"
    # 不足 3 层原样；空返空。
    assert tail_category("Bags > Totes") == "Bags > Totes"
    assert tail_category("") == ""


# ---------- clip_sentence（防半句） ----------
def test_clip_sentence_prefers_sentence_then_word_boundary() -> None:
    short = "a short clean description."
    assert clip_sentence(short, limit=300) == short  # 不超限原样
    # 超限且 [200,300] 内有句末标点 → 截到句末（含标点）。
    text = ("word " * 50) + "end here. " + ("tail " * 40)  # "." 落在 ~258
    out = clip_sentence(text, limit=300)
    assert out.endswith(".") and 200 <= len(out) <= 300
    # 超限且无句末标点 → 词边界，不切断单词、无尾空格。
    nopunct = "alpha " * 100
    out2 = clip_sentence(nopunct, limit=300)
    assert len(out2) <= 300 and not out2.endswith(" ") and out2.split()[-1] == "alpha"
