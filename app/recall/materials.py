"""商品材质 / 规格抽取：给编码文本的短属性行、材质软约束用。

两个来源取并集：
- **标题**（全库都有）：只认直接点名的材质词，跳过「for X / cutting X / than X」这类
  X 是用途或比较对象的语境，以及「Resin Mold / Canvas Frame」这类 X 是配套物的搭配。
- **features / details**（McAuley 补全，约六成商品有）：只在材质语境里抽——``100% Cotton``
  百分比行、``made of / Material:`` 之后、``Rubber sole`` 这类「材质 + 部位」短语、details
  的 Material / Fabric Type 键。全文扫会把「能钻 stone」「breaks down」也当材质。

材质只做**软约束**：抽不到 = 未知，不等于「不含」。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# 长词在前，保证 faux leather / stainless steel 先于 leather / steel 命中。
_VOCAB = [
    "faux leather",
    "vegan leather",
    "pu leather",
    "genuine leather",
    "leather",
    "suede",
    "nubuck",
    "sheepskin",
    "canvas",
    "cotton",
    "linen",
    "merino wool",
    "wool",
    "cashmere",
    "silk",
    "polyester",
    "nylon",
    "spandex",
    "rayon",
    "denim",
    "fleece",
    "textile",
    "synthetic",
    "velvet",
    "felt",
    "microfiber",
    "mesh",
    "rubber",
    "latex",
    "eva",
    "memory foam",
    "foam",
    "silicone",
    "plastic",
    "abs",
    "polycarbonate",
    "polypropylene",
    "acrylic",
    "pvc",
    "resin",
    "stainless steel",
    "carbon steel",
    "steel",
    "cast iron",
    "aluminum",
    "aluminium",
    "copper",
    "brass",
    "titanium",
    "sterling silver",
    r"\d+k gold",
    "gold[- ]plated",
    "wooden",
    "wood",
    "bamboo",
    "cork",
    "jute",
    "tempered glass",
    "glass",
    "ceramic",
    "porcelain",
    "marble",
    "paper",
    "goose down",
    "duck down",
]
_MAT_RE = re.compile(r"\b(" + "|".join(_VOCAB) + r")\b", re.I)
# 同义归一。faux / vegan / PU leather 不并进 leather：用户说「不要真皮」时它们不该被排除。
_SYNONYMS = {
    "genuine leather": "leather",
    "wooden": "wood",
    "aluminium": "aluminum",
    "gold plated": "gold-plated",
    "merino wool": "wool",
}

# 材质词前 3 个词内出现这些 → X 是用途 / 适配 / 比较对象。
_NEG_PREV = re.compile(
    r"\b(for|compatible with|fits|to|cleans?|cut|cuts|cutting|drills?|than)"
    r"\s+(?:[\w/.-]+\s+){0,3}$",
    re.I,
)
# 材质词后紧跟这些 → 是护理品 / 工具 / 外观效果，不是本体材质。
_NEG_NEXT = re.compile(
    r"^[\s-]*(cleaner|cleaning|conditioner|care|glue|adhesive|polish|cutter|cutting|drill|bits?"
    r"|saws?|blades?|paint|dye|repair|shredder|holder|dispenser|detector|scraper|sander|sealer"
    r"|stain|oil|wax|punch|look|effect|pattern|print|grain|colou?r)\b",
    re.I,
)
# 只对个别词成立的配套搭配：Resin Mold 是浇树脂的模具，Canvas Frame 是装画布的框。
_NEG_PAIR = re.compile(r"^(resin\s+molds?|canvas\s+(?:floater\s+)?frames?)\b", re.I)

_JUNK = re.compile(
    r"to be updated|not_applicable|risk[- ]free|satisfaction guarantee|money[- ]back"
    r"|customer service|contact us",
    re.I,
)
_FEATURE_TRIGGER = re.compile(
    r"(^\s*\d{1,3}\s*%\s*|\b(?:made|crafted|constructed|built)\s+(?:of|from|with|out of)\b"
    r"|\bmaterials?\s*[:：]|\b(?:body|shell|upper|lining|fabric|frame)\s*[:：])",
    re.I,
)
_PART = re.compile(r"\b([a-z ]{3,25})\s+(sole|upper|lining|outsole|midsole|shell|frame)\b", re.I)
_DETAIL_KEY = re.compile(r"material|fabric type|lining", re.I)
_TRIGGER_WINDOW = 60
_SPEC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?"
    r"(?:L|liters?|mAh|W|watts?|GB|TB|V|lumens?|qt|quarts?|oz|lbs?|cu\.? ?in\.?)\b",
    re.I,
)
ATTR_LINE_MAX = 150


def _norm(word: str) -> str:
    w = re.sub(r"\s+", " ", word.lower())
    return _SYNONYMS.get(w, w)


def _add(out: list[str], word: str) -> None:
    w = _norm(word)
    if w not in out:
        out.append(w)


def title_materials(title: str) -> list[str]:
    """标题里直接点名的材质（去掉用途 / 比较 / 配套语境），按出现顺序去重。"""
    out: list[str] = []
    for m in _MAT_RE.finditer(title):
        before, after = title[: m.start()][-40:], title[m.start() :]
        if _NEG_PREV.search(before) or _NEG_NEXT.search(after[len(m.group(0)) :][:25]):
            continue
        if _NEG_PAIR.match(after):
            continue
        _add(out, m.group(0))
    return out


def clean_features(features: Iterable[str]) -> list[str]:
    """去掉占位行（To be updated / not_applicable）与售后话术行。"""
    return [f for f in features if f and not _JUNK.search(f)]


def feature_materials(features: Iterable[str], details: Mapping[str, object] | None) -> list[str]:
    """features / details 里「材质语境」中的材质词。"""
    spans = [str(v) for k, v in (details or {}).items() if _DETAIL_KEY.search(k)]
    for f in clean_features(features):
        if m := _FEATURE_TRIGGER.search(f):
            spans.append(f[m.start() : m.end() + _TRIGGER_WINDOW])
        spans.extend(p.group(0) for p in _PART.finditer(f))
    out: list[str] = []
    for s in spans:
        for m in _MAT_RE.finditer(s):
            _add(out, m.group(0))
    return out


def feature_specs(features: Iterable[str], limit: int = 4) -> list[str]:
    """features 里的容量 / 功率 / 重量类规格（``35L`` ``5000mAh`` ``16oz``）。"""
    out: list[str] = []
    for f in clean_features(features):
        for m in _SPEC_RE.findall(f):
            if m not in out:
                out.append(m)
    return out[:limit]


def attr_line(materials: list[str], specs: list[str]) -> str:
    """编码文本 / 精排用的短属性行，≤150 字符；都没有返回空串。"""
    parts = []
    if materials:
        parts.append("Material: " + ", ".join(materials[:5]))
    if specs:
        parts.append("Specs: " + ", ".join(specs))
    return "; ".join(parts)[:ATTR_LINE_MAX]
