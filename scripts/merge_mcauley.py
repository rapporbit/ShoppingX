"""把 McAuley 补全产物（``scripts/enrich_mcauley.py`` 的 ``enrich_<cat>.jsonl``）并入 amazon 主库。

输出一份**新** jsonl，原主库不动。四条规则（2026-09-29 定稿）：
  ① item_id == parent_asin：McAuley 记录就是这件商品 → features / description / brand 全取；
  ② 其余变体、标题尺寸件数不冲突 → 同上，features 去占位行与话术行；
  ③ 标题尺寸 / 件数数字冲突，或 McAuley 标题有尺寸而我们没有 → 只取 brand；
  ④ 标题永远用我们的（颜色尺码对得上价格和图），缺品牌才加 store 前缀。
price / rating 一律不取（McAuley 是历史值）；
评论数单列 ``parent_rating_count``（父商品跨变体合计）。
材质 = 标题抽取 ∪ features 抽取（规则③不取 features），见 ``app/recall/materials.py``。

内存：McAuley 记录按原始行（bytes）驻留，约 1.5GB；在本机跑，不在 gcjp 跑。

用法：
    uv run python scripts/merge_mcauley.py \\
        --main data/platforms/clean/by_platform/amazon_rag.jsonl \\
        --enrich-dir /path/to/out \\
        --out data/platforms/clean/by_platform/amazon_rag_mcauley.jsonl
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.recall.materials import (  # noqa: E402
    attr_line,
    clean_features,
    feature_materials,
    feature_specs,
    title_materials,
)
from app.utils.clean import fine_category_path  # noqa: E402

_SIZE_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:piece|pieces|pc|pcs|pack|set|inch|inches|in\b|\"|”|cm|mm"
    r"|l\b|liter|oz|qt|quart)",
    re.I,
)
_SIZE_SLASH = re.compile(r"\b\d{2}(?:/\d{2}){1,4}\b")


def size_tokens(title: str) -> set[str]:
    """标题里的尺寸 / 件数数字（``2-Piece`` → 2，``24-Inch`` → 24，``20/24/28`` 整串）。"""
    return {m.group(1) for m in _SIZE_NUM.finditer(title)} | set(_SIZE_SLASH.findall(title))


# McAuley store 里的占位值：brand 字段照收，但不加进标题（「Generic Luggage Cover」不像话）。
_NO_PREFIX_STORES = {"amazon renewed", "generic", "unknown", "artist unknown"}
_BRAND_STOP = {"the", "by", "and", "for", "of", "a", "an", "co", "inc"}
_WORD = re.compile(r"[a-z0-9]+")
_TAG = re.compile(r"<[^>]+>")
_AMP = re.compile(r"&amp;", re.I)  # 源数据有 &Amp; 这种大小写，html.unescape 不认


def strip_html(text: str) -> str:
    """去标签 + 反转义（``<br>`` → 空格，``&amp;`` / ``&Amp;`` → ``&``），再折叠空白。"""
    text = html.unescape(_AMP.sub("&", _TAG.sub(" ", text)))
    return re.sub(r"\s+", " ", text).strip()


def needs_brand_prefix(brand: str, title: str) -> bool:
    """标题缺品牌才加前缀：占位值不加；品牌首个实词已在标题前 3 个词里也不加
    （``SAXX Underwear Co.`` + ``SAXX Men's ...`` 会叠成两遍）。"""
    if not brand or brand.lower() in _NO_PREFIX_STORES or brand.lower() in title.lower():
        return False
    words = [w for w in _WORD.findall(brand.lower()) if len(w) >= 2 and w not in _BRAND_STOP]
    return not (words and words[0] in _WORD.findall(title.lower())[:3])


def variant_rule(item_id: str, our_title: str, mc: dict) -> int:
    """按定稿规则判 1 / 2 / 3。"""
    if item_id == mc.get("parent_asin"):
        return 1
    ours, theirs = size_tokens(our_title), size_tokens(mc.get("title") or "")
    if theirs and ours != theirs:  # 含「McAuley 有尺寸、我们没有」：判不了，按冲突处理
        return 3
    return 2


def merge_record(rec: dict, mc: dict | None) -> dict:
    """主库一条 + 可能的 McAuley 一条 → 并入后的一条（不改入参）。"""
    out = dict(rec)
    mats = title_materials(rec["title"])
    specs: list[str] = []
    if mc is not None:
        rule = variant_rule(rec["item_id"], rec["title"], mc)
        store = (mc.get("store") or "").strip()
        if store and not out.get("brand"):
            out["brand"] = store
        brand = out.get("brand") or ""
        if needs_brand_prefix(brand, rec["title"]):
            out["title"] = f"{brand} {rec['title']}"
        if rule in (1, 2):
            feats = clean_features(strip_html(f) for f in mc.get("features") or [])
            out["features"] = feats
            if not out.get("description") and mc.get("description"):
                out["description"] = strip_html(mc["description"])
            for m in feature_materials(feats, mc.get("details")):
                if m not in mats:
                    mats.append(m)
            if rule == 1:
                specs = feature_specs(feats)
        if mc.get("rating_number"):
            out["parent_rating_count"] = int(mc["rating_number"])
        # 细类目按父 ASIN 取、不看变体规则：规则③只是尺寸/件数冲突，品类不随变体变。
        out["fine_category"] = fine_category_path(mc.get("categories") or [])
        out["mcauley_rule"] = rule
        out["mcauley_parent_asin"] = mc.get("parent_asin", "")
    out["materials"] = mats
    out["attr_line"] = attr_line(mats, specs)
    return out


def _load_enrich(enrich_dir: Path) -> dict[str, bytes]:
    """item_id → McAuley 原始行。同一 item 出现在多个品类文件时取第一条。"""
    idx: dict[str, bytes] = {}
    for fn in sorted(enrich_dir.glob("enrich_*.jsonl")):
        with fn.open("rb") as f:
            for line in f:
                item_id = json.loads(line)["item_id"]
                idx.setdefault(item_id, line)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--enrich-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve() == args.main.resolve():
        sys.exit("--out 不能等于 --main：并入产物是新文件，原主库不覆盖")

    enrich = _load_enrich(args.enrich_dir)
    print(f"McAuley 记录 {len(enrich):,} 条", flush=True)
    stats: collections.Counter[str] = collections.Counter()
    with args.main.open() as src, args.out.open("w") as dst:
        for line in src:
            rec = json.loads(line)
            raw = enrich.get(rec["item_id"])
            merged = merge_record(rec, json.loads(raw) if raw else None)
            stats["total"] += 1
            stats[f"rule{merged.get('mcauley_rule', 0)}"] += 1
            stats["brand"] += bool(merged.get("brand"))
            stats["materials"] += bool(merged["materials"])
            stats["title_changed"] += merged["title"] != rec["title"]
            dst.write(json.dumps(merged, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
