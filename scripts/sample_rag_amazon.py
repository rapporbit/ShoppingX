"""从 ``data/rag/amazon_products.csv``（142 万条）蓄水池抽样 N 条 → amazon 平台扩充表。

「A 路」落地（见召回数据扩增评估）：用这份 142 万 Amazon 商品库补品类覆盖 + 真实
url/image + 销量信号，代价是**无 brand/description**（embed_text 退化为 ``title | 类目``）。

抽样：固定 seed 的**蓄水池抽样**（单遍、可复现、保持原始品类分布），并排除已在现有
``amazon.jsonl`` 里的 asin，避免与 CSV 源重复。产物 CleanItem 写到
``data/platforms/clean/by_platform/amazon_rag.jsonl``；``build_item_index.py`` 会把它作为
amazon 平台的**额外源**与现有 995 条一并建索引（平台名仍标 ``amazon``）。

用法：
    uv run python scripts/sample_rag_amazon.py                 # 默认抽 50000
    uv run python scripts/sample_rag_amazon.py -n 20000 --seed 7
下一步（必须**全平台**一起重建——build_item_index 首平台会 recreate 整个 collection，
只跑 amazon 会把其它平台从库里删掉）：
    uv run python scripts/build_item_index.py --require-remote
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 标题/类目字段无超大单元格，10MB 上限足够且防 OverflowError（对齐 clean_platforms.py）。
csv.field_size_limit(10 * 1024 * 1024)

from app.utils.clean import clean_rag_amazon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "rag" / "amazon_products.csv"
CAT = ROOT / "data" / "rag" / "amazon_categories.csv"
EXISTING = ROOT / "data" / "platforms" / "clean" / "by_platform" / "amazon.jsonl"
OUT = ROOT / "data" / "platforms" / "clean" / "by_platform" / "amazon_rag.jsonl"


def load_categories() -> dict[str, str]:
    """category_id → 类目名 维表。"""
    with CAT.open(encoding="utf-8") as f:
        return {r["id"].strip(): r["category_name"].strip() for r in csv.DictReader(f)}


def existing_asins() -> set[str]:
    """现有 amazon.jsonl 里的 item_id（asin）集合——抽样时排除，避免与 CSV 源重复。"""
    if not EXISTING.exists():
        return set()
    out: set[str] = set()
    with EXISTING.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line).get("item_id", ""))
    return out


def reservoir_sample(n: int, seed: int, skip: set[str]) -> list[dict]:
    """单遍蓄水池抽样：等概率取 n 行、保持原始品类分布，固定 seed 可复现。"""
    rng = random.Random(seed)
    res: list[dict] = []
    seen = 0
    with SRC.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("asin") or "").strip() in skip:
                continue
            seen += 1
            if len(res) < n:
                res.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    res[j] = row
    return res


def read_all(skip: set[str]) -> list[dict]:
    """全量读入（不抽样），仅排除已在 amazon.jsonl 里的 asin。"""
    with SRC.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("asin") or "").strip() not in skip]


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG amazon 抽样 → amazon 平台扩充表")
    ap.add_argument("-n", type=int, default=50000, help="抽样条数（默认 50000）")
    ap.add_argument("--seed", type=int, default=42, help="蓄水池抽样随机种子（可复现）")
    ap.add_argument("--all", action="store_true", help="全量入库，不抽样（忽略 -n / --seed）")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"❌ 找不到源数据 {SRC}（先跑 scripts/fetch_milistu.py 同级的数据准备）")

    cat = load_categories()
    skip = existing_asins()
    print(f"维表类目 {len(cat)} 个；现有 amazon asin {len(skip)} 个（将排除）")

    if args.all:
        rows = read_all(skip)
        print(f"全量读入 {len(rows):,} 行 → 清洗中…")
    else:
        rows = reservoir_sample(args.n, args.seed, skip)
        print(f"蓄水池抽样 {len(rows)} 行（seed={args.seed}）→ 清洗中…")
    items, stats = clean_rag_amazon(rows, cat)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.model_dump(), ensure_ascii=False) + "\n")

    # —— 报告 ——
    drops = "  ".join(
        f"{k.removeprefix('drop_')}={v}" for k, v in stats.items() if k.startswith("drop_")
    )
    kept = len(items)
    with_sold = sum(1 for it in items if it.sold)
    with_price = sum(1 for it in items if it.price)
    cats = Counter(it.category or "(空)" for it in items)
    print(f"\n抽样 {stats['read']} → 保留 {kept}    {drops or '（无丢弃）'}")
    print(f"  带销量(sold>0): {with_sold} ({with_sold * 100 // max(kept, 1)}%)")
    print(f"  有价格:        {with_price} ({with_price * 100 // max(kept, 1)}%)")
    print(f"  覆盖类目 {len(cats)} 个，Top5: {[c for c, _ in cats.most_common(5)]}")
    print(f"\n产物 → {OUT}")
    print("下一步（必须全平台一起，否则其它平台会被 recreate 删掉）：")
    print("    uv run python scripts/build_item_index.py --require-remote")


if __name__ == "__main__":
    main()
