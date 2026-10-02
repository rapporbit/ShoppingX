"""给线上 Qdrant 补 ``brand`` payload：brand 为空、标题开头对上库内已知品牌的商品。

不重编码、不动向量。

背景：amazon 主库原始 brand 全空，品牌是并入 McAuley 时按 item_id 补的，没对上的 50.9 万件
（36.8%）仍为空。其中一部分标题开头就是品牌（``AUSDOM Bluetooth Noise Cancelling ...``），而同品牌
别的商品已有 brand —— 用「库内已有的 brand 值」当词表，只补这一类。标题里没有品牌的不猜。

词表规则（:func:`build_vocab`）：占位值不收；全库不足 ``MIN_BRAND_ITEMS`` 件的不收（孤例多是店铺名
或脏值）；每个词都是标题常见词的不收（``Police`` / ``Home Decor`` 会把通用标题误判成品牌）。
匹配规则（:func:`brand_from_title`）：标题前 1~3 个词与品牌名整词相等，不分大小写，取最长。

三步走，写库与回滚都只认 scan 产出的那份 TSV（``item_id<TAB>brand``），所以写进去的就是抽检过的::

    python scripts/backfill_brand_from_title.py scan --out /tmp/brand_plan.tsv  # 只读
    python scripts/backfill_brand_from_title.py apply /tmp/brand_plan.tsv   # 只写 brand 仍为空的点
    python scripts/backfill_brand_from_title.py revert /tmp/brand_plan.tsv  # 把写过的改回空串

环境变量：``QDRANT_URL`` / ``QDRANT_COLLECTION``。点 id 与 item_id 无关，写入按 item_id 过滤
（有 keyword 索引），按 brand 分组、每组一个 ``SetPayloadOperation``。
"""

import argparse
import collections
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, make_client  # noqa: E402

MIN_BRAND_ITEMS = 3  # 词表准入：全库至少这么多件商品带这个 brand
COMMON_WORD_RATIO = 0.0005  # 一个词在 ≥0.05% 的标题非首位出现过，就算常见词
MAX_BRAND_WORDS = 3  # 只拿标题前 3 个词去对，更长的品牌名不进词表
SCROLL_PAGE = 5000
IDS_PER_OP = 1000  # 单个 MatchAny 的 id 上限，大组拆成多条操作
OPS_PER_REQUEST = 200

# 与 merge_mcauley._NO_PREFIX_STORES 同一批占位值：是 brand 字段里的真实取值，但不是品牌。
_PLACEHOLDERS = {"amazon renewed", "generic", "unknown", "artist unknown"}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'+.\-]*")


def tokens(text: str) -> list[str]:
    """切词并转小写，词尾的 ``. - '`` 去掉（``Co.`` → ``co``，``Bucilla,`` → ``bucilla``）。"""
    return [t for t in (m.rstrip(".-'+&").lower() for m in _TOKEN.findall(text or "")) if t]


def build_vocab(
    brand_counts: dict[str, int], word_freq: dict[str, int], n_titles: int
) -> dict[str, str]:
    """规范化品牌名（小写、空格连接）→ 写入用的写法（同名多种大小写取件数最多的那种）。"""
    total: dict[str, int] = collections.Counter()
    best: dict[str, tuple[int, str]] = {}
    for brand, n in brand_counts.items():
        key = " ".join(tokens(brand))
        if not key:
            continue
        total[key] += n
        if n > best.get(key, (0, ""))[0]:
            best[key] = (n, brand.strip())

    common_at = max(1.0, n_titles * COMMON_WORD_RATIO)

    def is_common(word: str) -> bool:
        return len(word) < 2 or word.isdigit() or word_freq.get(word, 0) >= common_at

    vocab: dict[str, str] = {}
    for key, n in total.items():
        words = key.split(" ")
        if key in _PLACEHOLDERS or n < MIN_BRAND_ITEMS or len(words) > MAX_BRAND_WORDS:
            continue
        if all(is_common(w) for w in words):
            continue
        vocab[key] = best[key][1]
    return vocab


def brand_from_title(title: str, vocab: dict[str, str]) -> str:
    """标题前 1~3 个词整词等于某个品牌名就返回它（最长优先），否则空串。"""
    head = tokens(title)[:MAX_BRAND_WORDS]
    for n in range(len(head), 0, -1):
        hit = vocab.get(" ".join(head[:n]))
        if hit:
            return hit
    return ""


def scan(out: Path, sample: int) -> None:
    """全库滚一遍（只读）：攒品牌件数与标题词频 → 建词表 → 对空品牌商品逐条匹配 → 写计划 TSV。"""
    client = make_client(timeout=120)
    brand_counts: dict[str, int] = collections.Counter()
    word_freq: dict[str, int] = collections.Counter()
    empty: list[tuple[str, str]] = []  # (item_id, title)
    n, offset, t0 = 0, None, time.monotonic()
    while True:
        points, offset = client.scroll(
            COLLECTION,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=["item_id", "title", "brand"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            title, brand = str(pl.get("title") or ""), str(pl.get("brand") or "").strip()
            word_freq.update(set(tokens(title)[1:]))
            if brand:
                brand_counts[brand] += 1
            elif pl.get("item_id"):
                empty.append((str(pl["item_id"]), title))
        n += len(points)
        if n % 100_000 < SCROLL_PAGE:
            print(f"  已扫 {n}  {time.monotonic() - t0:.0f}s", flush=True)
        if offset is None:
            break

    vocab = build_vocab(brand_counts, word_freq, n)
    plan = [(i, b) for i, t in empty if (b := brand_from_title(t, vocab))]
    out.write_text("".join(f"{i}\t{b}\n" for i, b in plan), encoding="utf-8")
    print(f"全库 {n} 件；brand 取值 {len(brand_counts)} 种 → 词表 {len(vocab)} 个")
    print(
        f"brand 为空 {len(empty)} 件，命中 {len(plan)} 件"
        f"（{len(plan) / max(1, len(empty)):.1%}）→ {out}"
    )
    for brand, k in collections.Counter(b for _, b in plan).most_common(10):
        print(f"  {k:>6}  {brand}")
    titles = dict(empty)
    for item_id, brand in random.Random(7).sample(plan, min(sample, len(plan))):
        print(f"  [{brand}] ← {titles[item_id][:70]}")


def _write(plan: Path, *, revert: bool) -> None:
    """apply：只写 brand 仍为空的点；revert：只把 brand 仍等于计划值的点改回空串。重跑安全。"""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for line in plan.read_text(encoding="utf-8").splitlines():
        item_id, _, brand = line.partition("\t")
        if item_id and brand:
            groups[brand].append(item_id)
    ops: list[models.SetPayloadOperation] = []
    for brand, ids in groups.items():
        old, new = (brand, "") if revert else ("", brand)
        for i in range(0, len(ids), IDS_PER_OP):
            must = [
                models.FieldCondition(
                    key="item_id", match=models.MatchAny(any=ids[i : i + IDS_PER_OP])
                ),
                models.FieldCondition(key="brand", match=models.MatchValue(value=old)),
            ]
            ops.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={"brand": new}, filter=models.Filter(must=must)
                    )
                )
            )
    n_items = sum(len(v) for v in groups.values())
    print(
        f"{'回滚' if revert else '写入'}：品牌 {len(groups)} 个，商品 {n_items} 件，"
        f"操作 {len(ops)} 条 → {COLLECTION}"
    )
    client = make_client(timeout=120)
    t0 = time.monotonic()
    for i in range(0, len(ops), OPS_PER_REQUEST):
        client.batch_update_points(COLLECTION, ops[i : i + OPS_PER_REQUEST], wait=True)
        print(
            f"  {min(i + OPS_PER_REQUEST, len(ops))}/{len(ops)}  {time.monotonic() - t0:.0f}s",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_scan = sub.add_parser("scan", help="只读：扫全库，产出计划 TSV 并打印抽检样例")
    p_scan.add_argument("--out", type=Path, required=True)
    p_scan.add_argument("--sample", type=int, default=60, help="打印多少条随机样例")
    for name, text in (
        ("apply", "按计划 TSV 写 brand"),
        ("revert", "按计划 TSV 把 brand 改回空串"),
    ):
        sub.add_parser(name, help=text).add_argument("plan", type=Path)
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.out, args.sample)
    else:
        _write(args.plan, revert=args.cmd == "revert")


if __name__ == "__main__":
    main()
