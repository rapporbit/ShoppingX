"""给线上 Qdrant 补 ``fine_category`` payload（不重编码、不动向量）。

输入是一份 TSV：``item_id<TAB>McAuley categories 用 " > " 连起来的原始路径``，由 gcjp 上
``/opt/globex-enrich/out/enrich_*.jsonl`` 导出（见 docstring 末尾）。路径经
:func:`app.utils.clean.fine_category_path` 去根、去人群层后写入。

点 id 是建库时的顺序整数、与 item_id 无关，所以不按点 id 写，而是**按清洗后的路径分组**，
每组一个 ``SetPayloadOperation``（filter = item_id MatchAny），批量提交。item_id 有 keyword
索引（「搜同款」加的），按它过滤不扫全库。幂等：重跑只是把同样的值再写一遍。

导出 TSV（在 gcjp 宿主机）::

    cd /opt/globex-enrich/out && python3 -c "
    import json, glob
    for f in sorted(glob.glob('enrich_*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            print(r['item_id'] + '\\t' + ' > '.join(r.get('categories') or []))
    " > /tmp/all_cats.tsv

用法（容器内，带 .env 的 QDRANT_URL）::

    python scripts/backfill_fine_category.py /tmp/all_cats.tsv --dry-run
    python scripts/backfill_fine_category.py /tmp/all_cats.tsv
"""

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import models  # noqa: E402

from app.recall.qdrant_store import COLLECTION, make_client  # noqa: E402
from app.utils.clean import fine_category_path  # noqa: E402

IDS_PER_OP = 1000  # 单个 MatchAny 的 id 上限，大组拆成多条操作
OPS_PER_REQUEST = 200


def load_groups(tsv: Path) -> dict[str, list[str]]:
    """清洗后的路径 → item_id 列表。同一 item 出现多次取第一条（与 merge_mcauley 一致）。"""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[str] = set()
    with tsv.open(encoding="utf-8") as f:
        for line in f:
            item_id, _, raw = line.rstrip("\n").partition("\t")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            path = fine_category_path(raw.split(" > ") if raw else [])
            if path:
                groups[path].append(item_id)
    return groups


def build_ops(groups: dict[str, list[str]]) -> list[models.SetPayloadOperation]:
    ops: list[models.SetPayloadOperation] = []
    for path, ids in groups.items():
        for i in range(0, len(ids), IDS_PER_OP):
            cond = models.FieldCondition(
                key="item_id", match=models.MatchAny(any=ids[i : i + IDS_PER_OP])
            )
            ops.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={"fine_category": path}, filter=models.Filter(must=[cond])
                    )
                )
            )
    return ops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写")
    args = ap.parse_args()

    groups = load_groups(args.tsv)
    n_items = sum(len(v) for v in groups.values())
    ops = build_ops(groups)
    print(f"路径 {len(groups)} 种，商品 {n_items} 件，操作 {len(ops)} 条 → {COLLECTION}")
    if args.dry_run:
        for path, ids in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]:
            print(f"  {len(ids):>7}  {path}")
        return

    client = make_client(timeout=120)
    t0 = time.monotonic()
    for i in range(0, len(ops), OPS_PER_REQUEST):
        client.batch_update_points(COLLECTION, ops[i : i + OPS_PER_REQUEST], wait=True)
        done = min(i + OPS_PER_REQUEST, len(ops))
        print(f"  {done}/{len(ops)}  {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
