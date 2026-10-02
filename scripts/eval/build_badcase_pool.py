"""picker 排序评测 · 真实池回归：线上 badcase 原样跑 planner + item_search，冻结真实召回池。

封闭池（``build_picker_replay.py``）里没有真实召回带进来的蹭词货（跑步短裤、板鞋冒充跑鞋），
这里补上：池子 = 线上同口径 item_search 的结果，标签按细类目路径判
（``fine_category`` 同时含全部指定路径词 = 相关 1，否则 0）；
没有细类目的件逐条人工标在 ``MANUAL`` 里，未标的不进标签（= 未判定）。

输出 ``data/eval/picker_replay/badcase.jsonl``，格式同 ``replay.jsonl``（set="badcase"）。

用法::

    uv run --extra db python scripts/eval/build_badcase_pool.py [--show]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.api.context import set_original_query  # noqa: E402
from app.api.run_state import reset_run_state  # noqa: E402
from app.tools._candidates import registry_snapshot  # noqa: E402
from app.tools.item_search import item_search  # noqa: E402
from app.tools.planner import planner  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

OUT = ROOT / "data/eval/picker_replay"
CASES = [
    {
        "qid": "b594ee78",
        "query": "adidas 男士黑色跑鞋",
        "search": {"query": "adidas men's running shoes black", "brand": ["adidas"]},
        "relevant_path": ["shoes", "running"],  # 路径须同时含（短裤路径也有 Running）
    },
]
MANUAL: dict[str, dict[str, int]] = {
    "b594ee78": {"B08JMD4LMM": 1},  # Racer TR21 Running Shoe（看标题判，无细类目）
}


async def _one(case: dict) -> dict:
    sd = OUT / "sessions" / f"badcase_{case['qid']}"
    sd.mkdir(parents=True, exist_ok=True)
    reset_run_state(sd)
    with thread_scope(f"rankeval-badcase-{case['qid']}", sd):
        set_original_query(case["query"])
        plan = await planner(intent=case["query"])
        await item_search(platform="amazon", **case["search"])
        cands = registry_snapshot()
    reset_run_state(sd)
    labels = {}
    for c in cands:
        path = (c.fine_category or "").lower()
        if path:
            labels[c.item_id] = int(all(p in path for p in case["relevant_path"]))
        elif c.item_id in MANUAL.get(case["qid"], {}):
            labels[c.item_id] = MANUAL[case["qid"]][c.item_id]
    return {
        "set": "badcase",
        "qid": case["qid"],
        "query": case["query"],
        "labels": labels,
        "plan": plan.model_dump(mode="json"),
        "search_text": case["search"]["query"],
        "cands": [c.model_dump(mode="json") for c in cands],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="打印池子与标签，方便补 MANUAL")
    ap.add_argument("--out", default="badcase.jsonl")
    args = ap.parse_args()
    recs = [await _one(c) for c in CASES]
    with (OUT / args.out).open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in recs:
        print(r["qid"], "pool", len(r["cands"]), "labeled", len(r["labels"]))
        if args.show:
            for c in r["cands"]:
                lab = r["labels"].get(c["item_id"], "?")
                path = c.get("fine_category") or "-"
                print(f"  {lab} {c['item_id']} {c['score']:.3f} | {path[:40]} | {c['title'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
