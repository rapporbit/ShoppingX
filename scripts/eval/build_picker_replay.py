"""picker 排序评测 · 第 1 步：每条 query 跑一次 planner + 取回封闭池候选，冻结成 ``replay.jsonl``。

冻结之后调权重 / 改公式只做离线重放（``eval_picker_rank.py``），不再碰 planner（唯一花钱的一步）。

- **planner**：线上精排 query = planner 判的品类（字段定义是中文品类名，+ 显式 must），
  P_t 也由它落，所以每条必须过一次；
  在独立 ``thread_scope`` 里跑真 ``planner``，存它返回的 ``PlanOutput``（已含预算折算）。
- **候选**：池 = ``qrels.jsonl`` 里库内已判件，从线上 Qdrant **只读**取 payload；向量分用 planner
  检索词（缺省退回原 query）编码后在这批 id 内 ``query_points`` 拿——
  它只决定精排额度（30）截哪段尾巴，
  与线上 item_search 同口径（模型拿 planner 检索词去搜）。
- 断点续跑：输出里已有的 (set, qid) 跳过。

用法::

    uv run --extra db python scripts/eval/build_picker_replay.py [--limit 5] [--concurrency 4]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from qdrant_client import models  # noqa: E402

from app.api.run_state import reset_run_state  # noqa: E402
from app.recall.qdrant_store import COLLECTION, DENSE_VEC, get_recall_client  # noqa: E402
from app.recall.schemas import RecallCandidate  # noqa: E402
from app.recall.towers import get_tower_client  # noqa: E402
from app.tools.planner import planner  # noqa: E402
from app.tools.schemas import ItemCandidate  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

OUT = ROOT / "data/eval/picker_replay"
SESS = OUT / "sessions"


def _pool(vec: list[float], ids: list[str]) -> list[ItemCandidate]:
    flt = models.Filter(must=[models.FieldCondition(key="item_id", match=models.MatchAny(any=ids))])
    res = get_recall_client().client.query_points(
        COLLECTION, query=vec, using=DENSE_VEC, query_filter=flt, limit=len(ids), with_payload=True
    )
    return [
        ItemCandidate.from_recall(RecallCandidate(**(p.payload or {}), score=float(p.score)))
        for p in res.points
    ]


async def _one(row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        sd = SESS / f"{row['set']}_{row['qid']}"
        sd.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with thread_scope(f"rankeval-{row['set']}-{row['qid']}", sd):
            plan = await planner(intent=row["query"])
        reset_run_state(sd)
        search_text = " ".join(plan.keywords) or plan.category or row["query"]
        vec = (await get_tower_client().encode_query(search_text)).tolist()
        cands = await asyncio.to_thread(_pool, vec, list(row["labels"]))
        return {
            **row,
            "plan": plan.model_dump(mode="json"),
            "search_text": search_text,
            "cands": [c.model_dump(mode="json") for c in cands],
            "secs": round(time.monotonic() - t0, 2),
        }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑用）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="replay.jsonl", help="planner 改版后另存一份对照")
    args = ap.parse_args()
    rows = [json.loads(ln) for ln in (OUT / "qrels.jsonl").open()]
    out_path = OUT / args.out
    done = set()
    if out_path.exists():
        done = {(r["set"], r["qid"]) for r in map(json.loads, out_path.open())}
    todo = [r for r in rows if (r["set"], r["qid"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"total {len(rows)} done {len(done)} todo {len(todo)}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    ok = fail = 0
    with out_path.open("a") as f:
        for fut in asyncio.as_completed([_one(r, sem) for r in todo]):
            try:
                rec = await fut
            except Exception as e:  # 单条失败不拖垮整批，重跑会补
                fail += 1
                print(f"[fail] {type(e).__name__}: {e}"[:200], flush=True)
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{len(todo)}", flush=True)
    print(f"ok {ok} fail {fail}")


if __name__ == "__main__":
    asyncio.run(main())
