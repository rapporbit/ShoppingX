"""picker 排序评测 · 第 2 步：在冻结的封闭池上离线重放 picker，出 nDCG@3 / 前 3 不相关件数。

重放口径 = 线上普通轮的 autopick：P_t 由冻结的 ``PlanOutput`` 经 ``_sync_turn_constraints`` 落，
``item_picker`` 无参（``_prepare_inputs`` 全取 P_t），然后逐段跑
②硬过滤 ③相关性门 ④语义分 ⑤综合分 ⑥定稿。
套装 plan 也按普通轮评（不落槽位），报告里单列条数。

四种排序同表对照（都只在封闭池里排）：
- ``shown``：线上实际展示的 ≤3 件（综合分 + 去重 + 展示相对门）—— **主指标**；
- ``scored``：综合分全序的前 3（不过展示门）；
- ``rerank``：只按精排分排（门没开的条退回向量序）—— 「精排分直接排」能到哪；
- ``vector``：只按向量分排 —— 不精排的下限。

增益：ESCI E/S/C/I = 1/0.1/0.01/0（KDD Cup 官方）；TREC23/24 = 标签值 0~3（trec_eval 线性）；
TREC25 = 3→10、2→1、其余 0（官方 ``ndcg_0=0,1=0,2=1,3=10``）。不相关 = ESCI I / TREC 0；
配件档 = ESCI C / TREC 1，单列。

reranker / embedding 结果缓存到 ``cache/``（自产 pickle），调权重重跑不再打 API。
权重走 picker 现有 env
（``PICK_W_*`` 等），同进程 ``_load_params()`` 生效。

用法::

    uv run --extra db python scripts/eval/eval_picker_rank.py [--tag baseline]
"""

import argparse
import asyncio
import contextvars
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.api.context import set_original_query  # noqa: E402
from app.api.run_state import reset_run_state  # noqa: E402
from app.recall.reranker import get_reranker  # noqa: E402
from app.recall.towers import get_tower_client  # noqa: E402
from app.tools import item_picker as ip  # noqa: E402
from app.tools._candidates import register  # noqa: E402
from app.tools.planner import PlanOutput, _sync_turn_constraints  # noqa: E402
from app.tools.schemas import ItemCandidate  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

OUT = ROOT / "data/eval/picker_replay"
CACHE = OUT / "cache"
K = 3
GAIN = {
    "esci": {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0},
    "trec23": {3: 3.0, 2: 2.0, 1: 1.0, 0: 0.0},
    "trec24": {3: 3.0, 2: 2.0, 1: 1.0, 0: 0.0},
    "trec25": {3: 10.0, 2: 1.0, 1: 0.0, 0: 0.0},
    "badcase": {1: 1.0, 0: 0.0},  # 真实池回归（build_badcase_pool.py），二值
}
IRRELEVANT = {"I", 0}
ACCESSORY = {"C", 1}


def _install_caches() -> tuple[dict, dict]:
    """reranker / embedding 的磁盘缓存（只缓存远程成功的结果，降级的不缓存）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    rr_path, em_path = CACHE / "rerank.pkl", CACHE / "embed.pkl"
    rr = pickle.loads(rr_path.read_bytes()) if rr_path.exists() else {}
    em = pickle.loads(em_path.read_bytes()) if em_path.exists() else {}
    reranker, tower = get_reranker(), get_tower_client()
    orig_rr, orig_em = reranker.score_detailed, tower.encode_texts

    async def score_detailed(query: str, docs: list[str]) -> tuple[list[float], bool]:
        miss = [d for d in dict.fromkeys(docs) if (query, d) not in rr]
        if miss:
            got, remote = await orig_rr(query, miss)
            if not remote:
                return [0.0] * len(docs), False
            rr.update({(query, d): s for d, s in zip(miss, got, strict=True)})
        return [rr[query, d] for d in docs], True

    async def encode_texts(texts: list[str]) -> np.ndarray:
        miss = [t for t in dict.fromkeys(texts) if t not in em]
        if miss:
            em.update(zip(miss, await orig_em(miss), strict=True))
        return np.stack([em[t] for t in texts]) if texts else await orig_em([])

    reranker.score_detailed = score_detailed  # type: ignore[method-assign]
    tower.encode_texts = encode_texts  # type: ignore[method-assign]
    return rr, em


def _save_caches(rr: dict, em: dict) -> None:
    (CACHE / "rerank.pkl").write_bytes(pickle.dumps(rr))
    (CACHE / "embed.pkl").write_bytes(pickle.dumps(em))


RERANK_QUERY = "category"  # 诊断开关：精排 query 用什么（category=线上口径：英文名优先）


W_REL = 0.0  # 实验开关：综合分加 W_REL × 精排分（0 = 线上口径）
_ORIG_SCORE = ip._score_candidates


def _score_with_rel(rel, inputs, sem, cheapness):  # noqa: ANN001, ANN202 —— 与 picker 同签名
    """原综合分 + W_REL × 精排分，重排。门没开（rel.on=False）的条不动——没有信号不硬加。"""
    scored = _ORIG_SCORE(rel, inputs, sem, cheapness)
    if W_REL <= 0 or not rel.on:
        return scored
    rows = [(s + W_REL * rel.scores.get(c.item_id, 0.0), m, a, c) for s, m, a, c in scored.rows]
    rows.sort(key=lambda t: t[0], reverse=True)
    return scored._replace(rows=rows)


ip._score_candidates = _score_with_rel


TITLE_QUERY = "same"  # 实验开关：same=标题与路径同一 query；keywords=标题分改用 planner 检索词
_TITLE_Q: contextvars.ContextVar[str] = contextvars.ContextVar("title_q", default="")
_PATHS: contextvars.ContextVar[frozenset] = contextvars.ContextVar("paths", default=frozenset())


def _install_split_query() -> None:
    """picker 一次请求里先送标题、后送细类目路径；按文本认出路径，两段各用各的 query 打分。"""
    reranker = get_reranker()
    inner = reranker.score_detailed

    async def score_detailed(query: str, docs: list[str]) -> tuple[list[float], bool]:
        tq, paths = _TITLE_Q.get(), _PATHS.get()
        if not tq:
            return await inner(query, docs)
        ti = [i for i, d in enumerate(docs) if d not in paths]
        pi = [i for i, d in enumerate(docs) if d in paths]
        st, ok1 = await inner(tq, [docs[i] for i in ti])
        sp, ok2 = await inner(query, [docs[i] for i in pi]) if pi else ([], True)
        out = [0.0] * len(docs)
        for i, v in zip(ti + pi, st + sp, strict=True):
            out[i] = v
        return out, ok1 and ok2

    reranker.score_detailed = score_detailed  # type: ignore[method-assign]


def _plan_for_eval(rec: dict) -> PlanOutput:
    """按诊断开关改写 plan.category（精排 query 的来源）；category 口径原样返回。"""
    plan = PlanOutput.model_validate(rec["plan"])
    text = None
    if RERANK_QUERY == "keywords" and plan.keywords:
        text = " ".join(plan.keywords)
    elif RERANK_QUERY == "query":
        text = rec["query"]
    elif RERANK_QUERY.startswith("fixed:"):  # 抖动敏感度实验：手给一个品类写法
        text = RERANK_QUERY.removeprefix("fixed:")
    elif RERANK_QUERY == "category_zh":  # 只用中文品类（= category_en 上线前的口径）
        plan.category_en = ""
    if text is not None:  # 诊断文本原样当 query（不过英文名清洗），清空 category_en 免得它优先
        plan.category, plan.category_en = text, ""
    return plan


async def _replay(rec: dict) -> dict:
    """在独立会话作用域里按 autopick 口径重放一条，返回四种排序（item_id 列表）。"""
    sd = OUT / "sessions" / f"eval_{rec['set']}_{rec['qid']}"
    reset_run_state(sd)
    if TITLE_QUERY == "keywords" and rec["plan"].get("keywords"):
        _TITLE_Q.set(" ".join(rec["plan"]["keywords"]))
        _PATHS.set(frozenset((c.get("fine_category") or "").lower() for c in rec["cands"]))
    with thread_scope(f"rankeval-{rec['set']}-{rec['qid']}", sd):
        _sync_turn_constraints(_plan_for_eval(rec))
        set_original_query(rec["query"])
        cands = [ItemCandidate.model_validate(c) for c in rec["cands"]]
        register(cands)
        inputs = await ip._prepare_inputs(None, None, None, None, None)
        filtered = ip._hard_filter(cands, inputs)
        rel = await ip._relevance_gate(filtered.survivors, inputs.hard_must)
        sem = await ip._semantic_scores(rel.survivors, inputs)
        scored = ip._score_candidates(rel, inputs, sem, filtered.cheapness)
        picks = ip._picks_from_pool(scored, ip.PICK_DISPLAY_CAP, rel)
    reset_run_state(sd)
    # 对照组也在硬过滤后的幸存池里排：排除词 / 预算是三种排序共用的前置，不该只让某一种吃到。
    survivors = filtered.survivors
    vec = [c.item_id for c in sorted(survivors, key=lambda c: c.score or 0.0, reverse=True)]
    rr = sorted(rel.scores, key=rel.scores.get, reverse=True) if rel.on else vec  # type: ignore[arg-type]
    return {
        "shown": [p.item_id for p in picks],
        "scored": [row[3].item_id for row in scored.rows],
        "rerank": rr,
        "vector": vec,
        "gate_on": rel.on,
        "scores": {k: round(v, 4) for k, v in rel.scores.items()},
    }


def _ndcg(ranked: list[str], labels: dict, gains: dict, pool: set[str]) -> float:
    g = [gains.get(labels.get(i), 0.0) for i in ranked[:K]]
    ideal = sorted((gains.get(labels[i], 0.0) for i in pool if i in labels), reverse=True)[:K]
    dcg = sum(x / math.log2(r + 2) for r, x in enumerate(g))
    idcg = sum(x / math.log2(r + 2) for r, x in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _metrics(rec: dict, out: dict) -> dict:
    labels, gains = rec["labels"], GAIN[rec["set"]]
    labels = {k: (int(v) if rec["set"] != "esci" else v) for k, v in labels.items()}
    pool = {c["item_id"] for c in rec["cands"]}
    m = {}
    for view in ("shown", "scored", "rerank", "vector"):
        top = out[view][:K]
        m[f"ndcg3_{view}"] = _ndcg(top, labels, gains, pool)
        m[f"irr3_{view}"] = sum(labels.get(i) in IRRELEVANT for i in top)
        m[f"acc3_{view}"] = sum(labels.get(i) in ACCESSORY for i in top)
    m["n_shown"] = len(out["shown"])
    m["gate_on"] = int(out["gate_on"])
    m["bundle_plan"] = int(bool(rec["plan"].get("bundle_slots")))
    return m


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--rerank-query",
        default="category",
        help="category|category_zh|keywords|query|fixed:<文本>",
    )
    ap.add_argument("--w-rel", type=float, default=0.0)
    ap.add_argument("--title-query", default="same", help="same|keywords")
    ap.add_argument("--replay", default="replay.jsonl")
    ap.add_argument(
        "--set", action="append", default=[], help="覆盖 picker 权重常量，如 _W_RATING=0"
    )
    args = ap.parse_args()
    global RERANK_QUERY, W_REL, TITLE_QUERY
    RERANK_QUERY, W_REL, TITLE_QUERY = args.rerank_query, args.w_rel, args.title_query
    ip._load_params()  # 让本进程 env 覆盖的权重生效
    for kv in args.set:  # 消融用：直接改模块常量（_W_PREF / _W_RATING / _W_CHEAP 不走 env）
        name, val = kv.split("=", 1)
        assert hasattr(ip, name), name
        setattr(ip, name, float(val))
    recs = [json.loads(ln) for ln in (OUT / args.replay).open()]
    rr, em = _install_caches()
    _install_split_query()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec: dict) -> tuple[dict, dict]:
        async with sem:
            return rec, await _replay(rec)

    rows, agg = [], defaultdict(list)
    try:
        for rec, out in await asyncio.gather(*(one(r) for r in recs)):
            m = _metrics(rec, out)
            rows.append({"set": rec["set"], "qid": rec["qid"], "query": rec["query"], **out, **m})
            for s in (rec["set"], "ALL"):
                for k, v in m.items():
                    agg[s, k].append(v)
    finally:
        _save_caches(rr, em)
    sets = sorted({r["set"] for r in recs}) + ["ALL"]
    summary = {
        s: {k: round(sum(v) / len(v), 4) for (s2, k), v in agg.items() if s2 == s} for s in sets
    }
    for s in sets:
        summary[s]["n"] = len(agg[s, "gate_on"])
    (OUT / f"result_{args.tag}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False)
    )
    cols = [
        "n",
        "gate_on",
        "n_shown",
        "ndcg3_shown",
        "ndcg3_scored",
        "ndcg3_rerank",
        "ndcg3_vector",
        "irr3_shown",
        "irr3_rerank",
        "acc3_shown",
        "acc3_rerank",
    ]
    print("set     " + " ".join(f"{c:>12}" for c in cols))
    for s in sets:
        print(f"{s:7} " + " ".join(f"{summary[s].get(c, 0):>12}" for c in cols))


if __name__ == "__main__":
    asyncio.run(main())
