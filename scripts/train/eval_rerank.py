"""体检第三段：算 rerank 段的指标——深度曲线 × query 形态 A/B，一张表出全部结论。

**这把尺子此前不存在。** M21 的 ``eval_recall.py`` 只量召回段（e15 打到 R@1000=.7714、
R@20=.2905），中间 48 个点的排序空间从没被量过；而 reranker 正是吃这块空间的东西。没有这张
表，「训 reranker 能涨多少」就只能靠猜。

**三条对照，缺一不可：**

- ``embed``    —— 不 rerank，e15 原序截断。这是**线上现状的下界**，也是 M21 基线的复现点。
- ``intent``   —— 现成 bge-reranker-v2-m3，query = 用户完整意图句。
- ``category`` —— 同一个模型，query = 粗品类词（线上 ``item_picker`` 的真实用法，且是作弊版）。

``intent`` 与 ``category`` 的差 = **换用法**能拿到的分；``intent`` 与 ``embed`` 的差 = 现成模型
**不训练**就能拿到的分。训练的收益空间只能从这两个数之上起算——先把它们量出来，再谈要不要训。

**天花板（``ceiling`` 列）** 是该深度候选池里正例的占比：rerank 是排序、不是召回，排得再准也
超不过池子里有什么。它同时解释了「加深 K」这个零训练杠杆能值多少。

口径与 ``eval_recall.py`` 逐条对齐（分母取库内正例数、gain 用 E=3/S=2/C=1 分级），否则两张表
接不上、M21 的基线数就白攒了。

用法::

    uv run --group train python scripts/train/eval_rerank.py \\
        --scores data/train/rerank_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "rerank_report.json"

DEPTHS = [20, 50, 100, 200, 500, 1000]  # rerank 候选深度（K）
GAIN = {"pos": 3.0, "sub": 2.0, "comp": 1.0}
CUTS = [8, 20]  # 线上展示位 PICK_DISPLAY_CAP=8；20 用于跟 M21 的 recall@20 对齐


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def gain_fn(row: dict):
    pos, sub = set(row["positives"]), set(row.get("substitutes") or [])
    comp = set(row.get("complements") or [])

    def g(item_id: str) -> float:
        if item_id in pos:
            return GAIN["pos"]
        if item_id in sub:
            return GAIN["sub"]
        if item_id in comp:
            return GAIN["comp"]
        return 0.0

    return g, pos, sub, comp


def score_ranking(ranked: list[str], row: dict) -> dict[str, float]:
    """给一个排好序的 item_id 列表算指标（分母口径同 eval_recall.py）。"""
    g, pos, sub, comp = gain_fn(row)
    ideal_all = sorted(
        [GAIN["pos"]] * len(pos) + [GAIN["sub"]] * len(sub) + [GAIN["comp"]] * len(comp),
        reverse=True,
    )
    out: dict[str, float] = {}
    for cut in CUTS:
        head = ranked[:cut]
        out[f"recall@{cut}"] = len(pos & set(head)) / len(pos) if pos else 0.0
        ideal = ideal_all[:cut]
        out[f"ndcg@{cut}"] = dcg([g(x) for x in head]) / dcg(ideal) if ideal else 0.0
    rr = 0.0
    for i, x in enumerate(ranked[: max(CUTS)]):
        if x in pos:
            rr = 1.0 / (i + 1)
            break
    out["mrr"] = rr
    # 诊断列：头部里「一个 ESCI 标注都没有」的候选占比。我们库 90.8% 的商品没标注，
    # cross-encoder 若比 embedding 更会找语义真相关的东西，就会把未标注的好货排上来——
    # 表现为指标下降而 unlabeled 上升。这一列是区分「模型差」与「尺子量不了」的关键证据，
    # 它高本身不代表好或坏（可能是好货也可能是垃圾），必须配 --dump-cases 人眼看几条。
    labeled = pos | sub | comp
    head8 = ranked[: CUTS[0]]
    out[f"unlabeled@{CUTS[0]}"] = (
        sum(1 for x in head8 if x not in labeled) / len(head8) if head8 else 0.0
    )
    return out


def accumulate(acc: dict[str, float], metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        acc[k] = acc.get(k, 0.0) + v


def labeled_pool_eval(qrels: dict[int, dict], score_rows: list[dict]) -> dict:
    """口径 B：**只在标注商品之间**排序，量模型区分 E/S/C/I 的纯能力。

    口径 A（全库 top-K 重排）测的是线上真实场景，但被「90.8% 商品没标注」严重污染——模型把
    未标注的好货排上来会被算成扣分。ESCI 官方 benchmark 本来就不是全库检索，而是**给定候选
    列表排序**；这里把候选池收缩到该 query 已被 ESCI 标过的那些商品（且被 top-1000 召回到的），
    未标注商品全部剔除，污染就消失了。

    代价是它不再是线上场景（线上候选池里当然有未标注商品）。所以 A 和 B 要一起看：
    **B 是「模型能力」，A 是「线上收益」**，训练直接优化的是 B，最终要兑现的是 A。

    零额外 GPU 成本：复用口径 A 已经打好的分数，只换个候选池重排。

    **它的天花板必须交代清楚（实测）：** 全量 qrels 里每 query 平均只有 4.8 条标注，且只有
    36.7% 的 query 同时有正例和 S/C——其余的池子全是同一档正例，怎么排 NDCG 都是 1.0，把它们
    算进平均只会把分数注水到 0.99。所以这里**只统计有区分空间的那部分 query**，并把覆盖率一起
    报出来。即便如此 B 的池子仍然很小（个位数），它能说的话有限，不能拿来当唯一的训练靶子。
    """
    acc: dict[str, float] = {}
    n = 0
    n_eligible = 0
    for rec in score_rows:
        row = qrels.get(rec["query_id"])
        scores = rec.get("intent")
        if row is None or not row["positives"] or scores is None:
            continue
        g, pos, sub, comp = gain_fn(row)
        if not (sub or comp):
            continue  # 池内全是同档正例 → NDCG 恒为 1，算进去只会注水
        n_eligible += 1
        labeled = pos | sub | comp
        idx = [i for i, x in enumerate(rec["item_ids"]) if x in labeled]
        if len({g(rec["item_ids"][i]) for i in idx}) < 2:  # 召回到的那些恰好同档，同理跳过
            continue
        n += 1
        ids = [rec["item_ids"][i] for i in idx]
        ideal = dcg(sorted((g(x) for x in ids), reverse=True))
        acc["ndcg_embed"] = acc.get("ndcg_embed", 0.0) + (dcg([g(x) for x in ids]) / ideal)
        order = sorted(idx, key=lambda i: scores[i], reverse=True)
        ranked = [rec["item_ids"][i] for i in order]
        acc["ndcg_rerank"] = acc.get("ndcg_rerank", 0.0) + (dcg([g(x) for x in ranked]) / ideal)
        acc["pool_size"] = acc.get("pool_size", 0.0) + len(ids)
    if not n:
        return {}
    return {
        "n_queries": n,
        "n_eligible": n_eligible,  # 有 S/C 标注的 query 数；n 还要再扣掉「召回到的恰好同档」
        **{k: round(v / n, 4) for k, v in acc.items()},
    }


def run(qrels: dict[int, dict], score_rows: list[dict]) -> dict:
    # 形态 → 深度 → 指标累加器；n 各自计数（category 形态样本少一些）
    acc: dict[str, dict[int, dict[str, float]]] = {}
    counts: dict[str, int] = {}

    for rec in score_rows:
        row = qrels.get(rec["query_id"])
        if row is None or not row["positives"]:
            continue
        item_ids: list[str] = rec["item_ids"]
        forms: list[tuple[str, list[float] | None]] = [
            ("embed", None),  # None = 保持 e15 原序
            ("intent", rec.get("intent")),
        ]
        # 品类词形态只覆盖「top-K 内召回到正例」的那部分 query，这批 query 的召回本来就更好，
        # 拿它的绝对值跟全量 embed/intent 比是**偷跑**。所以在同一子集上再算一份 embed/intent
        # 作为它的对照组（后缀 _sub），三者同分母，差值才有意义。
        if rec.get("category") is not None:
            forms += [
                ("embed_sub", None),
                ("intent_sub", rec.get("intent")),
                ("category", rec.get("category")),
            ]
        for form, scores in forms:
            counts[form] = counts.get(form, 0) + 1
            for depth in DEPTHS:
                pool = item_ids[:depth]
                if scores is None:
                    ranked = pool
                else:
                    s = scores[:depth]
                    order = sorted(range(len(pool)), key=lambda i: s[i], reverse=True)
                    ranked = [pool[i] for i in order]
                bucket = acc.setdefault(form, {}).setdefault(depth, {})
                accumulate(bucket, score_ranking(ranked, row))
                # 天花板：该深度候选池里的正例占全部正例的比例（rerank 的绝对上界）
                pos = set(row["positives"])
                bucket["ceiling"] = bucket.get("ceiling", 0.0) + len(pos & set(pool)) / len(pos)

    report: dict = {"forms": {}, "labeled_pool": labeled_pool_eval(qrels, score_rows)}
    for form, by_depth in acc.items():
        n = counts[form]
        report["forms"][form] = {
            "n_queries": n,
            "by_depth": {
                str(d): {k: round(v / n, 4) for k, v in m.items()} for d, m in by_depth.items()
            },
        }
    return report


def print_table(report: dict) -> None:
    cols = [
        f"recall@{CUTS[0]}",
        f"ndcg@{CUTS[0]}",
        f"recall@{CUTS[1]}",
        "mrr",
        f"unlabeled@{CUTS[0]}",
        "ceiling",
    ]
    print(f"\n{'形态':<10}{'K':>6}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (16 + 12 * len(cols)))
    for form in ("embed", "intent", "embed_sub", "intent_sub", "category"):
        block = report["forms"].get(form)
        if not block:
            continue
        for depth in DEPTHS:
            m = block["by_depth"].get(str(depth))
            if not m:
                continue
            print(f"{form:<10}{depth:>6}" + "".join(f"{m.get(c, 0.0):>12.4f}" for c in cols))
        print(f"{'':<10}{'n=' + str(block['n_queries']):>6}")

    lp = report.get("labeled_pool") or {}
    if lp:
        print(
            f"\n口径 B（只在标注池内排序，n={lp['n_queries']}，池均 {lp['pool_size']:.1f} 条）："
            f"NDCG e15 原序 {lp['ndcg_embed']:.4f} → rerank {lp['ndcg_rerank']:.4f}"
            f"（{lp['ndcg_rerank'] - lp['ndcg_embed']:+.4f}）"
        )


def dump_cases(qrels: dict[int, dict], score_rows: list[dict], cand_path: Path, n: int) -> None:
    """人眼验货：并排打印 e15 原序 top-5 与 intent rerank 后 top-5。

    ``unlabeled`` 那一列只能说明「头部换了一批没标注的东西」，说明不了换上来的是好是坏——
    这个判断机器做不了（我们没有那些商品的标注），只能人看。看的时候只回答一个问题：
    **rerank 排上来的东西，如果是我搜的，我会不会点？** 会 → 是尺子量不了，不是模型差。
    """
    want = {r["query_id"]: r for r in score_rows[:n] if r.get("intent")}
    texts: dict[int, list[dict]] = {}
    with cand_path.open(encoding="utf-8") as f:
        for line in f:  # 177MB，流式扫，只留要看的那几条
            row = json.loads(line)
            if row["query_id"] in want:
                texts[row["query_id"]] = row["candidates"]
            if len(texts) == len(want):
                break

    for qid, rec in want.items():
        row, cands = qrels.get(qid), texts.get(qid)
        if row is None or cands is None:
            continue
        g, pos, sub, comp = gain_fn(row)
        labeled = pos | sub | comp

        def tag(item_id: str, pos: set = pos, labeled: set = labeled) -> str:
            return "★" if item_id in pos else ("·" if item_id in labeled else " ")

        scores = rec["intent"]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        print(f"\n{'=' * 96}\nQ{qid}  {row['query']}   （★=标注正例 ·=其他标注 空=未标注）")
        print("-- e15 原序 top-5 " + "-" * 78)
        for c in cands[:5]:
            print(f"  {tag(c['item_id'])} {c['text'][:86]}")
        print("-- intent rerank top-5 " + "-" * 73)
        for i in order:
            print(f"  {tag(cands[i]['item_id'])} {cands[i]['text'][:80]}  [{scores[i]:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    default_scores = PROJECT_ROOT / "data" / "train" / "rerank_scores.jsonl"
    ap.add_argument("--scores", default=str(default_scores))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--dump-cases", type=int, default=0, help="并排打印前 N 条排序对比（人眼验货）")
    ap.add_argument(
        "--candidates", default=str(PROJECT_ROOT / "data" / "train" / "rerank_candidates.jsonl")
    )
    args = ap.parse_args()

    qrels = {
        r["query_id"]: r
        for r in (json.loads(x) for x in QRELS_PATH.open(encoding="utf-8") if x.strip())
    }
    score_rows = [json.loads(x) for x in Path(args.scores).open(encoding="utf-8") if x.strip()]
    print(f"评测 query：{len(score_rows)}（qrels {len(qrels)} 条）")

    report = run(qrels, score_rows)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(report)
    print(f"\n已写 {args.out}")
    if args.dump_cases:
        dump_cases(qrels, score_rows, Path(args.candidates), args.dump_cases)


if __name__ == "__main__":
    main()
