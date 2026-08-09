"""组装 reranker 训练集：相对闸过滤假负 + 正例展开 → ms-swift reranker 格式。

**闸为什么是「相对」的。** M21 给 embedding 挖负例时用的是绝对阈值 0.5，实测闸掉率 52.24%。
在这份数据上不能照搬——同一个 cross-encoder 的标定分布是：人工正例 E 中位仅 .5540、p10 低到
.0027，而人工负例 S 的 p90 到 .6679、C 到 .8927。**E 的 p10 比 I 的 p90 还低，尺子本身就是
倒挂的**。拍任何一个绝对阈值，都会在闸掉疑似假负的同时，把大量人工确认的 S/C 一并闸掉——
而那恰恰是最贵的 hard negative。（同一个教训在本项目已经吃过一次：槽位 rerank 的绝对阈值门
被真实标定证伪、最后默认关掉。）

改成 query 内相对判据：**候选分数高于「该 query 已知正例分数的某个分位」才判疑似假负**。
每条 query 自带参照物，模型对该 query 整体打分高低就被约掉了。

``--gate-quantile`` 默认 **1.0（取正例最高分）**，即只剔除「比所有已知正例都更像」的候选。
刻意保守：我们要治的病是「深池里捞不出正例」，训练数据宁可留一点噪声，也不能把 hard negative
删光——M21 闸掉 52% 之后剩下的负例偏易，正是这次要避免的。闸掉率会打印出来，用数据说话。

**ESCI 标注的 S/C/I 一律不过闸**：人工标注优先于模型判据，理由同 ``mine_deep_negatives.py``。

**正例展开** 沿用 M21 的唯一有效杠杆（每 query 平均 3.94 个正例，只取第一个等于扔掉 74% 的
人工标注）。reranker 这里比 embedding 更安全——listwise loss 按组独立算 CE，没有 in-batch
negatives，同一 query 的多条样本落进同一 batch 也不会互相当假负。shuffle 照做，图个稳。

用法::

    uv run --group train python scripts/train/build_rerank_train.py \\
        --scores data/train/deep_neg_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "train"
CAND_PATH = DATA_DIR / "deep_neg_candidates.jsonl"

SEED = 42
VAL_RATIO = 0.02
# refdocs §10.2 的 4 档相关性标注，对上 ESCI 的人工标签
ESCI_GAIN = {"esci_S": 2, "esci_C": 1, "esci_I": 0}


def quantile(values: list[float], q: float) -> float:
    """线性插值分位（q=1 取最大、0 取最小）。样本量小到个位数，不值得引 numpy。"""
    if not values:
        return 1.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def to_row(query: str, pos_text: str, negs: list[str]) -> dict:
    """ms-swift reranker/embedding 共用的三段式（同 to_swift_format.py）。"""
    return {
        "messages": [{"role": "user", "content": query}],
        "positive_messages": [[{"role": "assistant", "content": pos_text}]],
        "negative_messages": [[{"role": "assistant", "content": n}] for n in negs],
    }


def to_graded_row(query: str, pos_text: str, negs: list[tuple[str, int]]) -> dict:
    """分级格式：一组 = query + 一列 doc + 一列 gain（3/2/1/0）。

    ms-swift 的 reranker 模板只吃 positive/negative 二值，表达不了 refdocs §10.2 的四档，
    所以分级训练走自写的 ``train_reranker_graded.py``，格式也就不必迁就 swift 的三段式。
    """
    return {
        "query": query,
        "docs": [pos_text] + [t for t, _ in negs],
        "labels": [3] + [g for _, g in negs],
    }


def build(
    cands: list[dict], scores: dict[int, dict], args: argparse.Namespace
) -> tuple[list, dict]:
    rng = random.Random(SEED)
    out: list[dict] = []
    stats = {
        "queries": 0,
        "ann_kept": 0,
        "ann_gated": 0,
        "esci_kept": 0,
        "no_score": 0,
        # 组内最高负例档位的分布——它直接决定分级训练能学到多少「相关但不是最优」的信号。
        # 全 0 就是「这组里除了正例全是不相关」，ApproxNDCG 在这种组上退化成 MRR。
        "grade_0": 0,
        "grade_1": 0,
        "grade_2": 0,
    }

    for row in cands:
        sc = scores.get(row["query_id"])
        if sc is None:
            stats["no_score"] += 1
            continue
        by_id = dict(zip(sc["item_ids"], sc["intent"], strict=True))
        bar = quantile(sc.get("pos_scores") or [], args.gate_quantile)

        # 分开收集：ESCI 人工负例优先占坑，ANN 负例按 rank 层轮询补位。
        # **不能收集完再 shuffle 截断**——那样 S/C 这些最贵的 hard negative 会被随机丢掉，
        # 而它们正是「可替代品 / 互补配件」这类模型最容易判错的样本。
        # 负例带档位一起收：(文本, gain)。gain 即 refdocs §10.2 的 4 档相关性——
        # E=3 / S=2（可替代但不是要的那个）/ C=1（互补配件）/ I 与 ANN 挖的=0。
        # 二值格式下 gain 只用来排优先级，分级格式下它直接进 loss。
        esci_negs: list[tuple[str, int]] = []
        ann_layers: dict[str, list[tuple[str, int]]] = {}
        for c in row["candidates"]:
            if c["source"].startswith("esci"):  # 人工标注的负例，不过闸
                esci_negs.append((c["text"], ESCI_GAIN.get(c["source"], 0)))
                stats["esci_kept"] += 1
                continue
            s = by_id.get(c["item_id"])
            if s is not None and s > bar:
                stats["ann_gated"] += 1  # 比该 query 所有已知正例都更像 → 疑似未标注的真正例
                continue
            ann_layers.setdefault(c["source"], []).append((c["text"], 0))
            stats["ann_kept"] += 1

        # ESCI 档位高的先占坑：分级训练里 S/C 是唯一能教「相关但不是最优」的样本，
        # 二值训练里它们也是最像正例的 hard negative，两种格式下都该优先。
        esci_negs.sort(key=lambda t: -t[1])
        negs = esci_negs[: args.max_neg]
        # 轮询各 rank 层，保证浅/中/深三段都有代表——只喂浅层就退化成 M21 那种「近义干扰」数据集
        pools = [rng.sample(v, len(v)) for v in ann_layers.values()]
        while len(negs) < args.max_neg and any(pools):
            for p in pools:
                if p and len(negs) < args.max_neg:
                    negs.append(p.pop())
        if len(negs) < args.min_neg:
            continue
        stats[f"grade_{max((g for _, g in negs), default=0)}"] += 1

        seen: set[str] = set()
        for p in row["pos"][: args.max_pos]:  # 正例展开
            if p["text"] in seen:
                continue
            seen.add(p["text"])
            out.append(
                to_graded_row(row["query"], p["text"], negs)
                if args.format == "graded"
                else to_row(row["query"], p["text"], [t for t, _ in negs])
            )
        stats["queries"] += 1

    rng.shuffle(out)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DATA_DIR / "deep_neg_scores.jsonl"))
    ap.add_argument("--candidates", default=str(CAND_PATH))
    ap.add_argument("--out-prefix", default="swift_r1")
    ap.add_argument("--gate-quantile", type=float, default=1.0, help="1=正例最高分 0.5=中位 0=最低")
    ap.add_argument("--max-pos", type=int, default=3, help="每 query 展开的正例数上限")
    ap.add_argument("--max-neg", type=int, default=15)
    ap.add_argument("--min-neg", type=int, default=4, help="负例太少的组，listwise CE 学不到东西")
    ap.add_argument(
        "--format",
        choices=("swift", "graded"),
        default="swift",
        help="swift=ms-swift 三段式(二值)，graded=自写训练脚本用的四档分级",
    )
    args = ap.parse_args()

    cands = [json.loads(x) for x in Path(args.candidates).open(encoding="utf-8") if x.strip()]
    scores = {
        r["query_id"]: r
        for r in (json.loads(x) for x in Path(args.scores).open(encoding="utf-8") if x.strip())
    }
    print(f"候选 query {len(cands)}，打分 query {len(scores)}")

    rows, stats = build(cands, scores, args)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val, train = rows[:n_val], rows[n_val:]

    for name, part in (("train", train), ("val", val)):
        path = DATA_DIR / f"{args.out_prefix}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path.name}: {len(part)} 条")

    total_ann = stats["ann_kept"] + stats["ann_gated"]
    rate = stats["ann_gated"] / total_ann if total_ann else 0.0
    print(f"\n{stats}")
    print(f"相对闸掉率 {rate:.2%}（q={args.gate_quantile}）—— M21 的绝对阈值闸是 52.24%，可对照")


if __name__ == "__main__":
    main()
