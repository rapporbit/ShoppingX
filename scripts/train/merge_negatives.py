"""把过完闸的 ANN 难负例并进训练集，替换掉大部分随机负例。

原始数据集 97.5% 的样本掺了随机负例，人工难负例只占 16.4%——refdocs 04-2 §3.2 说 hard negative
决定模型上限，这个配比撑不起上限。本脚本按优先级重排每条 query 的负例：

    人工 S（可替代）→ 人工 C（配件）→ ANN 挖的（过完假负闸）→ 人工 I（无关）→ 随机兜底

ANN 负例取 ``ann_rank`` 最靠前的（语义最像却被判定不相关的，信号最强）。

**文本口径**：ANN 候选落盘时存的是 reranker 口径（标题+品牌+品类，小写），那是给闸用的；训练集
必须用 ``embed_text``（与线上建索引逐字同源），所以这里按 item_id 回库重取，不复用那份文本。

用法：``uv run --group train python scripts/train/merge_negatives.py [--max-neg 8]``
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.recall.text import embed_text  # noqa: E402
from app.utils.clean import CleanItem  # noqa: E402

CLEAN_DIR = PROJECT_ROOT / "data" / "platforms" / "clean" / "by_platform"
TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "esci_train.jsonl"
SCORED_PATH = PROJECT_ROOT / "data" / "train" / "neg_ann_scored.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "train" / "merge_report.json"


def load_all_items() -> dict[str, str]:
    """**全平台**商品文本（不只 amazon）。

    ANN 是 ``platform="all"`` 挖的，和线上检索口径一致，所以候选里会有 lazada / shopee 等平台的
    商品——它们同样是合法的负例。只加载 amazon 会在这里 KeyError（踩过）。
    """
    texts: dict[str, str] = {}
    for path in sorted(CLEAN_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    item = CleanItem(**json.loads(line))
                except Exception:
                    continue
                text = embed_text(item)
                if text:
                    texts[item.item_id] = text
        print(f"  读 {path.name}: 累计 {len(texts)}")
    return texts


def _take(
    iid: str,
    src: str,
    *,
    picked: list[tuple[str, str]],
    banned: set[str],
    seen_texts: set[str],
    texts: dict[str, str],
    max_neg: int,
) -> bool:
    """收一条负例；槽位满 / id 重复 / **文本重复** 任一命中就拒收。"""
    if len(picked) >= max_neg:
        return False
    text = texts.get(iid)
    if not text or iid in banned or text in seen_texts:
        return False
    picked.append((iid, src))
    banned.add(iid)
    seen_texts.add(text)
    return True


def load_ann(threshold: float) -> dict[int, list[str]]:
    """query_id → 过闸后的候选 item_id（按 ann_rank 升序，即越像越靠前）。"""
    out: dict[int, list[str]] = {}
    with SCORED_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            kept = [
                c
                for c in row["candidates"]
                # 分数全落了盘，阈值在这里重判——不依赖打分时写死的 verdict，便于做消融
                if c.get("rerank_score", 1.0) < threshold
            ]
            kept.sort(key=lambda c: c["ann_rank"])
            out[row["query_id"]] = [c["item_id"] for c in kept]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-neg", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="假负闸阈值（重判用）")
    ap.add_argument("--ann-per-query", type=int, default=8, help="每条 query 最多取几条 ANN 负例")
    ap.add_argument("--out", default="esci_train_v2.jsonl", help="输出文件名（便于做配方消融）")
    args = ap.parse_args()

    out_path = PROJECT_ROOT / "data" / "train" / args.out

    print("读商品文本…")
    texts = load_all_items()
    print(f"商品文本 {len(texts)}\n")
    ann = load_ann(args.threshold)
    print(f"ANN 过闸候选覆盖 query：{len(ann)}")

    rows = [json.loads(x) for x in TRAIN_PATH.open(encoding="utf-8") if x.strip()]
    pool = list(texts)
    rng = random.Random(20260807)
    src_count: Counter[str] = Counter()
    n_out = 0

    with out_path.open("w", encoding="utf-8") as out:
        for r in rows:
            picked: list[tuple[str, str]] = []
            banned = set(r["pos_ids"])
            # 按**文本**去重，不只按 item_id：库里有同款不同 ASIN（颜色/尺码变体），它们的
            # embed_text 逐字相同。只按 id 去重的话，同一段文本会既当正例又当负例——模型收到
            # 的是自相矛盾的梯度，比噪声更毒。文本相同是比 refdocs §3.3 的图片 pHash 更精确的
            # 同款判据：模型看到的就是这段文本，一样就是一样。
            seen_texts = {texts[i] for i in r["pos_ids"] if i in texts}
            take = partial(
                _take,
                picked=picked,
                banned=banned,
                seen_texts=seen_texts,
                texts=texts,
                max_neg=args.max_neg,
            )

            # 1) 人工标注的难负例（原样保留，这批不过闸——人工标过的最可信）
            for iid, src in zip(r["neg_ids"], r["neg_src"], strict=True):
                if src in ("esci_S", "esci_C"):
                    take(iid, src)
            # 2) ANN 挖的（过完闸），取语义最像的前几条
            n_ann = 0
            for iid in ann.get(r["query_id"], []):
                if len(picked) >= args.max_neg or n_ann >= args.ann_per_query:
                    break
                n_ann += take(iid, "ann")
            # 3) 人工无关项
            for iid, src in zip(r["neg_ids"], r["neg_src"], strict=True):
                if src == "esci_I":
                    take(iid, src)
            # 4) 随机兜底（前面都填不满才用）
            while len(picked) < args.max_neg:
                take(pool[rng.randrange(len(pool))], "random")

            src_count.update(s for _, s in picked)
            neg_ids = [i for i, _ in picked]
            out.write(
                json.dumps(
                    {
                        "query": r["query"],
                        "query_id": r["query_id"],
                        "pos": r["pos"],
                        "neg": [texts[i] for i in neg_ids],
                        "pos_ids": r["pos_ids"],
                        "neg_ids": neg_ids,
                        "neg_src": [s for _, s in picked],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1

    total = sum(src_count.values())
    hard = src_count["esci_S"] + src_count["esci_C"] + src_count["ann"]
    report = {
        "rows": n_out,
        "negatives_total": total,
        "by_source": dict(src_count),
        "hard_ratio": round(hard / max(total, 1), 4),
        "random_ratio": round(src_count["random"] / max(total, 1), 4),
        "threshold": args.threshold,
        "ann_per_query": args.ann_per_query,
        "out": args.out,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 结果 ===")
    for k, v in report.items():
        print(f"{k:18} {v}")
    print(f"已写 {out_path}")


if __name__ == "__main__":
    main()
