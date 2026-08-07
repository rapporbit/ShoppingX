"""用 ESCI 人工标注 × 本仓库商品库，产出 embedding / reranker 的微调数据与评测金标。

探针（``probe_esci_overlap.py``）确认交集足够大：29.7 万条标注落在我们库内，5.36 万条 query
至少有一个 Exact 正例。本脚本把这些标注翻译成训练三元组。

**口径（对齐 refdocs 04-2 §3.2，按真实标签做了取舍）：**

- 正例 = ``E`` Exact。
- 难负例 = ``S`` Substitute（可替代但不是要的那个）+ ``C`` Complement（互补配件）。
  把 C 算负例是刻意的：搜索相关性口径下，「手机壳」不是「手机」这个 query 的答案。
  我们那个「搜手机出配件」的老 bad case，此前查下来是数据问题、黑名单与品类过滤两方案
  都被实测证伪；这 8 千多条人工标注是目前手上最对症的信号。
- 易负例 = ``I`` Irrelevant，只在难负例不够时补位。
- **不做 refdocs §3.3 的假负样本过滤**（同 SKU join / 图片 pHash）。那一步是为「从日志或
  ANN 自动构造负例」准备的——自动构造分不清同款。ESCI 的 E/S/C/I 是人工逐对标的，同款不会
  被标成 S，这条路天然免疫，省掉整套 pHash 管线。

**商品文本用 ``app.recall.text.embed_text``**，与线上建索引逐字同源，避免 train/serve skew。

**只取 ``us`` locale**：es/jp 命中各只有几千行，撑不起跨语言训练；中文腿另走 query 改写。

用法：``uv run --group train python scripts/train/build_esci_pairs.py [--max-neg 8]``
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.recall.text import embed_text  # noqa: E402
from app.utils.clean import CleanItem  # noqa: E402

ESCI_PARQUET = PROJECT_ROOT / "data" / "train" / "_esci_examples.parquet"
CLEAN_DIR = PROJECT_ROOT / "data" / "platforms" / "clean" / "by_platform"
OUT_DIR = PROJECT_ROOT / "data" / "train"

LOCALE = "us"
POS_LABELS = {"E"}
HARD_NEG_LABELS = {"S", "C"}
EASY_NEG_LABELS = {"I"}


class QueryGroup:
    """一条 query 在我们库内命中的商品，按用途分好组。

    ``sub``(Substitute) 与 ``comp``(Complement) 分开存而不是合成一个 hard_neg：训练时两者都当
    难负例用，但**评测时只有 comp 能用来量「整机 query 召回了多少配件」**——那正是「搜手机出
    配件」这个 bad case 的度量。混在一起就做不了专项评测。
    """

    __slots__ = ("query", "split", "pos", "sub", "comp", "easy_neg")

    def __init__(self, query: str, split: str) -> None:
        self.query = query
        self.split = split
        self.pos: list[str] = []
        self.sub: list[str] = []
        self.comp: list[str] = []
        self.easy_neg: list[str] = []

    @property
    def hard_neg(self) -> list[str]:
        """训练用的难负例：可替代品在前（信号更强），配件在后。"""
        return [*self.sub, *self.comp]

    def add(self, asin: str, label: str) -> None:
        if label in POS_LABELS:
            self.pos.append(asin)
        elif label == "S":
            self.sub.append(asin)
        elif label == "C":
            self.comp.append(asin)
        elif label in EASY_NEG_LABELS:
            self.easy_neg.append(asin)


def load_our_items() -> dict[str, str]:
    """本仓库 amazon 商品：ASIN → 与线上同源的编码文本。"""
    texts: dict[str, str] = {}
    for path in sorted(CLEAN_DIR.glob("amazon*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("platform") != "amazon":
                    continue
                try:
                    item = CleanItem(**rec)
                except Exception:  # 脏行不该拖垮整条管线
                    continue
                text = embed_text(item)
                if text:
                    texts[item.item_id] = text
        print(f"  读 {path.name}: 累计 {len(texts)} 条商品文本")
    return texts


def load_groups(known_asins: set[str]) -> dict[int, QueryGroup]:
    """读 ESCI 标注，只留 locale=us 且商品在我们库内的行，按 query 归组。"""
    cols = pq.read_table(ESCI_PARQUET).to_pydict()
    groups: dict[int, QueryGroup] = {}
    kept = 0
    for qid, query, asin, locale, label, split in zip(
        cols["query_id"],
        cols["query"],
        cols["product_id"],
        cols["product_locale"],
        cols["esci_label"],
        cols["split"],
        strict=True,
    ):
        if locale != LOCALE or asin not in known_asins:
            continue
        query = query.strip()  # 原始标注里有 " revent 80 cfm" 这种带前导空格的，会扰动分词
        if not query:
            continue
        group = groups.get(qid)
        if group is None:
            group = groups[qid] = QueryGroup(query, split)
        group.add(asin, label)
        kept += 1
    print(f"ESCI 命中行（us + 库内）：{kept}，涉及 query：{len(groups)}")
    return groups


def write_dataset(
    groups: dict[int, QueryGroup], texts: dict[str, str], max_neg: int, random_neg: int
) -> dict[str, object]:
    """落两份产物：训练三元组（train split）+ 评测金标（test split）。

    ``random_neg``：负例不足时从全库随机补几条（refdocs 04-2 §5.5.4 认可的「跨类目随机负样本」）。
    大量 query 在我们库内只命中了正例——ESCI 标注的 S/C/I 商品恰好不在库里。不补的话这些 query
    全被丢掉，训练集从 4 万缩到 1.5 万。随机负例有极低概率抽中真相关商品（137 万里抽），代价可接受，
    且每条样本记了 ``n_random_neg`` 便于事后做消融。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "esci_train.jsonl"
    qrels_path = OUT_DIR / "esci_eval_qrels.jsonl"
    pool = list(texts)
    rng = random.Random(20260807)  # 固定 seed：数据集可复现

    n_train = n_eval = 0
    neg_hard_used = neg_easy_used = neg_random_used = 0

    with (
        train_path.open("w", encoding="utf-8") as ftrain,
        qrels_path.open("w", encoding="utf-8") as fqrels,
    ):
        for qid, g in groups.items():
            if not g.pos:  # 没有正例的 query 训不了也评不了
                continue
            if g.split == "test":
                # 评测只需 id：候选池是**全库 137 万**，不是这几条命中商品——否则 Recall 虚高。
                # substitutes / complements 分开落：前者量「召回没召准」，后者量「整机搜出配件」。
                fqrels.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query": g.query,
                            "positives": g.pos,
                            "substitutes": g.sub,
                            "complements": g.comp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_eval += 1
                continue
            # 负例带来源标签：后续掺进 ann / bm25 挖的负例后，能按来源做消融
            picked: list[tuple[str, str]] = [(a, "esci_S") for a in g.sub[:max_neg]]
            picked += [(a, "esci_C") for a in g.comp[: max_neg - len(picked)]]
            neg_hard_used += len(picked)
            if len(picked) < max_neg:
                fill = [(a, "esci_I") for a in g.easy_neg[: max_neg - len(picked)]]
                picked += fill
                neg_easy_used += len(fill)
            n_rand = 0
            if len(picked) < max_neg and random_neg > 0:
                need = min(random_neg, max_neg - len(picked))
                banned = set(g.pos) | {a for a, _ in picked}
                while n_rand < need:
                    cand = pool[rng.randrange(len(pool))]
                    if cand in banned:
                        continue
                    picked.append((cand, "random"))
                    banned.add(cand)
                    n_rand += 1
                neg_random_used += n_rand
            if not picked:
                continue
            neg_ids = [a for a, _ in picked]
            ftrain.write(
                json.dumps(
                    {
                        "query": g.query,
                        "query_id": qid,
                        "pos": [texts[a] for a in g.pos],
                        "neg": [texts[a] for a in neg_ids],
                        "pos_ids": g.pos,
                        "neg_ids": neg_ids,
                        "neg_src": [s for _, s in picked],
                        "n_random_neg": n_rand,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_train += 1

    return {
        "train_rows": n_train,
        "eval_rows": n_eval,
        "neg_hard_used": neg_hard_used,
        "neg_easy_used": neg_easy_used,
        "neg_random_used": neg_random_used,
        "max_neg": max_neg,
        "random_neg": random_neg,
        "train_path": str(train_path.relative_to(PROJECT_ROOT)),
        "qrels_path": str(qrels_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-neg", type=int, default=8, help="每条 query 的负例上限")
    ap.add_argument("--random-neg", type=int, default=4, help="负例不足时随机补几条（0=不补）")
    args = ap.parse_args()

    texts = load_our_items()
    print(f"本仓库 amazon 商品文本：{len(texts)}\n")
    groups = load_groups(set(texts))
    report = write_dataset(groups, texts, args.max_neg, args.random_neg)

    (OUT_DIR / "esci_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 结果 ===")
    for k, v in report.items():
        print(f"{k:16} {v}")


if __name__ == "__main__":
    main()
