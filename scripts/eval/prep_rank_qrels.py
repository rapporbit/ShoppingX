"""picker 排序评测 · 第 0 步：选题 + 标签（封闭池口径），产 ``qrels.jsonl``。

**为什么是封闭池**：用原始 query 跑线上召回（池 30），ESCI 池内已判件中位 1/30、好坏都有的只 10%，
在真实池上算 top-3 指标会被未判定件淹没。所以池 = 库内已判件
（同 ESCI Task 1 / TREC CatB 重排口径）。
代价：池里没有真实召回带进来的蹭词货，那部分由线上 badcase 池回归补。

数据源（两个都用 ASIN 与主库 ``amazon_rag.jsonl`` 的 item_id 对齐）：

- ESCI us：``data/train/_esci_examples.parquet``，E/S/C/I；
  可用 = 库内 ≥3 件已判且含 E 与 I|C，随机抽 N 条。
- TREC Product Search 2023/2024/2025：NIST pooled 0~3（-1 丢弃）；
  可用 = 库内含 ≥2 档与 ≤1 档；全取。
  qrels 用整数 docid，经 ``asin2trecid.pickle`` 转 ASIN ——
  **用禁止 find_class 的受限 Unpickler 读**。

**标签降噪**：同一 ASIN 在 ESCI/TREC 目录与我们库里可能已是不同标题
（卖家改标题、换规格、偶有换物），精排看的是我们的标题、标签却是对着当年标题标的。
流式扫 TREC 语料取标题，token Jaccard<0.2 的标注件剔除，
剔除件数写进报告（这是删标签噪声，不是删系统缺陷）。

用法（只要 pandas/pyarrow，不进项目环境，避免 uv 隐式 sync 卸 extra）::

    uv run --no-project --with pandas --with pyarrow \
        python scripts/eval/prep_rank_qrels.py --esci-n 300
"""

import argparse
import gzip
import json
import pickle
import random
import re
import subprocess
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval/picker_replay"
RAW = OUT / "raw"
CORPUS = ROOT / "data/platforms/clean/by_platform/amazon_rag.jsonl"
HF = "https://huggingface.co/datasets/trec-product-search"
NIST = "https://trec.nist.gov/data/product"
FILES = {
    "asin2trecid.pickle": f"{HF}/product-search-corpus/resolve/main/asin2trecid.pickle",
    "t23.qrel": f"{HF}/product-search-2023-qrels/resolve/main/2023test.qrel",
    "t23q.tsv.gz": f"{HF}/product-search-2023-queries/resolve/main/2023_test_queries.tsv.gz",
    "t24.qrel": f"{NIST}/2024-qrels.txt",
    "t24q.tsv": f"{NIST}/2024-test-queries.tsv",  # HF 那份 query 编号与 qrels 对不上，用 NIST 的
    "t25.qrel": f"{NIST}/2025-qrels.search",
    "t25q.tsv": f"{HF}/product-search-2025-test-queries/resolve/main/"
    "product-search-2025-test-queries.tsv",
}
TREC_CORPUS = f"{HF}/product-search-corpus/resolve/main/corpus-simple.jsonl.gz"
DRIFT_MIN_JACCARD = 0.2


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> None:  # 网上下的 pickle 只许纯数据
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def _fetch(name: str) -> Path:
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FILES[name], p)
    return p


def _queries(name: str, gz: bool = False, header: bool = False) -> dict[str, str]:
    p = _fetch(name)
    text = gzip.open(p, "rt").read() if gz else p.read_text()
    lines = text.splitlines()[1 if header else 0 :]
    return {a: b.strip() for a, b in (ln.split("\t", 1) for ln in lines if "\t" in ln)}


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _usable(labels: dict, good: set, bad: set, min_n: int) -> bool:
    v = list(labels.values())
    return len(v) >= min_n and any(x in good for x in v) and any(x in bad for x in v)


def _our_titles() -> dict[str, str]:
    out = {}
    with CORPUS.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["item_id"]] = d.get("title") or ""
    return out


def _esci(ours: dict[str, str], n: int, rng: random.Random) -> list[dict]:
    import pandas as pd

    cols = ["query_id", "query", "product_id", "product_locale", "esci_label"]
    df = pd.read_parquet(ROOT / "data/train/_esci_examples.parquet", columns=cols)
    us = df[(df.product_locale == "us") & df.product_id.isin(ours.keys())]
    rows = []
    for qid, g in us.groupby("query_id"):
        labels = dict(zip(g.product_id, g.esci_label, strict=True))
        if _usable(labels, {"E"}, {"I", "C"}, 3):
            rows.append(
                {"set": "esci", "qid": str(qid), "query": g["query"].iloc[0], "labels": labels}
            )
    return rng.sample(rows, min(n, len(rows)))


def _trec(ours: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    a2t = _NoGlobals(_fetch("asin2trecid.pickle").open("rb")).load()
    t2a = {str(v): k for k, v in a2t.items()}
    specs = [
        ("trec23", "t23.qrel", _queries("t23q.tsv.gz", gz=True)),
        ("trec24", "t24.qrel", _queries("t24q.tsv", header=True)),
        ("trec25", "t25.qrel", _queries("t25q.tsv", header=True)),
    ]
    rows = []
    for name, qrel, qs in specs:
        lab: dict[str, dict[str, int]] = defaultdict(dict)
        for line in _fetch(qrel).read_text().splitlines():
            a = line.split()
            if len(a) >= 4 and int(a[3]) >= 0 and (asin := t2a.get(a[2], a[2])) in ours:
                lab[a[0]][asin] = int(a[3])
        for qid, labels in lab.items():
            if qid in qs and _usable(labels, {2, 3}, {0, 1}, 2):
                rows.append({"set": name, "qid": qid, "query": qs[qid], "labels": labels})
    return rows, {v: k for k, v in t2a.items()}


def _trec_titles(need: set[str], a2t: dict[str, str]) -> dict[str, str]:
    """流式扫 TREC 语料（570MB gz，不落盘），只留需要的 ASIN 标题。"""
    want = {a2t[a]: a for a in need if a in a2t}
    proc = subprocess.Popen(["curl", "-sSL", TREC_CORPUS], stdout=subprocess.PIPE)
    out = {}
    for line in gzip.open(proc.stdout, "rt"):
        d = json.loads(line)
        if (a := want.get(str(d["docid"]))) is not None:
            out[a] = d.get("title") or ""
    proc.wait()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esci-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ours = _our_titles()
    rows = _esci(ours, args.esci_n, rng)
    trec_rows, asin2trec = _trec(ours)
    rows += trec_rows
    theirs = _trec_titles({a for r in rows for a in r["labels"]}, asin2trec)
    report: Counter = Counter()
    kept = []
    for r in rows:
        for a in list(r["labels"]):
            if a not in theirs:
                report[r["set"], "no_title_kept"] += 1  # 拿不到对照标题：保留，计数
                continue
            A, B = _tok(theirs[a]), _tok(ours[a])
            if len(A & B) / max(1, len(A | B)) < DRIFT_MIN_JACCARD:
                del r["labels"][a]
                report[r["set"], "drift_dropped"] += 1
        good, bad, n = ({"E"}, {"I", "C"}, 3) if r["set"] == "esci" else ({2, 3}, {0, 1}, 2)
        if _usable(r["labels"], good, bad, n):
            kept.append(r)
        else:
            report[r["set"], "query_dropped"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "qrels.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {f"{s}.{k}": v for (s, k), v in sorted(report.items())}
    summary.update({f"{s}.queries": v for s, v in Counter(r["set"] for r in kept).items()})
    (OUT / "prep_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
