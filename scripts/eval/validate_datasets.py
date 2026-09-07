"""评测数据集体检：跑分之前先确认「尺子本身没坏」。不达标退非零，可挂 CI。

跑分脚本（`run_product_recall.py` / `run_rubric.py` / `run_category_recall.py`）都假设数据集
是好的：字段齐、id 不撞、正例非空。这些假设一旦破，**症状不是报错而是分数变化**——recall 掉了
分不清是召回退化还是 golden 少了一半，rubric 汇总里两条同 id 的 case 会静默互相覆盖。所以这类
检查必须是独立一道闸，在跑分之前跑，几秒钟出结果。

三个数据集，各自的判据：

- **商品 golden**（ESCI qrels，M21 产物）：query 非空、正例非空、id 不撞、正/替/配三档不重叠
  （同一个商品既是正例又是配件，NDCG 的 gain 就取决于判定顺序，指标不可复现）。
- **Rubric 种子集**（`data/eval/queries.jsonl`）：id 不撞、必填字段齐、多轮 case 的 `query`
  必须等于 `turns[-1]`（run_rubric 只对最后一轮打分，对不上就是「拿 A 的答案对着 B 的尺子打」）。
  另**校验它与 `build_eval_queries.py` 同步**——种子集是回归基线，基线必须能从脚本复现
  （批 1 踩过：只改 jsonl，下次重建脚本就把改动冲掉了）。
- **品类金标**（`data/eval/category_recall.jsonl`）：query 非空、relevant 非空、id 不撞。

用法：
    uv run python scripts/eval/validate_datasets.py            # 全查
    uv run python scripts/eval/validate_datasets.py --only seeds
退出码：全过 0 / 有问题 1（缺文件算问题，但 `--allow-missing` 可放行 gitignore 的那几份）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

QRELS_PATH = PROJECT_ROOT / "data" / "train" / "esci_eval_qrels.jsonl"
SEEDS_PATH = PROJECT_ROOT / "data" / "eval" / "queries.jsonl"
CATEGORY_PATH = PROJECT_ROOT / "data" / "eval" / "category_recall.jsonl"

# 种子集每条必须有的字段。`turns` / `constraints` / `prior_context` 是选填（分别只对多轮、
# 有硬约束、需要跨会话事实的 case 才有意义）。
SEED_REQUIRED = ("id", "bucket", "intent", "query")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _dup_ids(ids: list) -> list:
    seen, dup = set(), []
    for i in ids:
        if i in seen and i not in dup:
            dup.append(i)
        seen.add(i)
    return dup


def check_qrels(rows: list[dict]) -> list[str]:
    errs: list[str] = []
    if dup := _dup_ids([r.get("query_id") for r in rows]):
        errs.append(f"query_id 重复 {len(dup)} 个，前 5：{dup[:5]}")
    for r in rows:
        qid = r.get("query_id", "?")
        if not str(r.get("query") or "").strip():
            errs.append(f"query_id={qid}：query 为空")
        pos = set(r.get("positives") or [])
        if not pos:
            errs.append(f"query_id={qid}：positives 为空（量不出 recall）")
        sub, comp = set(r.get("substitutes") or []), set(r.get("complements") or [])
        # 档位重叠 → gain 取决于判定顺序，同一份数据能算出两个 NDCG，指标不可复现。
        pairs = ((pos, sub, "正例/替代"), (pos, comp, "正例/配件"), (sub, comp, "替代/配件"))
        for a, b, name in pairs:
            if overlap := a & b:
                errs.append(f"query_id={qid}：{name} 标注重叠 {sorted(overlap)[:3]}")
    return errs[:50]  # 同族错误刷屏没意义，留前 50 条定位用


def check_seeds(rows: list[dict]) -> list[str]:
    errs: list[str] = []
    if dup := _dup_ids([r.get("id") for r in rows]):
        errs.append(f"id 重复：{dup}（rubric 汇总会静默互相覆盖）")
    for r in rows:
        rid = r.get("id", "?")
        for field in SEED_REQUIRED:
            if not str(r.get(field) or "").strip():
                errs.append(f"{rid}：缺字段 {field}")
        turns = r.get("turns")
        if turns is not None:
            if not isinstance(turns, list) or not turns:
                errs.append(f"{rid}：turns 不是非空列表")
            elif turns[-1] != r.get("query"):
                errs.append(f"{rid}：query 与 turns[-1] 不一致（打分打的是最后一轮）")
        prior = r.get("prior_context")
        if prior is not None and not str(prior).strip():
            errs.append(f"{rid}：prior_context 是空串（要么写实，要么别写这个键）")
    return errs


def check_seeds_in_sync(rows: list[dict]) -> list[str]:
    """种子集必须与生成脚本一致——它是回归基线，基线只能从脚本复现。"""
    from scripts.eval.build_eval_queries import QUERIES

    on_disk = {r["id"] for r in rows if r.get("id")}
    in_script = {q["id"] for q in QUERIES}
    errs = []
    if missing := sorted(in_script - on_disk):
        errs.append(f"脚本里有、jsonl 里没有（该重跑 build_eval_queries.py）：{missing}")
    if extra := sorted(on_disk - in_script):
        errs.append(f"jsonl 里有、脚本里没有（改动只写了产物，下次重建会被冲掉）：{extra}")
    by_id = {q["id"]: q for q in QUERIES}
    for r in rows:
        src = by_id.get(r.get("id"))
        if src and src.get("query") != r.get("query"):
            errs.append(f"{r['id']}：query 与脚本不一致")
    return errs


def check_category(rows: list[dict]) -> list[str]:
    errs: list[str] = []
    if dup := _dup_ids([r.get("query") for r in rows]):
        errs.append(f"query 重复：{dup[:5]}")
    for r in rows:
        if not str(r.get("query") or "").strip():
            errs.append("存在空 query")
        if not (r.get("relevant") or []):
            errs.append(f"{r.get('query', '?')[:30]}：relevant 为空")
    return errs[:50]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=("qrels", "seeds", "category"),
        default=None,
        help="只查一个数据集（默认全查）",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="数据集文件缺失时只警告不失败（data/ 在 gitignore 里，靠 scripts/ 复现）",
    )
    args = ap.parse_args()

    plan = [
        ("qrels", QRELS_PATH, check_qrels, None),
        ("seeds", SEEDS_PATH, check_seeds, check_seeds_in_sync),
        ("category", CATEGORY_PATH, check_category, None),
    ]
    if args.only:
        plan = [p for p in plan if p[0] == args.only]

    failed = False
    for name, path, checker, extra_checker in plan:
        if not path.exists():
            print(f"[{'warn' if args.allow_missing else 'FAIL'}] {name}: 找不到 {path}")
            failed = failed or not args.allow_missing
            continue
        rows = _read_jsonl(path)
        errs = checker(rows) + (extra_checker(rows) if extra_checker else [])
        if errs:
            failed = True
            print(f"[FAIL] {name}（{len(rows)} 行，{len(errs)} 个问题）")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"[ok]   {name}：{len(rows)} 行，无问题")

    print("\n数据集体检未通过。" if failed else "\n数据集体检通过。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
