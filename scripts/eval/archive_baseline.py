"""把一次 Rubric 全跑存档成**带环境指纹**的基线，供后续 A/B 对照。

**为什么非要存指纹**：A/B 两组只有在同一环境下才可比。M22 的端到端 A/B 之所以无结论，
噪声是一半原因，另一半是「两次跑之间到底还有什么变了」说不清。模型名、索引名、OpenSearch
在不在、代码 commit——任何一个不同，分差就不能全算到被测改动头上。

**支持合并多份报告**：`run_rubric.py --only` 补跑失败条目时只会写它跑的那几条。补跑完要和
上一份合并才是完整基线，按 id 去重、后来者覆盖。

用法：
    uv run python scripts/eval/archive_baseline.py --name A_M23_S0_4 \\
        --merge /tmp/rubric_90_partial.json data/eval/rubric_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "baselines"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def _fingerprint(stamp: str) -> dict:
    """环境指纹。**只记真正会影响分数的东西**，记一堆无关配置反而让人看不出差异在哪。"""
    return {
        "时间": stamp,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "分支": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "工作区干净": not _git("status", "--porcelain"),
        "LLM_MAIN": os.getenv("LLM_MAIN", ""),
        "LLM_FAST": os.getenv("LLM_FAST", ""),
        "LLM_JUDGE": os.getenv("LLM_JUDGE", ""),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", ""),
        "EMBED_MODEL": os.getenv("EMBED_MODEL", ""),
        "RERANK_MODEL": os.getenv("RERANK_MODEL", ""),
        "OPENSEARCH": os.getenv("OPENSEARCH_HOST", "") or "（未起，品类 KB 走本地回退）",
    }


def _summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    tot = [r["result"]["total"] for r in ok]
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        by_bucket[r["bucket"]].append(r["result"]["total"])

    target = [v for b, vs in by_bucket.items() if b.startswith("靶-") for v in vs]
    normal = [v for b, vs in by_bucket.items() if not b.startswith("靶-") for v in vs]
    n = len(tot) or 1
    sd = statistics.pstdev(tot) if len(tot) > 1 else 0.0
    return {
        "条数": {"总计": len(rows), "完成": len(ok), "失败": len(rows) - len(ok)},
        "均分": round(statistics.mean(tot), 2) if tot else 0.0,
        "单条标准差": round(sd, 2),
        # 均值标准误才是 A/B 该比的尺度——单条 σ 大不代表均值不稳，它按 √n 收敛
        "均值标准误": round(sd / (n ** 0.5), 2),
        "P0 红线失败": sum(1 for r in ok if r["result"]["p0_failures"]),
        "overall_pass": f"{sum(1 for r in ok if r['result']['overall_pass'])}/{len(ok)}",
        "靶心族均分": round(statistics.mean(target), 2) if target else None,
        "常规均分": round(statistics.mean(normal), 2) if normal else None,
        "分桶": {
            b: {"n": len(v), "均分": round(statistics.mean(v), 1)}
            for b, v in sorted(by_bucket.items())
        },
        "失败条目": [r["id"] for r in rows if not r.get("ok")],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="基线名，如 A_M23_S0_4")
    ap.add_argument(
        "--merge", nargs="+", required=True, help="要合并的 rubric 报告（后者覆盖前者）"
    )
    ap.add_argument("--stamp", default="", help="时间戳（不传则由 git 最近提交时间代）")
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    for p in args.merge:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"报告不存在：{path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        # 后来者覆盖：补跑的结果应当顶掉上一份里那条 error
        for r in rows:
            merged[r["id"]] = r
        print(f"  合入 {path.name}：{len(rows)} 条")

    rows = sorted(merged.values(), key=lambda r: r["id"])
    stamp = args.stamp or _git("log", "-1", "--format=%cI")
    doc = {
        "基线名": args.name,
        "环境指纹": _fingerprint(stamp),
        "汇总": _summarize(rows),
        "明细": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"环境指纹": doc["环境指纹"], "汇总": {
        k: v for k, v in doc["汇总"].items() if k != "分桶"
    }}, ensure_ascii=False, indent=2))
    print(f"\n→ {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
