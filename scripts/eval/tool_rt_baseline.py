"""跑一批真实 query，量出各工具的耗时基线，据此校准 RT 告警阈值（refdocs 16-3 §6）。

**为什么需要这个脚本。** ``app/observability/alerts.py`` 的 ``DEFAULT_RULES`` 里那些阈值只是占位；
refdocs 给的 ``item_search=3000ms`` 是本地 mock 数据的水位，而本项目的 item_search 要过 Qdrant 检索
+ siliconflow reranker 跨网 API，照抄会天天响、响到没人再看。**告警阈值是观测的结论，不是观测的
输入**——先量基线，再定阈值。

**为什么不读 /metrics。** 那是 API 进程内的 registry，评测脚本进程里没有那个端点。就算有，
``TOOL_DURATION`` 是 Histogram，只有桶计数（边界 …/5s/10s/30s/60s），P95 只能定位到「落在 5~10s 这
一档」——定阈值不够用。这里直接在打点处收原始样本再算精确分位。

**为什么不跑 judge。** 基线只关心工具耗时，与打分无关。跳过 ``evaluate`` 能省掉每条 query 两次 judge
调用（细则生成 + 打分），跑得更快也更省。

用法::

    uv run python scripts/eval/tool_rt_baseline.py                  # 全集，并发 3
    uv run python scripts/eval/tool_rt_baseline.py --limit 5        # 小样试跑
    uv run python scripts/eval/tool_rt_baseline.py --concurrency 1  # 按配额调

产出 ``data/eval/tool_rt_baseline.json``（gitignore）+ 终端表格 + **建议阈值**（可直接抄进
DEFAULT_RULES）。建议值按 P95×1.8 取整、hard 按 max×2 取整——留出足够余量，宁可漏报也不要天天误报，
一个天天响的告警等于没有告警。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.main_agent import run_agent  # noqa: E402
from app.agent.tracing import flush_traces  # noqa: E402
from app.observability import alerts  # noqa: E402

QUERIES_PATH = Path("data/eval/queries.jsonl")
OUT_PATH = Path("data/eval/tool_rt_baseline.json")

# 采集到的原始样本：tool -> [耗时毫秒, ...]。middleware 只在 status=ok 时打点，与告警窗同口径。
_samples: dict[str, list[float]] = defaultdict(list)

# P50 超过它 → 判定为「阻塞等待型」工具（不是慢，是在等人），不该设 RT 告警规则。
_BLOCKING_P50_MS = 30_000.0


def _collect(tool: str, duration_ms: float, trace_id: str | None = None, now: float | None = None):
    """顶替 alerts.record_tool_sample：收**全部**工具（不限 DEFAULT_RULES 注册表）的原始耗时。

    签名必须与被顶替者一致——middleware 是按位置传前三个参数的。
    """
    _samples[tool].append(duration_ms)


def _percentile(values: list[float], q: float) -> float:
    """与 alerts.percentile 同款 nearest-rank，保证「量出来的」和「告警时算的」是同一把尺子。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = math.ceil(q * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _round_up(ms: float, step: int) -> int:
    """向上取整到 step 的倍数，让建议阈值是个人看得舒服的整数。"""
    return int(math.ceil(ms / step) * step) if ms > 0 else 0


def _load_queries(limit: int | None) -> list[dict]:
    if not QUERIES_PATH.exists():
        raise SystemExit(f"找不到种子集 {QUERIES_PATH}，请先跑 scripts/eval/build_eval_queries.py")
    rows = [
        json.loads(line)
        for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


async def _run_one(q: dict, sem: asyncio.Semaphore) -> bool:
    """跑一条 query（不打分）。单条失败不中断整批——基线是统计量，少一条样本无妨。"""
    async with sem:
        qid = q["id"]
        try:
            await run_agent(q["query"], thread_id=f"rtbase_{qid}", user_id=None)
            print(f"  [done] {qid}", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {qid}: {type(exc).__name__}: {exc}", flush=True)
            return False


def _report() -> dict[str, dict[str, float]]:
    """打印基线表 + 建议阈值，返回结构化统计。"""
    stats: dict[str, dict[str, float]] = {}
    print("\n" + "=" * 92)
    print(
        f"{'工具':<20}{'样本':>6}{'P50':>10}{'P95':>10}{'P99':>10}{'max':>10}   建议阈值(p95/hard)"
    )
    print("=" * 92)

    for tool in sorted(_samples, key=lambda t: -_percentile(_samples[t], 0.95)):
        vals = _samples[tool]
        p50, p95, p99, mx = (
            _percentile(vals, 0.5),
            _percentile(vals, 0.95),
            _percentile(vals, 0.99),
            max(vals),
        )
        # P95×1.8 留抖动余量；hard 取 max×2，兜「单次异常长」而非日常波动。
        sug_p95 = _round_up(p95 * 1.8, 500)
        sug_hard = _round_up(mx * 2.0, 5_000)
        stats[tool] = {
            "count": len(vals),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "max_ms": round(mx, 1),
            "suggest_p95_threshold_ms": sug_p95,
            "suggest_hard_ms": sug_hard,
        }
        # P50 就上到几十秒的，多半不是「慢」而是「本来就在等」（ask_user 等用户回复等满超时）。
        # 这类工具的「建议阈值」是荒谬的（实测 ask_user 会建议出 216 秒），绝不能照抄。
        if p50 >= _BLOCKING_P50_MS:
            warn = "  ⛔阻塞型·勿设RT规则"
        elif len(vals) < 20:
            warn = "  ⚠样本少"
        else:
            warn = ""
        print(
            f"{tool:<20}{len(vals):>6}{p50:>10.0f}{p95:>10.0f}{p99:>10.0f}{mx:>10.0f}"
            f"   {sug_p95:>6} / {sug_hard:<7}{warn}"
        )

    print("=" * 92)
    print(
        "\n⛔ 标记的工具 P50 就在几十秒——不是慢，是**语义上就在等**（如 ask_user 阻塞等用户回复，\n"
        "   等满超时才返回）。给它设 RT 规则等于每次都告警。别抄它那行的建议值。\n"
        "\n⚠ 样本 < 20 的工具，其 P95 在统计上接近 max，建议阈值仅供参考——要么多跑几轮攒样本，\n"
        "   要么只靠 hard_ms 兜底（分位数在低频工具上本就没有统计意义）。"
    )
    return stats


async def main(limit: int | None, concurrency: int) -> int:
    queries = _load_queries(limit)
    print(f"量 RT 基线：{len(queries)} 条 query，并发 {concurrency}，不跑 judge\n")

    # 顶替告警采样点：收全部工具的原始耗时。middleware 通过模块属性访问，故 setattr 生效。
    alerts.record_tool_sample = _collect  # type: ignore[assignment]

    sem = asyncio.Semaphore(concurrency)
    try:
        results = await asyncio.gather(*(_run_one(q, sem) for q in queries))
    finally:
        flush_traces()  # 短命进程：不 flush 则本批的 trace 会随进程蒸发

    ok = sum(results)
    print(f"\n{ok}/{len(queries)} 条跑通，采到 {sum(len(v) for v in _samples.values())} 个工具样本")
    if not _samples:
        print("没有采到任何样本——检查 run_agent 是否真的调了工具。")
        return 1

    stats = _report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整基线已落 {OUT_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量工具耗时基线，校准 RT 告警阈值")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=3, help="并发条数（按 LLM 配额调）")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.limit, args.concurrency)))
