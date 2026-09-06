"""提示词 A/B 判读：把一份或多份 Rubric 报告按提示词版本聚合，出对照表与「达标扩桶」结论。

判读口径与放量策略在 :mod:`app.eval.ab_report`（本脚本只做 IO 与打印）。

怎么攒到两组数据（两条路，选一条别混）：

1. **线上分桶**（真流量）：``.env`` 配 ``PROMPT_AB_VARIANTS=1.1.0:10``，跑一段时间后从
   Langfuse 按 ``version`` / ``metadata.ab_bucket`` 导出，或用同一批种子集在两种身份下重跑。
2. **离线钉版本**（推荐先做这个，快且干净）：跑两遍种子集，每遍用 ``PROMPT_VERSION`` 钉死一个
   版本，产出两份报告文件；本脚本同时吃这两份。

    PROMPT_VERSION=1.0.0 uv run python scripts/eval/run_rubric.py --only ... \\
        && cp data/eval/rubric_report.json data/eval/ab_control.json
    PROMPT_VERSION=1.1.0 uv run python scripts/eval/run_rubric.py --only ... \\
        && cp data/eval/rubric_report.json data/eval/ab_candidate.json
    uv run python scripts/eval/ab_report.py data/eval/ab_control.json data/eval/ab_candidate.json

用法：

    uv run python scripts/eval/ab_report.py [报告.json ...] [--control 1.0.0] [--json 输出.json]
    uv run python scripts/eval/ab_report.py --gate     # 有候选未达标则退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.ab import default_version, variant_weights  # noqa: E402
from app.eval.ab_report import (  # noqa: E402
    UNKNOWN_VERSION,
    VariantStats,
    aggregate,
    policy_from_env,
    rollout_decision,
)

DEFAULT_REPORTS = [Path("data/eval/rubric_report.json")]


def _load(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"找不到报告 {path}（先跑 scripts/eval/run_rubric.py）")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"{path} 不像 rubric 报告（期望一个 list）")
        records.extend(data)
    return records


def _print_table(stats: dict[str, VariantStats], control_version: str) -> None:
    print("=" * 78)
    print(f"提示词 A/B 对照表（对照组 = {control_version}）")
    print("=" * 78)
    print(
        f"{'版本':<16}{'n':>5}{'err':>5}{'P0失败率':>10}"
        f"{'P2均分':>9}{'轮数':>7}{'token':>9}{'总分':>8}"
    )
    for version, s in sorted(stats.items()):
        tag = "  ← 对照" if version == control_version else ""
        print(
            f"{version:<16}{s.n:>5}{s.errored:>5}{s.p0_fail_rate:>9.1%}"
            f"{s.p2_avg:>9.2f}{s.avg_rounds:>7.2f}{s.avg_tokens:>9.0f}{s.avg_total:>8.1f}{tag}"
        )
    if UNKNOWN_VERSION in stats:
        print(
            f"\n[note] 有 {stats[UNKNOWN_VERSION].n} 条记录没有 prompt_version"
            "（旧报告，或整轮缓存命中那种压根没跑提示词的轮次），单列一档不参与判读。"
        )


def main(paths: list[Path], control: str | None, out_json: Path | None, gate: bool) -> int:
    records = _load(paths)
    stats = aggregate(records)
    control_version = control or default_version()
    if control_version not in stats:
        raise SystemExit(
            f"报告里没有对照组版本 {control_version} 的记录（现有：{sorted(stats)}）。"
            "用 --control 指定，或确认跑评测时 PROMPT_VERSION 配对了。"
        )
    _print_table(stats, control_version)

    current = dict(variant_weights())
    policy = policy_from_env()
    decisions = []
    for version, s in sorted(stats.items()):
        if version in (control_version, UNKNOWN_VERSION):
            continue
        d = rollout_decision(stats[control_version], s, current.get(version, 0), policy)
        decisions.append(d)
        print(f"\n候选 {version}（当前放量 {d.current_pct}%）")
        for line in d.reasons:
            print(f"  {line}")
        if d.meets and d.suggested_pct > d.current_pct:
            print(f"  → 达标，可扩桶到 {d.suggested_pct}%：")
            print(f"     PROMPT_AB_VARIANTS={version}:{d.suggested_pct}")
            print("     （**手工改 .env**，本脚本不改任何配置，见 app/eval/ab_report 模块说明）")
        elif d.meets:
            print(
                f"  → 达标，且已到阶梯顶（{d.current_pct}%）。"
                "可考虑把它设为 PROMPT_VERSION 基线。"
            )
        else:
            print(f"  → 未达标，维持 {d.current_pct}%。")

    if not decisions:
        print("\n只有对照组一个版本，没有可判读的候选。")

    if out_json is not None:
        payload = {
            "control": control_version,
            "variants": {v: s.as_dict() for v, s in stats.items()},
            "decisions": [
                {
                    "version": d.version,
                    "meets": d.meets,
                    "current_pct": d.current_pct,
                    "suggested_pct": d.suggested_pct,
                    "reasons": d.reasons,
                }
                for d in decisions
            ],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结论已落 {out_json}")

    if gate and any(not d.meets for d in decisions):
        print("\n[gate] 有候选版本未达标，退出码 1")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*", type=Path, help="rubric 报告 json（可多份）")
    parser.add_argument("--control", default=None, help="对照组版本号（默认取 PROMPT_VERSION）")
    parser.add_argument("--json", dest="out_json", type=Path, default=None, help="结论落盘路径")
    parser.add_argument("--gate", action="store_true", help="有候选未达标则退出码 1")
    args = parser.parse_args()
    raise SystemExit(main(args.reports or DEFAULT_REPORTS, args.control, args.out_json, args.gate))
