"""P0 bad case → 自动沉淀脱敏规则（refdocs 18-2 §4，上下文级飞轮的 P0 腿，无 GPU）。

与 ``distill_fewshot.py`` 对称：那条是 P2/质量腿（高分轨迹 → few-shot），这条是 P0/安全腿
（泄露类 bad case → learned 脱敏规则）。

    uv run python scripts/eval/evolve_p0.py                    # dry-run：打印提议 + 分流
    uv run python scripts/eval/evolve_p0.py --write            # 落候选规则到 learned_rules.json
    uv run python scripts/eval/evolve_p0.py --review           # 列当前候选规则（待人工确认）
    uv run python scripts/eval/evolve_p0.py --verify           # 无 LLM 误伤回归：新规则会不会误脱

**只有 LEAK 类自动产规则，且过两道闸**（可脱类别 + 真泄露，见 p0_fixer）。BANNED / JUDGMENT 只打印
上报，不动任何规则——判断类红线规则堵不住，违禁品类没有正确的规则落点。

**闭环的诚实边界**：``--verify`` 只做**确定性半场**——拿新规则重扫所有已有 eval 产物，报告有没有
把本来干净的回复误脱（over-redaction，这是坏规则的真实风险，零 LLM 可查）。「那条 P0 现在过没过」
的另半场要重跑 Agent + judge（``run_rubric.py``），带 LLM，不在本脚本内。
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.tracing import trace_url  # noqa: E402
from app.evolution.collector import load_bad_cases  # noqa: E402
from app.evolution.p0_fixer import ProposedRule, propose_rules  # noqa: E402
from app.evolution.router import Route  # noqa: E402
from app.security import learned_rules  # noqa: E402
from app.security.output_guard import audit_output  # noqa: E402

REPORT_PATH = Path("data/eval/rubric_report.json")


def _collect(report_path: Path) -> tuple[list[ProposedRule], dict[str, list[str]]]:
    """跑采集 + 分流 + 提议。返回（去重后的候选规则, 分流上报桶）。"""
    cases = load_bad_cases(report_path)
    print(f"读到 {len(cases)} 条 P0-破 bad case：{', '.join(c.id for c in cases) or '（无）'}\n")

    proposals: list[ProposedRule] = []
    seen: set[str] = set()
    buckets: dict[str, list[str]] = {"banned": [], "judgment": [], "leak_no_rule": []}
    for case in cases:
        rules, routes = propose_rules(case)
        for dim, route in routes.items():
            tag = f"{case.id}/{dim}"
            if route is Route.BANNED:
                buckets["banned"].append(tag)
            elif route is Route.JUDGMENT:
                buckets["judgment"].append(tag)
        # LEAK 但没产出规则（被真泄露闸拦下，如 q08 假阳性）
        if Route.LEAK in routes.values() and not rules:
            buckets["leak_no_rule"].append(case.id)
        for r in rules:
            if r.literal in seen:
                continue
            seen.add(r.literal)
            proposals.append(r)
    return proposals, buckets


def _print_report(proposals: list[ProposedRule], buckets: dict[str, list[str]]) -> None:
    print("=== 自动可修（LEAK，过两道闸）===")
    if proposals:
        for r in proposals:
            print(f"  + {r.name}: {r.literal!r}  ← {r.source_case}")
            # 证据链接：判词只说「暴露了内部端点」，trace 里能看到是哪个工具的返回把它带出来的。
            if url := trace_url(r.evidence_trace):
                print(f"      证据 trace: {url}")
    else:
        print("  （无——所有泄露类 P0 要么是 judge 假阳性、要么已被 curated 覆盖）")
    if buckets["leak_no_rule"]:
        print(
            f"\n  被真泄露闸拦下（judge 报泄露但用户文本无）：{', '.join(buckets['leak_no_rule'])}"
        )
    print("\n=== 不自动修（只上报）===")
    print(f"  BANNED 违禁/仿品 → 人工/工具层：{', '.join(buckets['banned']) or '（无）'}")
    print(f"  JUDGMENT 判断类红线 → prompt 腿/人工：{', '.join(buckets['judgment']) or '（无）'}")


def _do_write(proposals: list[ProposedRule]) -> None:
    if not proposals:
        print("\n无候选规则可写。")
        return
    added = 0
    for r in proposals:
        # created 由脚本外部时钟拿不到（对齐工作区约定），用固定占位；真实时间以文件 mtime 为准。
        if learned_rules.add_candidate(
            r.name,
            r.literal,
            r.source_case,
            created="via evolve_p0",
            evidence_trace=r.evidence_trace,
        ):
            added += 1
    print(f"\n已写入 {added} 条候选规则到 {learned_rules.RULES_PATH}")
    print("候选态立即生效（P0 宁可误杀），人工确认转 active / 否掉转 disabled。")


def _do_review() -> None:
    cands = learned_rules.list_candidates()
    if not cands:
        print("当前没有待确认的候选规则。")
        return
    print(f"待人工确认的候选规则（{len(cands)} 条）：")
    for r in cands:
        print(f"  [{r.state}] {r.name}: {r.pattern!r}  ← {r.source_case}")
        if url := trace_url(r.evidence_trace):
            print(f"      证据 trace: {url}")
    print(
        "\n确认：把该条 state 改为 active；否掉：改为 disabled（不生效）。文件："
        f"{learned_rules.RULES_PATH}"
    )


def _do_verify(output_root: str = "output") -> None:
    """确定性误伤回归：拿当前规则（含新候选）重扫所有 eval 最终回复，报告误脱。"""
    print("=== 误伤回归（无 LLM）：新规则会不会误脱正常回复 ===")
    collateral = 0
    for f in sorted(glob.glob(f"{output_root}/eval_*/summary.md")):
        text = Path(f).read_text(encoding="utf-8")
        clean, cleaned, hits = audit_output(text)
        if not clean:
            collateral += 1
            qid = Path(f).parent.name
            print(f"  ⚠ {qid} 被脱敏，命中 {hits}")
    if collateral == 0:
        print("  ✓ 没有任何已有回复被误脱——新规则安全。")
    else:
        print(
            f"\n  {collateral} 条已有回复被脱敏。若其中有本该干净的，说明规则过宽，"
            "去 learned_rules.json 把对应规则 state 改 disabled。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 bad case 自动沉淀脱敏规则")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Rubric 评测报告路径")
    parser.add_argument("--write", action="store_true", help="落盘候选规则（默认 dry-run）")
    parser.add_argument("--review", action="store_true", help="列当前候选规则")
    parser.add_argument("--verify", action="store_true", help="无 LLM 误伤回归：重扫已有回复")
    args = parser.parse_args()

    if args.review:
        _do_review()
        return
    if args.verify:
        _do_verify()
        return

    proposals, buckets = _collect(Path(args.report))
    _print_report(proposals, buckets)
    if args.write:
        _do_write(proposals)
    else:
        print("\n（dry-run，未落盘；加 --write 写入候选规则）")


if __name__ == "__main__":
    main()
