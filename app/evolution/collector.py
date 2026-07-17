"""从 Rubric 评测报告采集 P0-破 bad case，并配上**用户真实看到的**最终回复。

关键：采集的不只是 judge 的判词，还有 ``output/eval_<id>/summary.md``——用户可见文本。P0 修复的
验证闸要拿它去核对「judge 说泄露的串，是不是真的出现在给用户的回答里」，而不是只在评估轨迹里
（judge 常把给它做评估用的轨迹误当成 Agent 的回答，见 rubric.py:111 / memory
rubric-judge-calibration-pitfalls）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("shoppingx.evolution.collector")

DEFAULT_REPORT = Path("data/eval/rubric_report.json")
DEFAULT_OUTPUT_ROOT = Path("output")


@dataclass
class BadCase:
    """一条 P0-破的评测记录 + 用户可见的最终回复。"""

    id: str
    query: str
    #: dimension -> rationale（judge 对每条已 fail 的 P0 的判词）
    p0_failures: dict[str, str] = field(default_factory=dict)
    #: 用户真实看到的最终回复（summary.md）。取不到则空串——验证闸会因此拒绝所有该 case 的规则。
    final_text: str = ""
    #: 这条 case 的 Langfuse trace（评测报告里带的）。跟着规则一路传到 learned_rules.json，
    #: 人工确认规则时能一键点开「当时到底怎么泄露的」。未启用观测则空串。
    trace_id: str = ""


def _load_final_text(case_id: str, output_root: Path) -> str:
    """读 ``output/eval_<id>/summary.md``——用户可见的最终回复。取不到返回空串（不抛）。"""
    path = output_root / f"eval_{case_id}" / "summary.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("找不到 %s 的最终回复产物 %s", case_id, path)
        return ""


def load_bad_cases(
    report_path: Path = DEFAULT_REPORT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[BadCase]:
    """读评测报告，返回全部 P0-破（``overall_pass=False``）的 case。

    按 id 天然去重（报告每 id 一条）。
    """
    if not report_path.exists():
        raise SystemExit(f"找不到评测报告 {report_path}，请先跑 scripts/eval/run_rubric.py")
    records = json.loads(report_path.read_text(encoding="utf-8"))

    out: list[BadCase] = []
    for rec in records:
        if not rec.get("ok"):
            continue
        result = rec.get("result", {})
        if result.get("overall_pass", True):
            continue  # 只要 P0-破的
        p0 = {
            s.get("dimension", s.get("id", "")): s.get("rationale", "")
            for s in result.get("scores", [])
            if s.get("tier") == "P0" and s.get("passed") is False
        }
        if not p0:
            continue
        cid = rec.get("id", "")
        out.append(
            BadCase(
                id=cid,
                query=result.get("query", ""),
                p0_failures=p0,
                final_text=_load_final_text(cid, output_root),
                # 报告里没有（旧报告 / 未启用观测）就留空，下游一律降级为「无证据链接」。
                trace_id=rec.get("trace_id") or "",
            )
        )
    return out
