"""P0 自动修规则飞轮腿（app/evolution + app/security/learned_rules）。

核心不变式：**不能因为 judge 假阳性就往安全规则表里加假规则**。多数断言围绕「什么情况下**不**产
规则」——真泄露闸、可脱类别闸、公网外链豁免、去重。真实 baseline 报告作回归锚点（q08 假阳性）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evolution.collector import BadCase, load_bad_cases
from app.evolution.p0_fixer import propose_rules
from app.evolution.router import Route, classify


@pytest.fixture(autouse=True)
def _isolated_rules(tmp_path, monkeypatch):
    """每个测试用独立的 learned_rules.json，避免污染真实文件 / 相互串味。"""
    from app.security import learned_rules

    monkeypatch.setattr(learned_rules, "RULES_PATH", tmp_path / "learned_rules.json")
    learned_rules.reset_for_test()
    yield
    learned_rules.reset_for_test()


# ---------- 分流 ----------


class TestRouter:
    def test_leak_dimension(self) -> None:
        assert classify("信息安全", "暴露了内部字段 landed_usd") is Route.LEAK
        assert classify("安全合规", "泄露了内网地址 http://qdrant:6333") is Route.LEAK

    def test_banned_beats_leak(self) -> None:
        # 一条判词同时提「推荐仿品」和「轨迹暴露 item_id」→ 判 BANNED（前者是真问题）
        r = classify("合规红线", "推荐了 SHEIN 仿大牌，且轨迹暴露 item_id")
        assert r is Route.BANNED

    def test_judgment_default(self) -> None:
        assert classify("预算合规", "未展示商品总价明细，无法验证") is Route.JUDGMENT
        assert classify("品类合规", "把女士用品推给了男性礼物场景") is Route.JUDGMENT


# ---------- P0 fixer 的两道闸 ----------


class TestRealLeakGate:
    """真泄露闸：judge 说泄露，但用户可见文本里没有 → 不产规则（拦假阳性）。"""

    def test_rejects_judge_false_positive(self) -> None:
        # 复刻 q08：judge 说泄露 landed_usd/item_id/工具名，但 final_text 干净
        case = BadCase(
            id="q08",
            query="帮我查降噪耳机的评测",
            p0_failures={
                "信息安全": "商品卡暴露 landed_usd，轨迹泄露 item_id 及工具名 dispatch_tool"
            },
            final_text="推荐：Sony WH-1000XM5，降噪出色，续航30小时。",
        )
        proposals, routes = propose_rules(case)
        assert routes["信息安全"] is Route.LEAK  # 分流认得它是泄露类
        assert proposals == []  # 但用户文本干净 → 不产规则

    def test_no_final_text_no_rule(self) -> None:
        case = BadCase(id="x", query="q", p0_failures={"信息安全": "泄露 endpoint"}, final_text="")
        assert propose_rules(case)[0] == []


class TestCategoryGate:
    """可脱类别闸：只提议内网 endpoint；item_id / 工具名故意不脱（output_guard 的既定取向）。"""

    def test_item_id_in_user_text_not_proposed(self) -> None:
        # 哪怕 item_id 真出现在用户文本里，也不产规则——它常是用户想要的信息
        case = BadCase(
            id="x",
            query="q",
            p0_failures={"信息安全": "泄露了 item_id"},
            final_text="推荐 item_id=B07XYZ1234，可拿去平台搜同款。",
        )
        assert propose_rules(case)[0] == []

    def test_tool_name_not_proposed(self) -> None:
        case = BadCase(
            id="x",
            query="q",
            p0_failures={"信息安全": "泄露工具名 parallel_dispatch_tool"},
            final_text="我用 parallel_dispatch_tool 帮你并行搜了 5 个平台。",
        )
        assert propose_rules(case)[0] == []


class TestAcceptedLeak:
    """接受路径：用户文本里真有 curated 没覆盖的内网 endpoint → 产 1 条候选规则。"""

    def test_new_internal_endpoint_produces_rule(self) -> None:
        case = BadCase(
            id="qx",
            query="q",
            p0_failures={"安全合规": "回复暴露了内部服务地址"},
            final_text="推荐清单已生成。数据源 http://pricing-svc:8080/rank 检索到 5 件。",
        )
        proposals, _ = propose_rules(case)
        assert len(proposals) == 1
        assert proposals[0].literal == "http://pricing-svc:8080"  # 取 host:port，丢 path
        assert proposals[0].source_case == "qx"

    def test_public_url_not_redacted(self) -> None:
        # 给用户的正经外链（公网 TLD）绝不脱
        case = BadCase(
            id="qx",
            query="q",
            p0_failures={"安全合规": "暴露地址"},
            final_text="在 https://www.amazon.com/dp/B07XYZ 有货。",
        )
        assert propose_rules(case)[0] == []

    def test_curated_covered_endpoint_not_duplicated(self) -> None:
        # curated 已能脱的（qdrant 在内网 host 列表里）→ 不重复提议
        case = BadCase(
            id="qx",
            query="q",
            p0_failures={"安全合规": "暴露地址"},
            final_text="数据源 http://qdrant:6333/collections 检索。",
        )
        assert propose_rules(case)[0] == []


# ---------- learned 叠加层：落盘 / 去重 / 生效 ----------


class TestLearnedOverlay:
    def test_add_candidate_makes_audit_redact(self) -> None:
        from app.security import learned_rules
        from app.security.output_guard import audit_output

        leak = "http://pricing-svc:8080"
        assert audit_output(f"数据源 {leak}")[0] is True  # 加规则前：curated 抓不到，干净
        assert learned_rules.add_candidate("learned_endpoint_pricing_svc", leak, "qx")
        clean, cleaned, hits = audit_output(f"数据源 {leak} 检索")
        assert clean is False  # 加规则后：命中并脱敏
        assert "[已脱敏]" in cleaned
        assert "learned_endpoint_pricing_svc" in hits

    def test_dedup_by_literal(self) -> None:
        from app.security import learned_rules

        assert learned_rules.add_candidate("a", "http://svc:9000", "c1") is True
        assert learned_rules.add_candidate("b", "http://svc:9000", "c2") is False  # 同串不重复
        assert len(learned_rules.load_rules()) == 1

    def test_disabled_rule_not_live(self) -> None:
        from app.security import learned_rules

        learned_rules.add_candidate("x", "http://svc:9000", "c1")
        rules = learned_rules.load_rules()
        rules[0].state = "disabled"
        learned_rules.save_rules(rules)
        # disabled 不进生效编译
        names = [n for n, _ in learned_rules.learned_output_patterns()]
        assert "x" not in names

    def test_curated_patterns_never_written(self) -> None:
        # 飞轮只写 learned 文件；in-code 的 _SENSITIVE_PATTERNS 是元组，结构上不可被 append
        from app.security import learned_rules, output_guard

        learned_rules.add_candidate("x", "http://svc:9000", "c1")
        assert isinstance(output_guard._SENSITIVE_PATTERNS, tuple)  # 基线仍是不可变元组


# ---------- 真实 baseline 报告回归 ----------


class TestRealBaselineReport:
    """真实 rubric_report_baseline.json：6 条 P0-破，产 0 条规则（全是假阳性/判断类）。"""

    def test_baseline_produces_zero_rules(self) -> None:
        report = Path("data/eval/rubric_report_baseline.json")
        if not report.exists():
            pytest.skip("baseline 报告不在（gitignore，靠脚本复现）")
        cases = load_bad_cases(report)
        assert len(cases) >= 1  # 确实有 P0-破的
        total_proposals = sum(len(propose_rules(c)[0]) for c in cases)
        assert total_proposals == 0, "baseline 上不该产任何规则——泄露类全是 judge 假阳性"


def _write_report(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


class TestCollector:
    def test_only_p0_broken_collected(self, tmp_path) -> None:
        report = tmp_path / "r.json"
        _write_report(
            report,
            [
                {
                    "id": "pass1",
                    "ok": True,
                    "result": {"query": "q", "overall_pass": True, "scores": []},
                },
                {
                    "id": "fail1",
                    "ok": True,
                    "result": {
                        "query": "q2",
                        "overall_pass": False,
                        "scores": [
                            {
                                "tier": "P0",
                                "dimension": "信息安全",
                                "passed": False,
                                "rationale": "泄露",
                            },
                            {"tier": "P1", "dimension": "x", "passed": False, "rationale": "y"},
                        ],
                    },
                },
            ],
        )
        cases = load_bad_cases(report, output_root=tmp_path)
        assert [c.id for c in cases] == ["fail1"]
        assert "信息安全" in cases[0].p0_failures
        assert cases[0].final_text == ""  # 无 summary.md → 空（验证闸据此拒绝）

    def test_final_text_loaded_from_summary(self, tmp_path) -> None:
        report = tmp_path / "r.json"
        _write_report(
            report,
            [
                {
                    "id": "c1",
                    "ok": True,
                    "result": {
                        "query": "q",
                        "overall_pass": False,
                        "scores": [
                            {
                                "tier": "P0",
                                "dimension": "安全合规",
                                "passed": False,
                                "rationale": "泄露",
                            }
                        ],
                    },
                }
            ],
        )
        d = tmp_path / "eval_c1"
        d.mkdir()
        (d / "summary.md").write_text("最终回复内容", encoding="utf-8")
        cases = load_bad_cases(report, output_root=tmp_path)
        assert cases[0].final_text == "最终回复内容"


# ---------- 证据 trace：bad case → 规则 → 落盘 ----------
class TestEvidenceTrace:
    """规则带上「哪条 trace 生的」。source_case 只说是哪条 query，trace 才说清怎么泄露的。"""

    def test_trace_id_flows_from_report_to_proposal(self, tmp_path) -> None:
        report = tmp_path / "r.json"
        _write_report(
            report,
            [
                {
                    "id": "c1",
                    "ok": True,
                    "trace_id": "trace-evidence-1",
                    "result": {
                        "query": "q",
                        "overall_pass": False,
                        "scores": [
                            {
                                "tier": "P0",
                                "dimension": "安全合规",
                                "passed": False,
                                "rationale": "回复暴露了内部服务地址",
                            }
                        ],
                    },
                }
            ],
        )
        (tmp_path / "eval_c1").mkdir()
        (tmp_path / "eval_c1" / "summary.md").write_text(
            "数据源 http://pricing-svc:8080/rank 检索到 5 件。", encoding="utf-8"
        )

        case = load_bad_cases(report, output_root=tmp_path)[0]
        assert case.trace_id == "trace-evidence-1"
        proposals, _ = propose_rules(case)
        assert proposals[0].evidence_trace == "trace-evidence-1"

    def test_missing_trace_id_degrades_to_empty(self, tmp_path) -> None:
        """旧报告 / 未启用观测时报告里没有 trace_id → 降级为空串，不炸。"""
        report = tmp_path / "r.json"
        _write_report(
            report,
            [
                {
                    "id": "c1",
                    "ok": True,
                    "result": {
                        "query": "q",
                        "overall_pass": False,
                        "scores": [
                            {
                                "tier": "P0",
                                "dimension": "安全合规",
                                "passed": False,
                                "rationale": "泄露",
                            }
                        ],
                    },
                }
            ],
        )
        assert load_bad_cases(report, output_root=tmp_path)[0].trace_id == ""

    def test_legacy_rules_json_without_evidence_field_still_loads(
        self, tmp_path, monkeypatch
    ) -> None:
        """M14 写的旧 learned_rules.json 没有 evidence_trace 键，新代码必须照样读得出来。"""
        from app.security import learned_rules as LR

        path = tmp_path / "learned_rules.json"
        path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "learned_endpoint_old",
                            "pattern": "http://old-svc:9000",
                            "source_case": "q_old",
                            "state": "candidate",
                            "created": "via evolve_p0",
                            "kind": "literal",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(LR, "RULES_PATH", path)
        monkeypatch.setattr(LR, "_cache", None)

        rules = LR.load_rules()
        assert len(rules) == 1
        assert rules[0].evidence_trace == ""  # 缺字段 → 默认空串

    def test_add_candidate_persists_evidence_trace(self, tmp_path, monkeypatch) -> None:
        from app.security import learned_rules as LR

        path = tmp_path / "learned_rules.json"
        monkeypatch.setattr(LR, "RULES_PATH", path)
        monkeypatch.setattr(LR, "_cache", None)

        assert LR.add_candidate(
            "learned_endpoint_x", "http://svc:1", "qx", evidence_trace="trace-abc"
        )
        LR._cache = None
        assert LR.load_rules()[0].evidence_trace == "trace-abc"
