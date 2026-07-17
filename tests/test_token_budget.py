"""F 块 · token / 成本预算闸的确定性单测：计费 + 去重 + 档位阈值 + 全树聚合 + 夺权联动。

成本用可控的 usage_metadata（手造 token 数）+ 已知费率（env 覆盖），断言累计成本、软/硬档位、
按 session_dir 全树聚合、AIMessage id 去重不重复计费；并验证越硬线后 middleware 摘掉成本放大器工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import token_budget as tb
from app.utils.thread_ctx import thread_scope


def _ai(mid: str, *, input_tokens: int, output_tokens: int, cache_read: int = 0) -> AIMessage:
    """造一条带 usage_metadata 的 AIMessage（驱动计费）。"""
    return AIMessage(
        content="",
        id=mid,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
    )


@pytest.fixture
def fixed_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """固定费率：input=1.0 / output=2.0 / cache_read=0.0（美元每百万 token），便于手算。"""
    monkeypatch.setenv("TOKEN_PRICE_INPUT", "1.0")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT", "2.0")
    monkeypatch.setenv("TOKEN_PRICE_CACHE_READ", "0.0")


def test_charge_accumulates_cost(tmp_path: Path, fixed_price: None) -> None:
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        # 1M input @1.0 + 1M output @2.0 = 3.0 美元
        cost = tb.charge_tree_usage([_ai("m1", input_tokens=1_000_000, output_tokens=1_000_000)])
        assert cost == pytest.approx(3.0)
        snap = tb.tree_snapshot()
        assert snap is not None
        assert snap["model_calls"] == 1
        assert snap["input_tokens"] == 1_000_000
        tb.reset_tree()


def test_cache_read_discounted(tmp_path: Path, fixed_price: None) -> None:
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        # input=1M 其中 cache_read=0.6M：非缓存 0.4M @1.0 + 缓存 0.6M @0.0 = 0.4；output 0
        cost = tb.charge_tree_usage(
            [_ai("m1", input_tokens=1_000_000, output_tokens=0, cache_read=600_000)]
        )
        assert cost == pytest.approx(0.4)
        tb.reset_tree()


def test_dedup_by_message_id(tmp_path: Path, fixed_price: None) -> None:
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        msg = _ai("same", input_tokens=1_000_000, output_tokens=0)
        tb.charge_tree_usage([msg])
        tb.charge_tree_usage([msg])  # 同 id 重复传：不重复计费
        snap = tb.tree_snapshot()
        assert snap is not None
        assert snap["model_calls"] == 1
        assert snap["cost_usd"] == pytest.approx(1.0)
        tb.reset_tree()


def test_budget_status_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKEN_PRICE_INPUT", "1.0")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT", "0.0")
    monkeypatch.setenv("TOKEN_PRICE_CACHE_READ", "0.0")
    monkeypatch.setenv("TOKEN_BUDGET_USD", "1.0")
    monkeypatch.setenv("TOKEN_BUDGET_SOFT_RATIO", "0.8")
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        assert tb.budget_status() == "ok"  # 还没花
        tb.charge_tree_usage([_ai("m1", input_tokens=700_000, output_tokens=0)])  # 0.7 < 0.8
        assert tb.budget_status() == "ok"
        tb.charge_tree_usage([_ai("m2", input_tokens=150_000, output_tokens=0)])  # 0.85 ≥ 0.8
        assert tb.budget_status() == "soft"
        tb.charge_tree_usage([_ai("m3", input_tokens=200_000, output_tokens=0)])  # 1.05 ≥ 1.0
        assert tb.budget_status() == "hard"
        tb.reset_tree()


def test_budget_disabled_when_cap_nonpositive(tmp_path: Path, fixed_price: None) -> None:
    # 上限 <=0 视为不设闸：花再多也是 ok（关闭预算控制场景）。
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        import os

        os.environ["TOKEN_BUDGET_USD"] = "0"
        tb.charge_tree_usage([_ai("m1", input_tokens=10_000_000, output_tokens=10_000_000)])
        assert tb.budget_status() == "ok"
        del os.environ["TOKEN_BUDGET_USD"]
        tb.reset_tree()


def test_no_scope_returns_none() -> None:
    # 无 session 作用域（单测裸调）：不计费、不设档（不平添门槛）。
    assert tb.charge_tree_usage([_ai("m1", input_tokens=1, output_tokens=1)]) is None
    assert tb.peek_tree_cost() is None
    assert tb.budget_status() == "ok"
    assert tb.tree_snapshot() is None


def test_tree_aggregates_across_scopes(tmp_path: Path, fixed_price: None) -> None:
    # 同一 session_dir 多次 charge（模拟主 + 子按同一 key 聚合）累加到一棵树。
    with thread_scope("main", tmp_path):
        tb.reset_tree()
        tb.charge_tree_usage([_ai("m1", input_tokens=1_000_000, output_tokens=0)])
    # 同 session_dir、不同 thread_id（模拟子 Agent 继承父 session_dir）。
    with thread_scope("sub", tmp_path):
        cost = tb.charge_tree_usage([_ai("s1", input_tokens=1_000_000, output_tokens=0)])
        assert cost == pytest.approx(2.0)  # 主 1.0 + 子 1.0 同树累加
        tb.reset_tree()


def test_non_ai_and_missing_usage_ignored(tmp_path: Path, fixed_price: None) -> None:
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        tb.charge_tree_usage([HumanMessage(content="hi"), AIMessage(content="no usage")])
        snap = tb.tree_snapshot()
        assert snap is not None
        assert snap["model_calls"] == 0  # 非 AIMessage / 无 usage 都跳过
        tb.reset_tree()


async def test_hard_budget_blocks_cost_amplifiers_at_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 越硬线后：执行层（pre_tool_call 的 token_budget_gate）拦截成本放大器工具。
    from app.harness.agent_middleware import HarnessAgentMiddleware
    from app.harness.budgets import COST_AMPLIFIER_TOOLS
    from app.harness.setup import setup_harness
    from app.harness.state import GuardState

    monkeypatch.setenv("TOKEN_PRICE_INPUT", "1.0")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT", "0.0")
    monkeypatch.setenv("TOKEN_BUDGET_USD", "1.0")
    setup_harness()

    from types import SimpleNamespace

    from langchain_core.messages import ToolMessage

    def _treq(name: str) -> Any:
        return SimpleNamespace(tool_call={"name": name, "args": {}, "id": f"tc_{name}"})

    def _thandle(content: str, name: str) -> Any:
        async def h(_: Any) -> ToolMessage:
            return ToolMessage(content=content, tool_call_id="x", name=name)

        return h

    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        mw = HarnessAgentMiddleware(guard=GuardState(loop_threshold=99, tree_retrieval_cap=99))
        # 未越预算：成本放大器工具正常执行
        ok = await mw.awrap_tool_call(_treq("item_search"), _thandle("ok", "item_search"))
        assert "ok" in ok.content

        tb.charge_tree_usage([_ai("big", input_tokens=2_000_000, output_tokens=0)])  # 2.0 ≥ 1.0

        # 越硬线后：执行层拦截成本放大器工具（工具表始终不变，见 test_main_agent 的不变式测试）
        for name in COST_AMPLIFIER_TOOLS:
            mw2 = HarnessAgentMiddleware(guard=GuardState(loop_threshold=99, tree_retrieval_cap=99))
            result = await mw2.awrap_tool_call(_treq(name), _thandle("ok", name))
            assert "token 预算已超限" in result.content

        # 收尾链工具正常放行
        mw3 = HarnessAgentMiddleware(guard=GuardState(loop_threshold=99))
        result = await mw3.awrap_tool_call(_treq("price_compare"), _thandle("ok", "price_compare"))
        assert "ok" in result.content
        tb.reset_tree()


def test_charge_usage_never_raises_on_malformed(tmp_path: Path, fixed_price: None) -> None:
    # 计费绝不反噬主链路：非常规 result 形状（非可迭代）会让 charge 内部抛，_charge_usage 必须吞掉。
    from types import SimpleNamespace

    from app.harness.agent_middleware import HarnessAgentMiddleware

    mw = HarnessAgentMiddleware()
    with thread_scope("t1", tmp_path):
        tb.reset_tree()
        mw._charge_usage(SimpleNamespace(result=42))  # 非可迭代 → for 抛 TypeError，须吞掉
        mw._charge_usage(SimpleNamespace())  # 无 result 属性 → 安全跳过
        mw._charge_usage(SimpleNamespace(result=None))  # result=None → 安全跳过
        # 都不抛即通过；顺带确认没误记账。
        snap = tb.tree_snapshot()
        assert snap is None or snap["model_calls"] == 0
        tb.reset_tree()


def _fake_req(tool_names: list[str]) -> Any:
    """造一个假 ModelRequest：只需 .tools（带 .name）+ .override(tools=...)。"""
    from types import SimpleNamespace

    def make(names: list[str]) -> Any:
        ns = SimpleNamespace(tools=[SimpleNamespace(name=n) for n in names])
        ns.override = lambda tools: make([t.name for t in tools])
        return ns

    return make(tool_names)


def test_charge_tool_llm_usage(tmp_path: Path, fixed_price: None) -> None:
    """工具内部 LLM 调用（planner / shopping_summary / chat_fallback）经 callback 收集的
    usage 入同一棵树——修「总账恰好只等于主 loop 之和」的漏账（perf-audit-r5 实测）。"""
    with thread_scope("t-tool-usage", tmp_path):
        tb.reset_tree()
        tb.charge_tree_usage([_ai("m1", input_tokens=1_000_000, output_tokens=0)])
        # UsageMetadataCallbackHandler.usage_metadata 的形状：model_name → UsageMetadata
        tb.charge_tool_llm_usage(
            {
                "deepseek-v4-flash": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "input_token_details": {"cache_read": 500_000},
                }
            }
        )
        snap = tb.tree_snapshot()
        assert snap is not None
        assert snap["model_calls"] == 2  # 主 loop 1 次 + 工具内 1 次
        assert snap["input_tokens"] == 2_000_000
        assert snap["cache_read_tokens"] == 500_000
        # 主 1M input @1.0 + 工具 (0.5M 非缓存 @1.0 + 0.5M cache @0.0 + 1M output @2.0) = 3.5
        assert snap["cost_usd"] == pytest.approx(3.5)
        tb.reset_tree()


def test_charge_tool_llm_usage_no_scope_is_noop() -> None:
    """无会话作用域（单测直调工具）→ 静默跳过，绝不抛。"""
    tb.charge_tool_llm_usage({"m": {"input_tokens": 1, "output_tokens": 1}})
