"""round3 刀 2：检索合流后自动比价 + 精挑（app/harness/autopick.py）。

测的是机制：武装 / 解除的触发条件、自动执行后控制面状态与模型亲手调等价、套装轮与关开关
两条豁免。真实 LLM 不参与（price_compare / item_picker 都是本地计算 + 本地回退 rerank）。
"""

import tempfile
from pathlib import Path

import pytest

from app.harness.autopick import arm_on_tool, maybe_autopick
from app.harness.phase_machine import Phase, PhaseStateMachine, set_phase_machine
from app.harness.session import HarnessSession
from app.tools._candidates import get_last_picks, register
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope


def _cands() -> list[ItemCandidate]:
    return [
        ItemCandidate(item_id="A1", platform="amazon", title="canvas travel pouch set",
                      price=20, currency="USD", rating=4.6),
        ItemCandidate(item_id="A2", platform="amazon", title="nylon packing cubes",
                      price=15, currency="USD", rating=4.4),
        ItemCandidate(item_id="E1", platform="ebay", title="leather toiletry bag",
                      price=30, currency="USD", rating=4.8),
    ]


def test_arm_on_tool_rules() -> None:
    s = HarnessSession()
    arm_on_tool(s, "item_search", {"query": "x"})
    assert s.autopick_armed
    arm_on_tool(s, "item_picker", {})
    assert not s.autopick_armed  # 模型显式精挑 → 解除
    arm_on_tool(s, "task_dispatch", {"subagent_type": "trade"})
    assert not s.autopick_armed  # 交易派发不武装
    arm_on_tool(s, "task_dispatch", {"subagent_type": "search"})
    assert s.autopick_armed


@pytest.mark.asyncio
async def test_autopick_runs_pricing_and_picking_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPICK", "1")
    s = HarnessSession(original_query="旅行收纳三件套")
    s.autopick_armed = True
    with thread_scope("t-autopick", Path(tempfile.mkdtemp()), user_id="u-autopick"):
        set_phase_machine(PhaseStateMachine(Phase.SEARCHING))
        register(_cands())
        await maybe_autopick(s)
        picks = get_last_picks()
        # 控制面状态与模型亲手调等价：两个工具都进 called_tools、picks 计数落 session。
        assert {"price_compare", "item_picker"} <= s.called_tools
        assert s.last_picks == len(picks) > 0
        assert not s.autopick_armed
        # 结果经 inject 通道给模型，且带「无需再调」的指路。
        assert s.pending_inject and "[系统已自动执行]" in s.pending_inject[0]["content"]
        # 阶段机被推到 COMPARING（精挑收尾通告的前提）。
        from app.harness.phase_machine import get_phase_machine

        assert get_phase_machine().phase is not Phase.SEARCHING
        # 未武装时再调是空操作（不会重复精挑、不会重复注入）。
        await maybe_autopick(s)
        assert len(s.pending_inject) == 1


@pytest.mark.asyncio
async def test_autopick_skips_bundle_turn_and_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tools._bundle import BundleSlot, set_session_bundle

    s = HarnessSession()
    s.autopick_armed = True
    with thread_scope("t-autopick-bundle", Path(tempfile.mkdtemp())):
        register(_cands())
        set_session_bundle([BundleSlot(name="包"), BundleSlot(name="杯")])
        await maybe_autopick(s)
        assert "item_picker" not in s.called_tools
        assert s.autopick_armed  # 套装轮不消费武装态，交给 ask_user 入口

    monkeypatch.setenv("AUTOPICK", "0")
    s2 = HarnessSession()
    s2.autopick_armed = True
    with thread_scope("t-autopick-off", Path(tempfile.mkdtemp())):
        register(_cands())
        await maybe_autopick(s2)
        assert "item_picker" not in s2.called_tools
