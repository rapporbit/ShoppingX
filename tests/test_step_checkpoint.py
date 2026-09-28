"""按步检查点（``app.agent.checkpoint``）：存得下、读得回、对不上就放弃。"""

import zlib
from pathlib import Path
from typing import Any

import pytest

from app.agent import checkpoint
from app.api.run_state import run_slot
from app.harness.retrieval_budget import _RunRetrieval
from app.harness.session import HarnessSession
from app.utils.thread_ctx import thread_scope
from tests.conftest import FakeRedis
from tests.test_orchestrator import _user


class _FakeState:
    cur_iter = 2

    def model_dump_json(self) -> str:
        return '{"context": []}'


class _FakeAgent:
    state = _FakeState()


@pytest.fixture
def redis() -> FakeRedis:
    client = FakeRedis()
    checkpoint.set_client(client)
    return client


def _session() -> HarnessSession:
    s = HarnessSession(original_query="旅行三件套")
    s.prefilled = True
    s.planner_done = True
    s.autopick_armed = True
    s.called_tools.add("item_search")
    s.guard.detector.record("item_search")
    return s


async def test_roundtrip(redis: FakeRedis, tmp_path: Path) -> None:
    with thread_scope("t1", tmp_path, run_id="r1"):
        slot = run_slot(_RunRetrieval)
        assert slot is not None
        slot.count = 3
        await checkpoint.save("r1", _FakeAgent(), _session(), turn_start=5)

    ck = await checkpoint.load("r1")
    assert ck is not None
    assert (ck.cur_iter, ck.turn_start, ck.state_json) == (2, 5, '{"context": []}')
    assert ck.run_slots[_RunRetrieval].count == 3  # type: ignore[attr-defined]
    h = ck.harness
    assert (h.prefilled, h.planner_done, h.autopick_armed) == (True, True, True)
    assert h.called_tools == {"item_search"}
    assert len(h.guard.detector._recent) == 1

    await checkpoint.discard("r1")
    assert await checkpoint.load("r1") is None


async def test_shape_drift_is_rejected(redis: FakeRedis, tmp_path: Path) -> None:
    """模拟发版前留下的检查点：少一个属性就整份作废，别让续跑半路 AttributeError。"""
    s = _session()
    del s.autopick_armed
    with thread_scope("t1", tmp_path, run_id="r1"):
        await checkpoint.save("r1", _FakeAgent(), s, turn_start=0)
    assert await checkpoint.load("r1") is None


async def test_corrupt_blob_is_ignored(redis: FakeRedis) -> None:
    await redis.set(f"{checkpoint.KEY_PREFIX}r1", zlib.compress(b"not a pickle"))
    assert await checkpoint.load("r1") is None


async def test_disabled_or_no_run_id_is_noop(tmp_path: Path) -> None:
    client = FakeRedis()
    checkpoint.set_client(None)
    await checkpoint.save("r1", _FakeAgent(), _session(), turn_start=0)
    checkpoint.set_client(client)
    await checkpoint.save("", _FakeAgent(), _session(), turn_start=0)
    assert client.store == {}


async def test_assembled_agent_writes_checkpoint(
    redis: FakeRedis, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真装配的主 Agent：一次模型调用前后各写一次，读回来的 state 能反序列化。"""
    from agentscope.state import AgentState

    from app.agent import agents as ag
    from app.agent.events import pump_events
    from app.harness.adapter import HarnessAgentAdapter
    from app.harness.setup import setup_harness
    from tests.test_orchestrator import _fake_model

    async def _no_prefill(self: Any, agent: Any) -> None:
        return None

    setup_harness()
    model = _fake_model()
    monkeypatch.setattr(HarnessAgentAdapter, "_prefill", _no_prefill)
    monkeypatch.setattr(ag, "get_tier_llm", lambda _tier: model)
    monkeypatch.setattr("app.agent.llm.get_llm", lambda: model)
    monkeypatch.setattr("app.agent.llm.get_fast_llm", lambda: model)

    writes: list[str] = []
    real_set = redis.set

    async def _spy(key: str, value: Any, **kw: Any) -> object:
        writes.append(key)
        return await real_set(key, value, **kw)

    redis.set = _spy  # type: ignore[method-assign]
    with thread_scope("t1", tmp_path, run_id="r1"):
        agent, _ = await ag.build_main_agent(original_query="你好")
        await pump_events(agent.reply_stream(_user("你好"), yield_final_msg=True))

    assert len(writes) >= 2 and set(writes) == {f"{checkpoint.KEY_PREFIX}r1"}
    ck = await checkpoint.load("r1")
    assert ck is not None
    restored = AgentState.model_validate_json(ck.state_json)
    assert any(m.role == "user" for m in restored.context)
