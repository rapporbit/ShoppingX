"""D 块 · 事件回放的确定性单测：Redis Stream 持久化 + 重连补发缺口 + 降级。

用一个内存 fake redis（实现 xadd/xrange 的最小子集，支持 "(id" 排他下界）注入 event_log，
不依赖真 Redis / fakeredis 包，也能断言「补发的是 last_event_id 之后的缺口」与降级路径。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.api import event_log


def _cmp_id(sid: str) -> tuple[int, int]:
    a, b = sid.split("-")
    return int(a), int(b)


class FakeRedis:
    """最小 redis Stream 子集：xadd 单调发号、xrange 支持 "(id" 排他下界 + maxlen 裁剪。"""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.counter = 0
        self.fail_times = 0  # >0 时前 N 次操作抛错（测断路器降级）
        self.expires: dict[str, int] = {}  # key → TTL（秒），断言 append 刷新了过期

    async def xadd(
        self, key: str, fields: dict[str, Any], maxlen: int | None = None, approximate: bool = True
    ) -> str:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("fake redis down")
        self.counter += 1
        sid = f"{self.counter}-0"
        self.streams.setdefault(key, []).append((sid, fields))
        if maxlen:
            self.streams[key] = self.streams[key][-maxlen:]
        return sid

    async def expire(self, key: str, ttl: int) -> bool:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("fake redis down")
        self.expires[key] = ttl
        return True

    async def xrange(self, key: str, min: str = "-", max: str = "+") -> list[tuple[str, dict]]:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("fake redis down")
        entries = self.streams.get(key, [])
        if min.startswith("("):
            lo = _cmp_id(min[1:])
            return [(s, f) for s, f in entries if _cmp_id(s) > lo]
        return entries


@pytest.fixture(autouse=True)
def _fresh_event_log() -> Any:
    """每个用例注入新的 fake redis（顺带 reset 断路器）；用完复位懒加载。"""
    fake = FakeRedis()
    event_log.set_client(fake)
    yield fake
    event_log.set_client(None)


async def test_append_returns_stream_id(_fresh_event_log: FakeRedis) -> None:
    sid = await event_log.append("t1", {"event": "tool_start"})
    assert sid is not None
    assert sid.endswith("-0")


async def test_replay_returns_only_gap_after_last_id(_fresh_event_log: FakeRedis) -> None:
    id1 = await event_log.append("t1", {"event": "a"})
    id2 = await event_log.append("t1", {"event": "b"})
    id3 = await event_log.append("t1", {"event": "c"})
    assert id1 and id2 and id3

    # 前端「最后收到 id2」→ 只补发 id2 之后的（即 c），且每条回填 stream id。
    gap = await event_log.replay_after("t1", id2)
    assert [e["event"] for e in gap] == ["c"]
    assert gap[0]["id"] == id3


async def test_append_sets_ttl(_fresh_event_log: FakeRedis) -> None:
    await event_log.append("t1", {"event": "a"})
    # 每次 append 刷新 TTL（滑动过期），防止流的数量随会话无界累积。
    assert _fresh_event_log.expires.get(event_log._key("t1")) == event_log._TTL


async def test_replay_empty_when_caught_up(_fresh_event_log: FakeRedis) -> None:
    id1 = await event_log.append("t1", {"event": "a"})
    assert id1
    assert await event_log.replay_after("t1", id1) == []  # 已是最新，无缺口


async def test_replay_empty_last_id_returns_nothing(_fresh_event_log: FakeRedis) -> None:
    await event_log.append("t1", {"event": "a"})
    assert await event_log.replay_after("t1", "") == []  # 无 last_event_id → 不补发


async def test_streams_isolated_per_thread(_fresh_event_log: FakeRedis) -> None:
    await event_log.append("t1", {"event": "a"})
    idb = await event_log.append("t2", {"event": "b"})
    assert idb
    # t2 的补发不串到 t1。
    gap_t1 = await event_log.replay_after("t1", "0-0")
    assert [e["event"] for e in gap_t1] == ["a"]


async def test_replay_current_run_slices_from_last_session_created(
    _fresh_event_log: FakeRedis,
) -> None:
    """多轮事件首尾相接时，「当前这轮」= 最后一个 session_created 起——历史轮已落 turns.json，
    续看只取尚未收尾的最新一轮，避免与历史轮重出。"""
    # 第一轮（已收尾）。
    await event_log.append("t1", {"event": "session_created"})
    await event_log.append("t1", {"event": "tool_start"})
    await event_log.append("t1", {"event": "task_result"})
    # 第二轮（在跑）。
    await event_log.append("t1", {"event": "session_created"})
    last = await event_log.append("t1", {"event": "tool_start"})
    assert last

    run = await event_log.replay_current_run("t1")
    assert [e["event"] for e in run] == ["session_created", "tool_start"]
    assert run[-1]["id"] == last  # 每条回填 stream id，前端据此去重 + 记断点


async def test_replay_current_run_no_session_created_returns_all(
    _fresh_event_log: FakeRedis,
) -> None:
    """流里没有 session_created（极长一轮把开头挤掉的退化情形）：退回从流首开始，不丢现有事件。"""
    await event_log.append("t1", {"event": "tool_start"})
    await event_log.append("t1", {"event": "tool_end"})
    run = await event_log.replay_current_run("t1")
    assert [e["event"] for e in run] == ["tool_start", "tool_end"]


async def test_replay_current_run_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVENT_REPLAY", "false")
    event_log.set_client(None)
    assert await event_log.replay_current_run("t1") == []  # 关闭：无回放


async def test_disabled_degrades_to_no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVENT_REPLAY", "false")
    event_log.set_client(None)  # 复位，让 _get_client 重新判定 enabled
    assert await event_log.append("t1", {"event": "a"}) is None  # 关闭：不持久化
    assert await event_log.replay_after("t1", "1-0") == []  # 关闭：不补发


async def test_circuit_breaker_degrades_on_redis_failure(_fresh_event_log: FakeRedis) -> None:
    _fresh_event_log.fail_times = 10  # 让 redis 持续失败
    # append 失败不抛、返回 None（降级），不拖垮主链路。
    for _ in range(5):
        assert await event_log.append("t1", {"event": "x"}) is None
    # 断路器应已熔断（连续失败到阈值）。
    assert event_log._breaker.state == "open"
