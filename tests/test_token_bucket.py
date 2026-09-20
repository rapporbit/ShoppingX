"""LLM 令牌桶（阶段 2 第 4 条）的单测：限额解析 / 预估 / 本地退化桶 / 放行口径 / Lua 真行为。

**这组测试分两半，分法是刻意的。** 前半是纯 Python 的取舍（两级限额、工具 schema 要计入预估、
等满即放行、Redis 一坏就退本地），用假客户端钉得死。后半是**双桶原子性与预扣结算**，那些逻辑
全在 Lua 里跑——用假客户端重写一遍 Lua 等于测了个替身，Lua 真写错照样绿。所以后半要真 Redis，
没有就 skip（``BUCKET_TEST_REDIS_URL``，本地起法：
``docker run -d --rm -p 6380:6379 redis:7-alpine``）。
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from app.agent import token_bucket as tb
from app.observability import metrics


@pytest.fixture(autouse=True)
def _clean_bucket(monkeypatch):
    """每个用例开局都是干净的：没有客户端、没有桶、开关打开、限额由用例自己给。"""
    tb.reset_buckets()
    tb.set_client(None)
    monkeypatch.setenv("LLM_BUCKET_ENABLED", "1")
    monkeypatch.delenv("LLM_RPM", raising=False)
    monkeypatch.delenv("LLM_TPM", raising=False)
    yield
    tb.reset_buckets()


def _counter(event: str) -> float:
    return metrics.LLM_BUCKET_EVENTS.labels(event=event)._value.get()


# ── 限额解析 ──────────────────────────────────────────────────────────────────
def test_limits_provider_overrides_global(monkeypatch):
    """按家配的限额盖全局；没盖到的那一维仍落全局。"""
    monkeypatch.setenv("LLM_RPM", "100")
    monkeypatch.setenv("LLM_TPM", "1000")
    monkeypatch.setenv("PROVIDER_DASHSCOPE_BASE_URL", "http://x")  # 前缀配过才算 provider
    monkeypatch.setenv("PROVIDER_DASHSCOPE_RPM", "7")
    limits = tb.limits_for("dashscope/qwen3.8-flash")
    assert (limits.rpm, limits.tpm) == (7, 1000)


def test_limits_model_level_wins(monkeypatch):
    """同一个出口下两个模型配额差 4 倍，只按 provider 配就只能取小的——模型级必须能盖。"""
    monkeypatch.setenv("LLM_TPM", "100")
    monkeypatch.setenv("PROVIDER_DASHSCOPE_BASE_URL", "http://x")
    monkeypatch.setenv("PROVIDER_DASHSCOPE_TPM", "1200000")
    monkeypatch.setenv("MODEL_QWEN3_8_FLASH_TPM", "5000000")
    monkeypatch.setenv("LLM_RPM", "60")
    assert tb.limits_for("dashscope/qwen3.8-flash").tpm == 5_000_000
    assert tb.limits_for("dashscope/deepseek-v4-flash").tpm == 1_200_000
    # 逐维度取：模型只配了 TPM，RPM 照样落回全局
    assert tb.limits_for("dashscope/qwen3.8-flash").rpm == 60


def test_limits_inactive_when_unset():
    assert not tb.limits_for("dashscope/qwen3.8-flash").active


# ── 预估 ─────────────────────────────────────────────────────────────────────
def test_estimate_counts_tool_schema():
    """工具 schema 必须计入：漏掉它预扣会系统性偏小（本仓每轮都发 18 个工具的 schema）。"""
    messages = [{"role": "user", "content": "想买便宜又抗造的旅行三件套"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "item_search",
                "description": "单平台商品检索",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    bare = tb.estimate_prompt_tokens(messages)
    with_tools = tb.estimate_prompt_tokens(messages, tools)
    assert 0 < bare < with_tools


def test_estimate_bad_shape_returns_zero():
    """形状认不出来返回 0，绝不抛——预估是为了少撞 429，不是记账。"""
    assert tb.estimate_prompt_tokens(object()) == 0


def test_estimate_cost_adds_output_reserve(monkeypatch):
    monkeypatch.setenv("LLM_BUCKET_EST_OUTPUT", "500")
    assert tb.estimate_cost_tokens(1200) == 1700


# ── 开关与空转 ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_disabled_returns_zero(monkeypatch):
    monkeypatch.setenv("LLM_BUCKET_ENABLED", "0")
    monkeypatch.setenv("LLM_RPM", "10")
    assert await tb.acquire("dashscope/qwen", 100) == 0


@pytest.mark.asyncio
async def test_no_limits_returns_zero():
    """开关开着但没配限额 = 整层空转（返回 0 即「无需结算」）。"""
    assert await tb.acquire("dashscope/qwen", 100) == 0


# ── 进程内退化桶 ──────────────────────────────────────────────────────────────
def test_local_bucket_rpm_runs_out():
    """RPM=60（1 格/秒）：满桶 60 格连取完，下一格要等约 1s。"""
    bucket = tb._LocalBucket()
    limits = tb.BucketLimits(rpm=60, tpm=0)
    for _ in range(60):
        assert bucket.take(limits, 1) == 0
    wait = bucket.take(limits, 1)
    assert 0 < wait <= 1.0


def test_local_bucket_settle_refunds():
    """预扣 900、真实只用 100 → 回补 800，随后 200 的请求能直接放行。"""
    bucket = tb._LocalBucket()
    limits = tb.BucketLimits(rpm=0, tpm=1000)
    assert bucket.take(limits, 900) == 0
    assert bucket.take(limits, 200) > 0  # 只剩 100，不够
    bucket.settle(limits, 100 - 900)
    assert bucket.take(limits, 200) == 0


def test_local_bucket_settle_can_go_negative():
    """估少了就得扣成负数：欠的那笔留在账上，下一次自然多等，而不是一笔勾销。"""
    bucket = tb._LocalBucket()
    limits = tb.BucketLimits(rpm=0, tpm=1000)
    assert bucket.take(limits, 100) == 0
    bucket.settle(limits, 5000 - 100)  # 真实烧了 5000
    assert bucket.take(limits, 1) > 0


# ── 等满即放行 ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_wait_timeout_passes_through(monkeypatch):
    """等满上限仍无令牌 → **放行**并记 overflow，而不是抛错把用户任务饿死在门口。"""
    monkeypatch.setenv("LLM_RPM", "1")
    monkeypatch.setenv("LLM_BUCKET_WAIT_MAX_SEC", "0.05")
    monkeypatch.setenv("LLM_BUCKET_EST_OUTPUT", "0")
    ref = "dashscope/qwen"
    assert await tb.acquire(ref, 10) == 10  # 满桶那一格
    before = _counter("overflow")
    started = time.monotonic()
    reserved = await tb.acquire(ref, 10)  # 下一格要等 60s，等满 50ms 就放行
    elapsed = time.monotonic() - started
    assert reserved == 10  # 放行也要报预扣值，否则真实用量在结算里凭空消失
    assert elapsed < 1.0
    assert _counter("overflow") == before + 1


# ── Redis 坏掉即退本地 ────────────────────────────────────────────────────────
class _BoomClient:
    """register_script 给出的脚本一调用就炸——模拟 Redis 连不上 / 卡住。"""

    def __init__(self) -> None:
        self.calls = 0

    def register_script(self, script: str):
        async def _run(keys=None, args=None):
            self.calls += 1
            raise RuntimeError("boom")

        return _run


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_local(monkeypatch):
    """Redis 一坏就退进程内桶：调用照常放行，记一次 degraded，且**冷却期内不再碰 Redis**。"""
    monkeypatch.setenv("LLM_RPM", "100")
    monkeypatch.setenv("LLM_BUCKET_DEGRADE_COOLDOWN_SEC", "30")
    client = _BoomClient()
    tb.set_client(client)
    before = _counter("degraded")
    assert await tb.acquire("dashscope/qwen", 10) > 0
    assert _counter("degraded") == before + 1
    assert client.calls == 1
    assert await tb.acquire("dashscope/qwen", 10) > 0
    assert client.calls == 1  # 冷却期内直接走本地，不为每次调用再赔一个超时


# ── Lua 真行为（要真 Redis）────────────────────────────────────────────────────
REDIS_URL = os.environ.get("BUCKET_TEST_REDIS_URL", "redis://localhost:6380/9")


def _probe_redis() -> bool:
    async def go() -> bool:
        import redis.asyncio as aredis

        client = aredis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        try:
            await client.ping()
            return True
        except Exception:
            return False
        finally:
            await client.aclose()

    try:
        return asyncio.run(go())
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not _probe_redis(), reason=f"需要真 Redis（{REDIS_URL}）来验 Lua 的双桶原子性"
)


async def _fresh_client():
    import redis.asyncio as aredis

    client = aredis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    tb.set_client(client)
    return client


@requires_redis
@pytest.mark.asyncio
async def test_lua_rpm_exhausts_then_waits():
    client = await _fresh_client()
    try:
        bucket = tb.get_bucket("dashscope/qwen")
        limits = tb.BucketLimits(rpm=3, tpm=0)
        assert [await bucket.take(limits, 1) for _ in range(3)] == [0, 0, 0]
        wait = await bucket.take(limits, 1)
        assert 0 < wait <= 20  # 3 RPM = 一格 20s
    finally:
        await client.aclose()


@requires_redis
@pytest.mark.asyncio
async def test_lua_both_buckets_or_neither():
    """TPM 不够时 RPM 那一格**不能**被扣走——漏格不报错，只会表现成「桶没满却老是等」。"""
    client = await _fresh_client()
    try:
        bucket = tb.get_bucket("dashscope/qwen")
        limits = tb.BucketLimits(rpm=100, tpm=100)
        assert await bucket.take(limits, 100) == 0  # TPM 一次抽干
        rpm_left = float(await client.hget(f"{tb.KEY_PREFIX}dashscope/qwen:r", "tokens"))
        assert await bucket.take(limits, 100) > 0  # 这次 TPM 不够
        assert float(await client.hget(f"{tb.KEY_PREFIX}dashscope/qwen:r", "tokens")) == rpm_left
    finally:
        await client.aclose()


@requires_redis
@pytest.mark.asyncio
async def test_lua_settle_refunds_and_overdraws():
    client = await _fresh_client()
    try:
        bucket = tb.get_bucket("dashscope/qwen")
        limits = tb.BucketLimits(rpm=0, tpm=1000)
        assert await bucket.take(limits, 900) == 0
        assert await bucket.take(limits, 200) > 0
        await bucket.settle(limits, 100 - 900)  # 真实只烧了 100
        assert await bucket.take(limits, 200) == 0
        await bucket.settle(limits, 9000)  # 估少了一大笔：允许扣成负数
        assert float(await client.hget(f"{tb.KEY_PREFIX}dashscope/qwen:t", "tokens")) < 0
    finally:
        await client.aclose()


@requires_redis
@pytest.mark.asyncio
async def test_lua_shared_across_clients():
    """两条独立连接（= 两个副本）打同一个桶，总放行数受同一条 RPM 线约束。"""
    client = await _fresh_client()
    try:
        import redis.asyncio as aredis

        other = aredis.from_url(REDIS_URL, decode_responses=True)
        limits = tb.BucketLimits(rpm=5, tpm=0)
        bucket_a = tb.get_bucket("dashscope/qwen")
        tb.set_client(other)
        bucket_b = tb.TokenBucket("dashscope/qwen")
        passed = 0
        for turn in range(8):
            tb.set_client(client if turn % 2 == 0 else other)
            target = bucket_a if turn % 2 == 0 else bucket_b
            if await target.take(limits, 1) == 0:
                passed += 1
        assert passed == 5  # 不是 10：两个「副本」共用同一条线
        await other.aclose()
    finally:
        await client.aclose()


@requires_redis
@pytest.mark.asyncio
async def test_lua_cost_clamped_to_capacity():
    """单次预扣超过整桶容量时钳到容量——否则那条请求永远等不到令牌。"""
    client = await _fresh_client()
    try:
        bucket = tb.get_bucket("dashscope/qwen")
        limits = tb.BucketLimits(rpm=0, tpm=100)
        assert await bucket.take(limits, 10_000) == 0
    finally:
        await client.aclose()
