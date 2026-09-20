"""本轮 deadline 的下传（阶段 4-2）。

主 loop 外面那层 ``asyncio.timeout(MAIN_AGENT_TIMEOUT_SEC)`` 是一刀从外面砍下来的：出站点对它
一无所知，只能按自己的超时傻等，于是出现「主 loop 只剩 3 秒，这次检索照样按 5 秒等下去」——那 5 秒
注定等不到能被用上的结果。deadline 把同一个预算下传到每个出站点，让「再等也没意义了」当场生效。

所以本文件钉三件事：``clamp_timeout`` 只收不放、剩余见底时给的是**正数**下限而不是 0、以及各出站
点真的把收紧后的值带到了请求上（LLM 走 kwargs、Qdrant 走整秒参数）。
"""

import asyncio

import pytest
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel

from app.agent.gateway import GatewayThrottle, ThrottledChatModel
from app.api.context import (
    clamp_timeout,
    remaining_seconds,
    reset_deadline,
    set_deadline,
)
from app.recall import qdrant_store


@pytest.fixture(autouse=True)
def _no_leftover_deadline():
    """每个用例前后都清干净——deadline 是进程内 ContextVar，漏一个会让后面的用例莫名其妙变短。"""
    reset_deadline()
    yield
    reset_deadline()


def test_without_a_deadline_timeouts_pass_through() -> None:
    """没有 deadline 作用域（离线脚本 / 工具单测）时原样返回：这道闸不该凭空改变既有行为。"""
    assert remaining_seconds() is None
    assert clamp_timeout(5.0) == 5.0


def test_clamp_only_tightens() -> None:
    """只收不放：出站点自己的超时比剩余还短时，用它自己的——那是「对面正常响应要多久」的知识。"""
    set_deadline(2.0)
    assert clamp_timeout(1.0) == 1.0
    assert 1.5 < clamp_timeout(5.0) <= 2.0


def test_exhausted_budget_still_yields_a_positive_timeout() -> None:
    """预算见底给 50ms 而不是 0：0 在 httpx 是「无限等」、在 Qdrant 是参数错误，两种都更难查。"""
    set_deadline(-1.0)
    assert clamp_timeout(5.0) == pytest.approx(0.05)


def test_switch_off_restores_per_call_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DEADLINE_ENABLED=0`` 是回滚开关：关掉之后各出站点照自己的超时来。"""
    monkeypatch.setenv("DEADLINE_ENABLED", "0")
    set_deadline(0.5)
    assert clamp_timeout(5.0) == 5.0


async def test_child_task_inherits_the_deadline() -> None:
    """工具跑在各自的 context 里，靠 ContextVar 的快照继承读到入口设的 deadline。

    这是整个设计的支点：``run_agent`` 入口写一次，下游只读。哪天这条断了，deadline 会**静默**
    退化成「谁都没有 deadline」——不报错，只是又开始白等。
    """
    set_deadline(3.0)

    async def _child() -> float:
        return clamp_timeout(10.0)

    assert 2.5 < await asyncio.create_task(_child()) <= 3.0


async def test_to_thread_inherits_the_deadline() -> None:
    """Qdrant 那条路是同步客户端跑在 ``asyncio.to_thread`` 里——它同样带着 context 快照。"""
    set_deadline(3.0)
    assert 2.5 < await asyncio.to_thread(clamp_timeout, 10.0) <= 3.0


def test_qdrant_timeout_is_whole_seconds_and_at_least_one() -> None:
    """Qdrant 协议只认整秒：向上取整，且不许压到 0——0 在那边是「不限时」，等于把闸开到最大。"""
    set_deadline(1.2)
    assert qdrant_store._query_timeout() == 2
    set_deadline(-5.0)
    assert qdrant_store._query_timeout() == 1


async def test_llm_request_carries_the_clamped_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 出口：超时以**每次请求**的入参下去。

    放在 ``_call_api`` 这层是因为它同时覆盖两条出口路——直连那条 openai SDK 认这个 kwarg（盖过
    建客户端时的默认值），Router 那条 ``_RouterCompletions.create`` 是 ``setdefault``，我们给了
    它就不再塞自己那份。
    """
    seen: dict[str, object] = {}

    async def _capture(self: object, *args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(OpenAIChatModel, "_call_api", _capture)
    model = ThrottledChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="test-model",
        stream=False,
        max_retries=0,
        throttle=GatewayThrottle(),
        client_kwargs={"timeout": 60.0},
    )

    set_deadline(4.0)
    assert await model._call_api("test-model", []) == "ok"
    assert 3.5 < float(seen["timeout"]) <= 4.0  # type: ignore[arg-type]
