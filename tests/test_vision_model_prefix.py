"""直连出口也要认 ``provider/`` 前缀（修 LLM_VISION 带前缀必 404）。

线上现象：``LLM_VISION=dashscope/qwen3.5-flash`` 时 ``image_understand`` 每次都降级，
服务商原话是 ``The model 'dashscope/qwen3.5-flash' does not exist``。阶段 2-1 把前缀解析
放进了 Router 那条路，而视觉档恒走直连、``LLM_PROVIDER_ROUTER=0`` 回退时所有档也走直连——
直连不剥前缀，于是整串模型名被发给了服务商。

所以这组测试盯死的是**发出去的 body 里的 model 字段**和**打到哪个 base_url**，
自己当服务端（同 ``tests/test_provider_routing.py``）才看得见这两样。
"""

import json
import os
from typing import Any

import pytest
from aiohttp import web

SEEN: list[tuple[str, dict[str, Any]]] = []


def _sse(delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-t",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _ok(request: web.Request) -> web.StreamResponse:
    SEEN.append((request.path, await request.json()))
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(_sse({"role": "assistant", "content": "好"}))
    await resp.write(_sse({}, finish="stop"))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


@pytest.fixture
async def mock_api() -> Any:
    """两个 OpenAI 兼容端点：``/v1`` 当默认家，``/ds/v1`` 当 dashscope 家。"""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _ok)
    app.router.add_post("/ds/v1/chat/completions", _ok)
    app.router.add_post("/vis/v1/chat/completions", _ok)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    SEEN.clear()
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """本机 .env 里可能已经配了别的出口，先清干净。"""
    for key in list(os.environ):
        if key.startswith(("PROVIDER_", "VISION_")) or key in {
            "LLM_FALLBACK_CHAIN",
            "LLM_PROVIDER_ROUTER",
            "LLM_VISION",
            "COMPRESS_CACHE_CONTROL",
        }:
            monkeypatch.delenv(key, raising=False)


async def _drain(model: Any) -> None:
    from agentscope.message import Msg, TextBlock

    msgs = [Msg(name="user", role="user", content=[TextBlock(type="text", text="图里是啥")])]
    async for _chunk in await model(msgs):
        pass


def _sent() -> tuple[str, str]:
    """最后一次请求的（路径, model 字段）。"""
    path, body = SEEN[-1]
    return path, body["model"]


def _setup(base: str, monkeypatch: pytest.MonkeyPatch, *, dashscope: bool = True) -> None:
    from app.agent import llm

    monkeypatch.setenv("OPENAI_BASE_URL", f"{base}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default")
    if dashscope:
        monkeypatch.setenv("PROVIDER_DASHSCOPE_BASE_URL", f"{base}/ds/v1")
        monkeypatch.setenv("PROVIDER_DASHSCOPE_API_KEY", "sk-ds")
    llm._load_params()
    SEEN.clear()


class TestVisionPrefix:
    """视觉档恒走直连，所以它是这个 bug 唯一的线上表现。"""

    async def test_带前缀的看图模型发出去时前缀已剥(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """线上 404 的那条：发出去的必须是 ``qwen3.5-flash``，不是整串。"""
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_VISION", "dashscope/qwen3.5-flash")
        await _drain(llm.get_vision_llm())
        assert _sent() == ("/ds/v1/chat/completions", "qwen3.5-flash")

    async def test_不带前缀的看图模型逐字不变(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """向后兼容的底线：老写法照旧打 ``OPENAI_BASE_URL``、模型名原样。"""
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_VISION", "qwen3.5-flash")
        await _drain(llm.get_vision_llm())
        assert _sent() == ("/v1/chat/completions", "qwen3.5-flash")

    async def test_模型名自带斜杠不被吃掉一截(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """siliconflow 那种 ``Qwen/Qwen3-8B``：第一段没配过出口就整串是模型名。"""
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_VISION", "Qwen/Qwen3-VL-8B")
        await _drain(llm.get_vision_llm())
        assert _sent() == ("/v1/chat/completions", "Qwen/Qwen3-VL-8B")

    async def test_VISION_出口配全时压过前缀指向的那家(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """现有语义不变：``VISION_*`` 配了就继续优先；前缀这时只负责剥模型名。"""
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_VISION", "dashscope/qwen3.5-flash")
        monkeypatch.setenv("VISION_BASE_URL", f"{mock_api}/vis/v1")
        monkeypatch.setenv("VISION_API_KEY", "sk-vis")
        await _drain(llm.get_vision_llm())
        assert _sent() == ("/vis/v1/chat/completions", "qwen3.5-flash")


class TestRouterOffPrefix:
    """``LLM_PROVIDER_ROUTER=0`` 回退直连时，所有档位同样会踩这个 404。"""

    async def test_关掉_router_后主档带前缀也能剥(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER_ROUTER", "0")
        model = llm.build_model(
            "dashscope/deepseek-v4.1-flash", temperature=0.3, role="main", thinking=False
        )
        await _drain(model)
        assert _sent() == ("/ds/v1/chat/completions", "deepseek-v4.1-flash")

    def test_断路器与令牌桶的键仍是带前缀的_ref(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``self.model`` 是断路器 / 令牌桶的键（gateway 里写死的口径：两条出口路上都是
        ``provider/model``）。剥前缀只能发生在发请求那一刻，否则关掉 Router 会顺带把
        ``PROVIDER_<NAME>_RPM`` 这类按家配的限额静默停掉。"""
        from app.agent import llm

        _setup(mock_api, monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER_ROUTER", "0")
        model = llm.build_model("dashscope/m1", temperature=0.3, role="main", thinking=False)
        assert model.model == "dashscope/m1"
