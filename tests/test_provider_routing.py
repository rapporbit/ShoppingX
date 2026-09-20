"""阶段 2 第 2 条：provider / model 寻址 + LiteLLM Router 出口。

这组测试要守住的不是「能调通」，而是**两条路发出去的 body 逐字一样**——Router 是换出口，
不是换协议。任何一次框架/依赖升级把 cache_control 标记裁了、把 extra_body 摊平的方式改了，
这里的逐字比对就会红，而线上的表现只是「缓存命中率悄悄掉了」，没人会发现。
"""

import json
import os
from typing import Any

import pytest
from aiohttp import web  # litellm 的依赖，测试里借来当 mock 端点

from app.agent.providers import (
    Endpoint,
    build_model_list,
    fallback_chain,
    parse_model_ref,
    resolve_endpoint,
    router_enabled,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个用例从干净的出口配置开始（本机 .env 里可能已经配了别的 provider）。"""
    for key in list(os.environ):
        if key.startswith("PROVIDER_") or key in {"LLM_FALLBACK_CHAIN", "LLM_PROVIDER_ROUTER"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://default.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default")


class TestModelRef:
    def test_不带前缀的模型名走默认出口(self) -> None:
        """向后兼容的底线：老 .env 一个字不改也能跑。"""
        assert parse_model_ref("qwen3.8-flash") == ("default", "qwen3.8-flash")
        ep = resolve_endpoint("qwen3.8-flash")
        assert (ep.provider, ep.base_url, ep.api_key) == (
            "default",
            "https://default.example/v1",
            "sk-default",
        )
        assert ep.ref == "qwen3.8-flash"  # default 家不回填前缀

    def test_配过的前缀才当前缀(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROVIDER_DASHSCOPE_BASE_URL", "https://ds.example/v1")
        monkeypatch.setenv("PROVIDER_DASHSCOPE_API_KEY", "sk-ds")
        assert parse_model_ref("dashscope/qwen3.8-flash") == ("dashscope", "qwen3.8-flash")
        ep = resolve_endpoint("dashscope/qwen3.8-flash")
        assert (ep.base_url, ep.api_key, ep.ref) == (
            "https://ds.example/v1",
            "sk-ds",
            "dashscope/qwen3.8-flash",
        )

    def test_没配过的前缀不吃掉模型名(self) -> None:
        """siliconflow 的模型名本身带斜杠（``Qwen/Qwen3-8B``）。

        按字符串形状切，``Qwen`` 会被当成 provider、模型名被吃掉一截，请求打到默认出口上
        却只报 ``Qwen3-8B`` —— 网关回 404，而日志里看着像配置对了。
        """
        assert parse_model_ref("Qwen/Qwen3-8B") == ("default", "Qwen/Qwen3-8B")
        assert resolve_endpoint("Qwen/Qwen3-8B").model == "Qwen/Qwen3-8B"

    def test_provider_配了_url_没配_key_就回落主_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROVIDER_ACME_BASE_URL", "https://acme.example/v1")
        assert resolve_endpoint("acme/m1").api_key == "sk-default"

    def test_fallback_链按逗号切且保序(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", " a/b , c/d ,, ")
        assert fallback_chain() == ["a/b", "c/d"]
        monkeypatch.delenv("LLM_FALLBACK_CHAIN")
        assert fallback_chain() == []

    def test_router_默认开可用开关关掉(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert router_enabled() is True
        monkeypatch.setenv("LLM_PROVIDER_ROUTER", "0")
        assert router_enabled() is False


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
    usage = {
        "id": "chatcmpl-t",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _down(request: web.Request) -> web.Response:
    SEEN.append((request.path, await request.json()))
    return web.json_response({"error": {"message": "down"}}, status=500)


@pytest.fixture
async def mock_api() -> Any:
    """本地 OpenAI 兼容端点：``/v1`` 正常、``/down`` 恒 500，并把收到的 body 原样留下。

    自己当服务端是这组测试的关键——打别人的端点只看得到回复，看不见「我们发出去的 body
    被谁裁掉了什么」，而这里要守的恰恰是 body 的逐字形态。
    """
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _ok)
    app.router.add_post("/down/chat/completions", _down)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    SEEN.clear()
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


async def _one_call(base: str, routed: bool, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """用同一条 ``build_model`` 跑一轮，返回它实际发出去的 body。"""
    from agentscope.message import Msg, TextBlock

    from app.agent import llm

    monkeypatch.setenv("OPENAI_BASE_URL", f"{base}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("COMPRESS_CACHE_CONTROL", "1")
    monkeypatch.setenv("LLM_PROVIDER_ROUTER", "1" if routed else "0")
    llm._load_params()

    model = llm.build_model("m", temperature=0.3, role="main", thinking=False)
    msgs = [
        # system 要够长，否则达不到 formatter 的最小写入阈值、根本不会打 cache_control。
        Msg(
            name="system",
            role="system",
            content=[TextBlock(type="text", text="你是购物助手。" * 600)],
        ),
        Msg(name="user", role="user", content=[TextBlock(type="text", text="找个包")]),
    ]
    SEEN.clear()
    last = None
    async for chunk in await model(msgs):
        last = chunk
    return {"body": SEEN[-1][1], "last": last}


def test_model_list_按_ref_去重() -> None:
    """主出口与 fallback 撞上同一个 ref 时只留一条。

    Router 的 ``model_list`` 里同名 deployment 会被当成「同一模型的多个副本」参与负载均衡——
    把主出口重复登记一遍，等于让「fallback」变成「有一半概率直接打到同一个坏出口」。
    """
    a = Endpoint("p1", "m1", "https://a/v1", "k1")
    b = Endpoint("p2", "m2", "https://b/v1", "k2")
    rows = build_model_list(a, [b, a])
    assert [r["model_name"] for r in rows] == ["p1/m1", "p2/m2"]
    assert rows[0]["litellm_params"]["model"] == "openai/m1"
    assert rows[0]["litellm_params"]["api_base"] == "https://a/v1"


class TestSameBytesOnTheWire:
    """Router 是换出口，不是换协议——所以逐字比。"""

    async def test_两条路发出的_body_逐字一致(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        routed = await _one_call(mock_api, True, monkeypatch)
        direct = await _one_call(mock_api, False, monkeypatch)
        assert routed["body"] == direct["body"]
        # 顺带钉死两个最容易被中间层吃掉的字段（它们一旦没了，线上只表现为「缓存不命中」
        # 和「思考 token 白烧」，不会报错）。
        assert routed["body"]["enable_thinking"] is False
        sys_content = routed["body"]["messages"][0]["content"]
        assert sys_content[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_usage_用上游给的不用中间层估的(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """credit 结算按 usage 走，所以这个数必须是网关报的那份。

        litellm 在**上游不给** usage 时会自己按 token 数补算一份顶上（实测过，估出来的
        input 比真值大两个数量级）。上游给了就必须透传原值——否则换个出口，账就换了口径。
        """
        routed = await _one_call(mock_api, True, monkeypatch)
        direct = await _one_call(mock_api, False, monkeypatch)
        assert routed["last"].usage.input_tokens == 11
        assert routed["last"].usage.output_tokens == 7
        assert direct["last"].usage.input_tokens == routed["last"].usage.input_tokens


class TestFallbackChain:
    async def test_主出口_500_时切到链上的下一家(
        self, mock_api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """并且要确认**真的先打了坏的**——不然可能只是 Router 自己短路，没验证到切换。"""
        from app.agent import llm

        monkeypatch.setenv("PROVIDER_BAD_BASE_URL", f"{mock_api}/down")
        monkeypatch.setenv("PROVIDER_BAD_API_KEY", "sk-bad")
        monkeypatch.setenv("PROVIDER_GOOD_BASE_URL", f"{mock_api}/v1")
        monkeypatch.setenv("PROVIDER_GOOD_API_KEY", "sk-good")
        monkeypatch.setenv("OPENAI_BASE_URL", f"{mock_api}/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_PROVIDER_ROUTER", "1")
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "good/m")
        llm._load_params()

        from agentscope.message import Msg, TextBlock

        model = llm.build_model("bad/m", temperature=0.3, role="main", thinking=False)
        SEEN.clear()
        last = None
        async for chunk in await model(
            [Msg(name="user", role="user", content=[TextBlock(type="text", text="嗨")])]
        ):
            last = chunk
        paths = [p for p, _ in SEEN]
        assert any(p.startswith("/down") for p in paths), f"没打到坏出口：{paths}"
        assert any(p.startswith("/v1") for p in paths), f"没切到好出口：{paths}"
        assert last is not None and last.content

    async def test_只有主档吃_fallback_链(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """链路外的一次性调用（planner / judge…）暂不跨家——等能力矩阵门落地再放开。"""
        from app.agent.llm import _fallback_refs

        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "a/b")
        assert _fallback_refs("main") == ["a/b"]
        assert _fallback_refs("planner") == []
        assert _fallback_refs("judge") == []

    def test_配了链就不再叠加老的_fallback_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """两套都挂着的话，同一次失败会打出两轮额外请求，日志里还看不出是谁切的。"""
        from app.agent import llm

        monkeypatch.setenv("LLM_PROVIDER_ROUTER", "1")
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "a/b")
        monkeypatch.setenv("PROVIDER_A_BASE_URL", "https://a.example/v1")
        llm._load_params()
        assert llm.get_model_config().fallback_model is None
