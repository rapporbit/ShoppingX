"""阶段 5 spike：量 LiteLLM 当 provider 寻址层的四条判据 + 产 provider 能力矩阵。

判据与背景见 `docs/plans/spike-litellm-2026-09-20.md`（判据写在那份文档开头，先于本脚本落纸）。

两段：
- 默认（不花钱）：起一个本地 OpenAI 兼容 mock 端点，**把收到的 body 原样留下**，据此判
  P1 cache_control 透传 / P2 extra_body enable_thinking / P3 流式 tool call / P4 额外延迟。
  只有自己当服务端才看得见「发出去的 body 被裁掉了什么」——打真 provider 只能看到它的回复。
- `--live`：对真 provider 打少量小请求，产出阶段 2 第 3 条的能力矩阵。这段判的是 provider，
  不是 LiteLLM。

跑法（阶段 2 落地后 litellm 已进主依赖，不再需要 ``--group spike``）::

    uv run python scripts/spike_litellm.py          # mock 段
    uv run python scripts/spike_litellm.py --live   # 加跑能力矩阵
"""

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import time
from typing import Any

from aiohttp import web

# ── mock OpenAI 兼容端点 ─────────────────────────────────────────────────────

RECEIVED: list[dict[str, Any]] = []
# 与 RECEIVED 并行记录每条请求打到了哪个 path（Router 的 deployment 分布靠它看）。
# 不塞进 body 里是刻意的：body 要逐字比对，多一个键会污染 `_sent_keys` 那几行输出。
SEEN: list[str] = []

_NON_STREAM = {
    "id": "chatcmpl-spike",
    "object": "chat.completion",
    "created": 0,
    "model": "spike-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

# 流式 tool call 的增量形态照 OpenAI 规范切：第一片带 id/name、后续只带 arguments 片段。
# 故意把 arguments 切在 JSON 中间（`{"qu` / `ery": "b` / `ag"}`），这正是拼接会出错的地方。
_STREAM_TOOL_DELTAS = [
    {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "item_search", "arguments": ""}}]},
    {"tool_calls": [{"index": 0, "function": {"arguments": '{"qu'}}]},
    {"tool_calls": [{"index": 0, "function": {"arguments": 'ery": "b'}}]},
    {"tool_calls": [{"index": 0, "function": {"arguments": 'ag"}'}}]},
]


def _sse_chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-spike",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "spike-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _handler(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    RECEIVED.append(body)
    SEEN.append(request.path)
    if not body.get("stream"):
        return web.json_response(_NON_STREAM)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(_sse_chunk({"role": "assistant", "content": ""}))
    for delta in _STREAM_TOOL_DELTAS:
        await resp.write(_sse_chunk(delta))
    await resp.write(_sse_chunk({}, finish="tool_calls"))
    # include_usage 时真 OpenAI 会在最后补一片「没有 choices、只有 usage」的 chunk。
    # 少了它就看不出「谁报的 usage」——litellm 在上游不给时会自己按 token 数补算一份，
    # 而 credit 结算是按 usage 走的，两者混为一谈会让计费悄悄换了口径。
    usage_chunk = {
        "id": "chatcmpl-spike",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "spike-model",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    await resp.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _handler_fail(request: web.Request) -> web.Response:
    """恒 500 的端点：给 Router fallback 用（5xx 才是断路器与 fallback 该接的那类错）。"""
    RECEIVED.append(await request.json())
    SEEN.append(request.path)
    return web.json_response({"error": {"message": "spike: upstream down"}}, status=500)


async def _handler_half(request: web.Request) -> web.StreamResponse:
    """**流到一半断**：200 + 两片 tool call 增量之后直接掐掉连接。

    这是 spike 记录里点名「没测」的那条线——200 已经回了、fallback 的窗口早关上，坏消息发生在
    流中途。要问的不是「能不能切」（切不了），而是「客户端拿到的是异常还是半截 JSON」：
    静默返回半截 arguments 会让上层拼出 ``{"qu`` 这种废字符串当成功用，比抛错危险得多。
    """
    body = await request.json()
    RECEIVED.append(body)
    SEEN.append(request.path)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(_sse_chunk({"role": "assistant", "content": ""}))
    await resp.write(_sse_chunk(_STREAM_TOOL_DELTAS[0]))
    await resp.write(_sse_chunk(_STREAM_TOOL_DELTAS[1]))
    # 不发 finish_reason、不发 [DONE]，直接掐断底层连接。
    transport = request.transport
    if transport is not None:
        transport.close()
    with contextlib.suppress(Exception):
        await resp.write_eof()
    return resp


async def start_mock() -> tuple[web.AppRunner, str]:
    """起 mock 端点，返回 (runner, base_url)。端口交给系统分配，避免撞占用。"""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _handler)
    # /v2 与 /v1 指向同一个 handler，只为让 Router 有两个**健康**的 deployment 可挑——
    # 并发那条探针要看它把请求分到哪几个，全压一个就说明选择逻辑被锁串行化了。
    app.router.add_post("/v2/chat/completions", _handler)
    app.router.add_post("/down/chat/completions", _handler_fail)
    app.router.add_post("/half/chat/completions", _handler_half)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}/v1"


# ── 本仓真实 payload 的形态（照 formatter.py / llm.py 逐字复刻） ──────────────

MESSAGES = [
    {
        "role": "system",
        # CacheAwareOpenAIFormatter._mark 打出来的就是这个形状：字符串 content 被改写成
        # block 列表，最后一个 block 带 cache_control。
        "content": [
            {"type": "text", "text": "你是购物 Agent。", "cache_control": {"type": "ephemeral"}}
        ],
    },
    {"role": "user", "content": "帮我找个背包"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "item_search",
            "description": "单平台商品检索。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

EXTRA_BODY = {"enable_thinking": False}


def _find_cache_control(body: dict[str, Any]) -> bool:
    """body 的 messages 里还留着 cache_control 标记吗（逐字，不接受被改形）。"""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control") == {"type": "ephemeral"}:
                    return True
    return False


# ── 判据探针 ────────────────────────────────────────────────────────────────


async def probe_litellm(base_url: str) -> dict[str, Any]:
    """P1/P2/P3：用 LiteLLM 打 mock，看它发出去的 body 与它解析回来的 tool call。"""
    import litellm

    litellm.suppress_debug_info = True
    RECEIVED.clear()

    await litellm.acompletion(
        model="openai/spike-model",
        api_key="sk-spike",
        base_url=base_url,
        messages=MESSAGES,
        extra_body=EXTRA_BODY,
    )
    sent = RECEIVED[-1]

    stream = await litellm.acompletion(
        model="openai/spike-model",
        api_key="sk-spike",
        base_url=base_url,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )
    name, args = "", ""
    async for chunk in stream:
        # include_usage 会在末尾补一片没有 choices 的 usage chunk，硬取 [0] 会 IndexError。
        if not chunk.choices:
            continue
        calls = chunk.choices[0].delta.tool_calls or []
        for call in calls:
            if call.function.name:
                name = call.function.name
            if call.function.arguments:
                args += call.function.arguments

    return {
        "p1_cache_control": _find_cache_control(sent),
        "p2_enable_thinking": sent.get("enable_thinking") is False,
        "p3_stream_tool_call": name == "item_search" and _parsed(args) == {"query": "bag"},
        "_sent_keys": sorted(sent.keys()),
        "_stream_name": name,
        "_stream_args": args,
    }


def _parsed(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def probe_openai(base_url: str) -> dict[str, Any]:
    """同样的 payload 走裸 openai SDK —— **对照组**。

    没有这一组，LiteLLM 丢了标记时分不清是它裁的还是 mock / payload 本身有问题。
    """
    from openai import AsyncOpenAI

    RECEIVED.clear()
    client = AsyncOpenAI(api_key="sk-spike", base_url=base_url)
    await client.chat.completions.create(
        model="spike-model", messages=MESSAGES, extra_body=EXTRA_BODY
    )
    sent = RECEIVED[-1]

    stream = await client.chat.completions.create(
        model="spike-model", messages=MESSAGES, tools=TOOLS, tool_choice="auto", stream=True
    )
    name, args = "", ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        for call in chunk.choices[0].delta.tool_calls or []:
            if call.function and call.function.name:
                name = call.function.name
            if call.function and call.function.arguments:
                args += call.function.arguments
    await client.close()
    return {
        "p1_cache_control": _find_cache_control(sent),
        "p2_enable_thinking": sent.get("enable_thinking") is False,
        "p3_stream_tool_call": name == "item_search" and _parsed(args) == {"query": "bag"},
    }


async def bench(base_url: str, rounds: int) -> dict[str, float]:
    """P4：两条路各打 rounds 次非流式，比 p50。本地回环，量的是包装层自身开销。"""
    import litellm
    from openai import AsyncOpenAI

    litellm.suppress_debug_info = True
    client = AsyncOpenAI(api_key="sk-spike", base_url=base_url)

    async def _one_openai() -> float:
        t0 = time.perf_counter()
        await client.chat.completions.create(model="spike-model", messages=MESSAGES)
        return (time.perf_counter() - t0) * 1000

    async def _one_litellm() -> float:
        t0 = time.perf_counter()
        await litellm.acompletion(
            model="openai/spike-model", api_key="sk-spike", base_url=base_url, messages=MESSAGES
        )
        return (time.perf_counter() - t0) * 1000

    # 各预热 5 次：首次调用要建连接池 / 加载 tokenizer，算进 p50 是噪声。
    for _ in range(5):
        await _one_openai()
        await _one_litellm()
    raw_openai = [await _one_openai() for _ in range(rounds)]
    raw_litellm = [await _one_litellm() for _ in range(rounds)]
    await client.close()

    p50_o = statistics.median(raw_openai)
    p50_l = statistics.median(raw_litellm)
    return {
        "openai_p50_ms": round(p50_o, 3),
        "litellm_p50_ms": round(p50_l, 3),
        "overhead_p50_ms": round(p50_l - p50_o, 3),
        "openai_p95_ms": round(sorted(raw_openai)[int(rounds * 0.95)], 3),
        "litellm_p95_ms": round(sorted(raw_litellm)[int(rounds * 0.95)], 3),
    }


async def probe_router(base_url: str) -> dict[str, Any]:
    """判据之外的一刀：LiteLLM **Router** 路径（阶段 2 第 2 条真正要用的那条）。

    `acompletion` 过了不等于 Router 也过——Router 多一层 deployment 选择与 fallback 重写。
    这里验三件事：①坏 deployment（恒 500）会不会按 fallbacks 切到好的；②切过去之后
    cache_control / enable_thinking 还在不在 body 里；③失败那次是不是真打到了坏端点
    （而不是 Router 自己短路）。
    """
    from litellm import Router

    down_url = base_url.replace("/v1", "/down")
    RECEIVED.clear()
    router = Router(
        model_list=[
            {"model_name": "main", "litellm_params": {
                "model": "openai/spike-model", "api_key": "sk-spike", "api_base": down_url}},
            {"model_name": "backup", "litellm_params": {
                "model": "openai/spike-model", "api_key": "sk-spike", "api_base": base_url}},
        ],
        fallbacks=[{"main": ["backup"]}],
        num_retries=0,
    )
    resp = await router.acompletion(
        model="main", messages=MESSAGES, extra_body=EXTRA_BODY, mock_testing_fallbacks=False
    )
    sent = RECEIVED[-1] if RECEIVED else {}
    return {
        "fallback_worked": bool(resp) and resp.choices[0].message.content == "ok",
        "hit_down_first": len(RECEIVED) >= 2,
        "cache_control_after_fallback": _find_cache_control(sent),
        "enable_thinking_after_fallback": sent.get("enable_thinking") is False,
        "_requests_seen": len(RECEIVED),
    }


# ── 补测：spike 记录里点名「没测」的三条（阶段 2 开工前补） ──────────────────
#
# 判据（照旧先写后跑）：
#   R1 Router + stream=True + tools，坏 deployment 在**首 token 之前** 500 → 必须切到备用，
#      且 tool call 逐片拼出 {"query": "bag"}。不过 → 流式路径不能靠 Router 做 fallback。
#   R2 流**中途**断 → 必须抛异常，不得静默返回半截 arguments。半截被当成功用比抛错危险。
#   R3 真 CacheAwareOpenAIFormatter 的输出喂 Router → cache_control 逐字仍在 body 里。
#      （原 spike 的 MESSAGES 是手写复刻的，复刻对了不等于真 formatter 就是这个形状。）
#   R4 并发 50 下 Router 相对裸 acompletion 的 p50 额外开销 < 20ms（与 P4 同门槛），
#      且两个健康 deployment 都拿到 >10% 的请求（全压一个 = 选择逻辑被串行化了）。


def _router(entries: list[tuple[str, str]], fallbacks: list[dict[str, list[str]]] | None = None):
    """按 (model_name, api_base) 列表建一个 Router。``num_retries=0`` 是定死的口径：

    重试只由 ``LLM_MAX_RETRIES`` 一处决定，让 Router 再叠一层就是「一个 429 被试 9 次」。
    """
    from litellm import Router

    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/spike-model",
                    "api_key": "sk-spike",
                    "api_base": url,
                },
            }
            for name, url in entries
        ],
        fallbacks=fallbacks or [],
        num_retries=0,
    )


async def _drain_tool_call(stream: Any) -> tuple[str, str]:
    """把流式 tool call 逐片拼起来，返回 (name, arguments)。

    **逐片不 strip**：siliconflow 的首片带前导空格，strip 掉就把 JSON 拼坏（见能力矩阵那节）。
    """
    name, args = "", ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        for call in chunk.choices[0].delta.tool_calls or []:
            if call.function and call.function.name:
                name = call.function.name
            if call.function and call.function.arguments:
                args += call.function.arguments
    return name, args


async def probe_router_stream(base_url: str) -> dict[str, Any]:
    """R1 + R2：Router × 流式 × tool call，分「首 token 前失败」与「流中途断」两种。"""
    down_url = base_url.replace("/v1", "/down")
    half_url = base_url.replace("/v1", "/half")

    # R1：坏的先打，200 都没回来 → fallback 的窗口还开着。
    RECEIVED.clear()
    SEEN.clear()
    r1 = _router([("main", down_url), ("backup", base_url)], [{"main": ["backup"]}])
    stream = await r1.acompletion(
        model="main", messages=MESSAGES, tools=TOOLS, tool_choice="auto", stream=True
    )
    name, args = await _drain_tool_call(stream)
    r1_hit_down = any(p.startswith("/down") for p in SEEN)

    # R2：200 已回、两片已发，然后连接断。
    RECEIVED.clear()
    SEEN.clear()
    r2 = _router([("main", half_url), ("backup", base_url)], [{"main": ["backup"]}])
    raised, partial, err = "", "", ""
    try:
        stream = await r2.acompletion(
            model="main", messages=MESSAGES, tools=TOOLS, tool_choice="auto", stream=True
        )
        _, partial = await _drain_tool_call(stream)
    except BaseException as exc:  # noqa: BLE001 - 要的就是「它到底抛什么」
        raised, err = type(exc).__name__, str(exc)[:200]

    return {
        "r1_fallback_stream_ok": name == "item_search" and _parsed(args) == {"query": "bag"},
        "r1_hit_down_first": r1_hit_down,
        "r2_raised": raised or None,
        "r2_error": err or None,
        "r2_silent_partial": (not raised) and partial not in ("", '{"query": "bag"}'),
        "_r2_partial_args": partial,
        "_r2_requests_seen": list(SEEN),
    }


async def probe_real_formatter(base_url: str) -> dict[str, Any]:
    """R3：**真** ``CacheAwareOpenAIFormatter`` 的输出喂 Router，标记还在不在。

    原 spike 的 ``MESSAGES`` 是照 formatter 手写复刻的——复刻对了只证明「这个形状能透传」，
    不证明 formatter 真就产这个形状。这里把两者接上：真 formatter 产 → Router 发 → 看 body。
    """
    from agentscope.message import Msg, TextBlock

    from app.harness.formatter import MIN_CACHE_PREFIX_TOKENS, CacheAwareOpenAIFormatter

    # 标记只在 system 段超过最小写入阈值时才打，system 要够长才测得到东西。
    system_text = "你是全球电商购物 Agent。" * (MIN_CACHE_PREFIX_TOKENS // 2)
    msgs = [
        Msg(name="system", role="system", content=[TextBlock(type="text", text=system_text)]),
        Msg(name="user", role="user", content=[TextBlock(type="text", text="帮我找个背包")]),
    ]
    formatted = await CacheAwareOpenAIFormatter().format(msgs)
    marked_before = _find_cache_control({"messages": formatted})

    RECEIVED.clear()
    SEEN.clear()
    router = _router([("main", base_url)])
    await router.acompletion(model="main", messages=formatted, extra_body=EXTRA_BODY)
    sent = RECEIVED[-1] if RECEIVED else {}
    return {
        "r3_formatter_marked": marked_before,
        "r3_cache_control_through_router": _find_cache_control(sent),
        "r3_enable_thinking_through_router": sent.get("enable_thinking") is False,
        "_system_role": (formatted[0].get("role") if formatted else None),
    }


async def probe_router_concurrency(base_url: str, n: int) -> dict[str, Any]:
    """R4：并发 n 条同时打，Router 的 deployment 选择有没有把并发串行化。

    两个健康 deployment（/v1 与 /v2）。看两件事：额外延迟（对照裸 ``acompletion`` 同并发）、
    以及请求是不是真分到了两边。
    """
    import litellm

    litellm.suppress_debug_info = True
    router = _router([("main", base_url), ("main", base_url.replace("/v1", "/v2"))])

    async def _one_router() -> float:
        t0 = time.perf_counter()
        await router.acompletion(model="main", messages=MESSAGES)
        return (time.perf_counter() - t0) * 1000

    async def _one_plain() -> float:
        t0 = time.perf_counter()
        await litellm.acompletion(
            model="openai/spike-model", api_key="sk-spike", base_url=base_url, messages=MESSAGES
        )
        return (time.perf_counter() - t0) * 1000

    await asyncio.gather(*(_one_router() for _ in range(5)))  # 预热：建连接池
    RECEIVED.clear()
    SEEN.clear()
    r_lat = await asyncio.gather(*(_one_router() for _ in range(n)))
    dist = {
        "/v1": sum(p.startswith("/v1") for p in SEEN),
        "/v2": sum(p.startswith("/v2") for p in SEEN),
    }
    p_lat = await asyncio.gather(*(_one_plain() for _ in range(n)))

    p50_r, p50_p = statistics.median(r_lat), statistics.median(p_lat)
    spread = min(dist.values()) / max(1, sum(dist.values()))
    return {
        "r4_overhead_p50_lt_20ms": (p50_r - p50_p) < 20,
        "r4_both_deployments_used": spread > 0.10,
        "concurrency": n,
        "router_p50_ms": round(p50_r, 3),
        "plain_p50_ms": round(p50_p, 3),
        "overhead_p50_ms": round(p50_r - p50_p, 3),
        "router_p95_ms": round(sorted(r_lat)[int(n * 0.95)], 3),
        "plain_p95_ms": round(sorted(p_lat)[int(n * 0.95)], 3),
        "deployment_dist": dist,
    }


# ── live：provider 能力矩阵（阶段 2 第 3 条） ────────────────────────────────

LIVE_MESSAGES = [
    {"role": "system", "content": [
        {"type": "text", "text": "你是购物助手，需要搜商品时调用工具。",
         "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": "搜一个双肩包"},
]


async def live_provider(name: str, base_url: str, api_key: str, model: str) -> dict[str, Any]:
    """对一个真 provider 打 3 次小请求，产矩阵一行。

    **诚实口径**：`cache_control_accepted` / `enable_thinking_accepted` 只能证明「网关收下了、
    没报 400」，证明不了「它真的按语义生效」——命中率要看 usage 的 cached_tokens，思考开关要看
    延迟与 reasoning_content。两者都在返回里顺手记下来，判绿只认「不报错」。
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    row: dict[str, Any] = {"provider": name, "model": model, "calls": 0}

    async def _guard(key: str, coro: Any) -> Any:
        row["calls"] += 1
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 —— 矩阵要的就是「哪条炸了、怎么炸的」
            row[key] = False
            row[f"{key}_error"] = f"{type(exc).__name__}: {exc}"[:200]
            return None

    resp = await _guard("cache_control_accepted", client.chat.completions.create(
        model=model, messages=LIVE_MESSAGES, max_tokens=16,
        extra_body={"enable_thinking": False}))
    if resp is not None:
        row["cache_control_accepted"] = True
        row["enable_thinking_accepted"] = True
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        row["_cached_tokens"] = getattr(details, "cached_tokens", None)
        row["_reasoning"] = bool(getattr(resp.choices[0].message, "reasoning_content", None))
    else:
        row.setdefault("enable_thinking_accepted", False)

    stream = await _guard("tool_choice_auto_stream", client.chat.completions.create(
        model=model, messages=LIVE_MESSAGES, tools=TOOLS, tool_choice="auto",
        stream=True, max_tokens=64, extra_body={"enable_thinking": False}))
    if stream is not None:
        name_, args = "", ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            for call in chunk.choices[0].delta.tool_calls or []:
                if call.function and call.function.name:
                    name_ = call.function.name
                if call.function and call.function.arguments:
                    args += call.function.arguments
        row["tool_choice_auto_stream"] = name_ == "item_search" and isinstance(_parsed(args), dict)
        row["_auto_args"] = args[:120]

    forced = await _guard("_forced_ok", client.chat.completions.create(
        model=model, messages=LIVE_MESSAGES, tools=TOOLS, max_tokens=64,
        tool_choice={"type": "function", "function": {"name": "item_search"}},
        extra_body={"enable_thinking": False}))
    if forced is not None:
        calls = forced.choices[0].message.tool_calls or []
        args = calls[0].function.arguments if calls else ""
        parsed = _parsed(args)
        # 存根 = 调用回来了，但参数是空壳（见 structured-output-forced-toolchoice-stub）。
        row["stub_on_forced_tool_choice"] = not (
            isinstance(parsed, dict) and str(parsed.get("query", "")).strip()
        )
        row["_forced_args"] = args[:120]
    await client.close()
    return row


def live_targets() -> list[tuple[str, str, str, str]]:
    """(名字, base_url, api_key, model)；缺配置的 provider 直接跳过，不假装跑过。"""
    out = []
    if os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY"):
        out.append(("dashscope", os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"],
                    os.environ.get("SPIKE_DASHSCOPE_MODEL") or os.environ.get("LLM_PLANNER")
                    or os.environ["LLM_MAIN"]))
    if os.environ.get("EMBED_BASE_URL") and os.environ.get("EMBED_API_KEY"):
        out.append(("siliconflow", os.environ["EMBED_BASE_URL"], os.environ["EMBED_API_KEY"],
                    os.environ.get("SPIKE_SF_MODEL", "Qwen/Qwen3-8B")))
    return out


# ── 入口 ────────────────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> None:
    runner, base_url = await start_mock()
    try:
        lite = await probe_litellm(base_url)
        base = await probe_openai(base_url)
        router = await probe_router(base_url)
        rstream = await probe_router_stream(base_url)
        rfmt = await probe_real_formatter(base_url)
        rconc = await probe_router_concurrency(base_url, args.concurrency)
        timing = await bench(base_url, args.rounds)
    finally:
        await runner.cleanup()

    verdict = {
        "P1_cache_control": lite["p1_cache_control"],
        "P2_enable_thinking": lite["p2_enable_thinking"],
        "P3_stream_tool_call": lite["p3_stream_tool_call"],
        "P4_overhead_lt_20ms": timing["overhead_p50_ms"] < 20,
    }
    # 补测三条单独记账：它们不改「切不切 LiteLLM」的结论（那是 P1~P4 的事），
    # 改的是**阶段 2 那层包装怎么写**，所以不混进 all_pass。
    followup = {
        "R1_router_stream_fallback": rstream["r1_fallback_stream_ok"],
        "R2_mid_stream_not_silent": not rstream["r2_silent_partial"],
        "R3_real_formatter_through_router": rfmt["r3_cache_control_through_router"],
        "R4_no_lock_contention": rconc["r4_overhead_p50_lt_20ms"],
        "R4_both_deployments_used": rconc["r4_both_deployments_used"],
    }
    report: dict[str, Any] = {
        "verdict": verdict,
        "all_pass": all(verdict.values()),
        "followup_verdict": followup,
        "followup_all_pass": all(followup.values()),
        "litellm_probe": lite,
        "openai_control_group": base,
        "router_probe": router,
        "router_stream_probe": rstream,
        "real_formatter_probe": rfmt,
        "router_concurrency_probe": rconc,
        "timing": timing,
    }

    if args.live:
        from dotenv import load_dotenv

        load_dotenv()
        rows = []
        for name, url, key, model in live_targets():
            print(f"[live] {name} / {model} …", flush=True)
            rows.append(await live_provider(name, url, key, model))
        report["capability_matrix"] = rows

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 5 LiteLLM spike")
    parser.add_argument("--live", action="store_true", help="加跑真 provider 能力矩阵（花钱）")
    parser.add_argument("--rounds", type=int, default=200, help="P4 每条路的采样次数")
    parser.add_argument("--concurrency", type=int, default=50, help="R4 并发探针的并发数")
    parser.add_argument("--out", default="", help="把报告 JSON 另存到这个路径")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
