"""阶段 5 spike：量 LiteLLM 当 provider 寻址层的四条判据 + 产 provider 能力矩阵。

判据与背景见 `docs/plans/spike-litellm-2026-09-20.md`（判据写在那份文档开头，先于本脚本落纸）。

两段：
- 默认（不花钱）：起一个本地 OpenAI 兼容 mock 端点，**把收到的 body 原样留下**，据此判
  P1 cache_control 透传 / P2 extra_body enable_thinking / P3 流式 tool call / P4 额外延迟。
  只有自己当服务端才看得见「发出去的 body 被裁掉了什么」——打真 provider 只能看到它的回复。
- `--live`：对真 provider 打少量小请求，产出阶段 2 第 3 条的能力矩阵。这段判的是 provider，
  不是 LiteLLM。

跑法::

    uv run --group spike python scripts/spike_litellm.py          # mock 段
    uv run --group spike python scripts/spike_litellm.py --live   # 加跑能力矩阵
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any

from aiohttp import web

# ── mock OpenAI 兼容端点 ─────────────────────────────────────────────────────

RECEIVED: list[dict[str, Any]] = []

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
    if not body.get("stream"):
        return web.json_response(_NON_STREAM)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(_sse_chunk({"role": "assistant", "content": ""}))
    for delta in _STREAM_TOOL_DELTAS:
        await resp.write(_sse_chunk(delta))
    await resp.write(_sse_chunk({}, finish="tool_calls"))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _handler_fail(request: web.Request) -> web.Response:
    """恒 500 的端点：给 Router fallback 用（5xx 才是断路器与 fallback 该接的那类错）。"""
    RECEIVED.append(await request.json())
    return web.json_response({"error": {"message": "spike: upstream down"}}, status=500)


async def start_mock() -> tuple[web.AppRunner, str]:
    """起 mock 端点，返回 (runner, base_url)。端口交给系统分配，避免撞占用。"""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _handler)
    app.router.add_post("/down/chat/completions", _handler_fail)
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
        timing = await bench(base_url, args.rounds)
    finally:
        await runner.cleanup()

    verdict = {
        "P1_cache_control": lite["p1_cache_control"],
        "P2_enable_thinking": lite["p2_enable_thinking"],
        "P3_stream_tool_call": lite["p3_stream_tool_call"],
        "P4_overhead_lt_20ms": timing["overhead_p50_ms"] < 20,
    }
    report: dict[str, Any] = {
        "verdict": verdict,
        "all_pass": all(verdict.values()),
        "litellm_probe": lite,
        "openai_control_group": base,
        "router_probe": router,
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
    parser.add_argument("--out", default="", help="把报告 JSON 另存到这个路径")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
