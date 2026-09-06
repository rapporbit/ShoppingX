"""批 1 验收②：同轮多派 ``task_dispatch`` 到底是不是**真并行**跑的。

删掉 ``parallel_dispatch_tool`` 之后，并发不再由我们自己排——改成给工具标
``is_concurrency_safe=True``，指望框架看到同一轮里的多个 tool_call 就并行执行。这是个**关于
框架行为的假设**，而假设塌了不会报错：所有子任务照样跑完、结果照样对，只是从 2 秒变成 6 秒，
在真实链路的噪声里根本看不出来。所以要有一条确定性的证据。

做法是把变量掐到只剩「并发与否」：假模型（不打网络）在一轮里发 N 个 task_dispatch，假 worker
只 sleep 固定时长。并行则 wall ≈ 单个 worker 的时长，串行则 ≈ N 倍。同时记录每个 worker 的
起止时间戳算**重叠系数**——wall 时间会被装配开销带偏，重叠区间不会。

跑法（不烧 token、不打网络，秒级返回）：
    uv run python scripts/verify_parallel.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentscope.message import Msg, TextBlock, ToolCallBlock  # noqa: E402

WORKER_SLEEP_SEC = 2.0
N_TASKS = 3


class _FakeWorker:
    """假 worker：只睡 ``WORKER_SLEEP_SEC``，并把自己的起止时间戳记进 ``marks``。"""

    def __init__(self, marks: list[tuple[float, float]]) -> None:
        self._marks = marks

    async def reply(self, _msg: Any) -> Msg:
        t0 = time.perf_counter()
        await asyncio.sleep(WORKER_SLEEP_SEC)
        self._marks.append((t0, time.perf_counter()))
        return Msg(name="w", role="assistant", content=[TextBlock(type="text", text="找到 3 件")])


def _fake_model(rounds: list[list[dict[str, Any]]]) -> Any:
    """假模型：第 i 次被调用就吐 ``rounds[i]`` 描述的那批 tool_call，用完则收尾。

    ``rounds`` 的每一项是本轮要发的 demands 列表——一项里放 N 条 = 同轮多派（期望并行），
    放 1 条而给 N 项 = 一轮一条（串行基准）。
    """
    from agentscope.credential import OpenAICredential
    from agentscope.model import ChatResponse, OpenAIChatModel

    model = OpenAIChatModel(
        credential=OpenAICredential(api_key="sk-test", base_url="http://localhost:1/v1"),
        model="test-model",
        stream=False,
        max_retries=0,
    )
    seq = iter(rounds)

    async def _call(*_a: object, **_kw: object) -> ChatResponse:
        batch = next(seq, None)
        if batch is None:
            return ChatResponse(content=[TextBlock(type="text", text="都查完了")], is_last=True)
        # 注意 ``ToolCallBlock.input`` 是**原始 JSON 字符串**（流式累积用的），不是 dict——
        # 传 dict 不会当场报错，会在框架解析入参时才炸出一句不着边际的错。
        blocks = [
            ToolCallBlock(
                type="tool_call",
                id=f"call-{i}-{time.time_ns()}",
                name="task_dispatch",
                input=json.dumps(call, ensure_ascii=False),
            )
            for i, call in enumerate(batch)
        ]
        return ChatResponse(content=blocks, is_last=True)

    model._call_api = _call  # type: ignore[method-assign]
    return model


def _demand(platform: str) -> dict[str, Any]:
    return {
        "demands": f"在 {platform} 上找一个帆布旅行包，预算 300 美元，不要塑料件",
        "subagent_type": "search",
    }


async def _run(rounds: list[list[dict[str, Any]]]) -> tuple[float, list[tuple[float, float]]]:
    """按 ``rounds`` 驱动一次主 loop，返回 (wall 秒, 每个 worker 的起止时间戳)。"""
    from app.agent import agents as ag
    from app.agent import dispatch_tool as dt
    from app.agent.platform_scope import platform_scope
    from app.utils.thread_ctx import thread_scope

    marks: list[tuple[float, float]] = []
    model = _fake_model(rounds)
    ag.get_llm = lambda: model  # type: ignore[assignment]
    ag.get_fast_llm = lambda: model  # type: ignore[assignment]

    async def _build_worker(_kind: str = "search") -> Any:
        return _FakeWorker(marks)

    ag.build_worker_agent = _build_worker  # type: ignore[assignment]
    dt.build_preference_block = _no_prefs  # type: ignore[assignment]

    thread_id = f"verify-parallel-{time.time_ns()}"
    session_dir = Path("output") / thread_id
    session_dir.mkdir(parents=True, exist_ok=True)  # 不建则 P_t / 候选池落盘一路报错刷屏
    with thread_scope(thread_id, session_dir), platform_scope(PLATFORMS):
        agent, _ = await ag.build_main_agent(original_query="跨平台找旅行包")
        msg = Msg(name="user", role="user", content=[TextBlock(type="text", text="跨平台找旅行包")])
        t0 = time.perf_counter()
        await agent.reply(msg)
        wall = time.perf_counter() - t0
    return wall, marks


async def _no_prefs(*_a: Any, **_kw: Any) -> str:
    return ""


PLATFORMS = ["amazon", "ebay", "walmart"]


def _overlap_ratio(marks: list[tuple[float, float]]) -> float:
    """重叠系数 = (Σ各自时长) / (并集时长)。完全并行 ≈ N，完全串行 ≈ 1。

    比 wall 时间稳：wall 会被装配开销、事件上报带偏，区间并集不会。
    """
    if not marks:
        return 0.0
    total = sum(e - s for s, e in marks)
    span = max(e for _, e in marks) - min(s for s, _ in marks)
    return total / span if span > 0 else float(len(marks))


def _span(marks: list[tuple[float, float]]) -> float:
    """子任务段的墙钟：第一个 worker 起到最后一个 worker 止。"""
    return max(e for _, e in marks) - min(s for s, _ in marks) if marks else 0.0


async def main() -> int:
    par_calls = [[_demand(p) for p in PLATFORMS]]  # 一轮发 3 条 = 同轮多派
    ser_calls = [[_demand(p)] for p in PLATFORMS]  # 3 轮各发 1 条 = 串行基准

    await _run([[_demand("amazon")]])  # 热身：第一次跑要付 import / 索引加载的冷启动
    par_wall, par_marks = await _run(par_calls)
    ser_wall, ser_marks = await _run(ser_calls)

    par_span, ser_span = _span(par_marks), _span(ser_marks)
    ratio = par_span / ser_span if ser_span else 0.0
    print(f"\n{'':2}同轮多派 {len(par_marks)} 条：子任务段 {par_span:5.2f}s"
          f"（整轮 wall {par_wall:5.2f}s），重叠系数 {_overlap_ratio(par_marks):.2f}")
    print(f"{'':2}一轮一条 {len(ser_marks)} 条：子任务段 {ser_span:5.2f}s"
          f"（整轮 wall {ser_wall:5.2f}s），重叠系数 {_overlap_ratio(ser_marks):.2f}")
    print(f"{'':2}并行 / 串行（子任务段）= {ratio:.1%}，验收线 < 60%")

    # **判据刻意不用整轮 wall**：它把装配、事件上报、模型往返这些与并发无关的固定开销一起算进
    # 去，冷启动那次还要多背几秒——同一份并行行为，wall 比值能在 45% 与 67% 之间晃，判据就成了
    # 掷骰子。子任务段与重叠系数只反映「这几个 worker 有没有同时在跑」，那才是这里要证的事。
    ok = ratio < 0.6 and _overlap_ratio(par_marks) > N_TASKS * 0.8
    print(f"\n{'':2}{'✅ 通过' if ok else '❌ 未通过'}：is_concurrency_safe 的并发假设"
          f"{'成立' if ok else '不成立——同轮多个 tool_call 并没有被并行执行'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
