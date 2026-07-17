"""M6 示例：Cache Breakpoint 上下文压缩。

上半场（确定性，不调 LLM）：造一段越来越长的对话（每轮一次返回大结果的 item_search），
逐轮跑 post_step_compress，打印：
  - 原始 vs 压缩后的 token 估算（看「不随轮数线性爆炸」）；
  - 缓存前缀是否逐轮稳定（看 Cache Breakpoint 的根本不变式：前缀只增不变）。

下半场（可选真实 LLM）：把 Harness 控制面（压缩是其中的 pre_think Hook）挂到 create_agent 跑一个真实
查询，演示中间件对主 loop 透明——它只在每次请求模型前给「模型看到的历史」瘦身，
checkpoint 原文不动。无 LLM 环境会自动跳过。

运行：uv run python examples/06_compress.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from app.compress.breakpoint import compute_breakpoint  # noqa: E402
from app.compress.pipeline import post_step_compress  # noqa: E402
from app.harness.agent_middleware import build_agent_middleware  # noqa: E402

_CHARS_PER_TOKEN = 4


def _est_tokens(messages: list) -> int:
    return sum(len(str(m.content)) for m in messages) // _CHARS_PER_TOKEN


def _round(idx: int, payload_chars: int) -> list:
    return [
        HumanMessage(content=f"第{idx}轮：再换个平台搜搜"),
        AIMessage(content="", tool_calls=[{"name": "item_search", "args": {}, "id": f"c{idx}"}]),
        ToolMessage(content="商品详情" * payload_chars, tool_call_id=f"c{idx}", name="item_search"),
    ]


def demo_compression() -> None:
    print("=== 上半场：逐轮压缩 + 前缀稳定（确定性）===")
    print(f"{'轮次':>4} | {'原始tok':>8} | {'压缩tok':>8} | {'省':>5} | 前缀稳定")
    print("-" * 52)

    msgs: list = []
    prev_view: list | None = None
    prev_prefix_len = 0
    for n in range(1, 13):
        msgs.extend(_round(n, payload_chars=800))  # 每条工具结果 ~3200 字符 ≈ 800 token
        view = post_step_compress(msgs, keep_recent=3, max_tool_tokens=300)

        raw_tok = _est_tokens(msgs)
        view_tok = _est_tokens(view)
        saved = f"{(1 - view_tok / raw_tok) * 100:.0f}%" if raw_tok else "-"

        stable = "—"
        if prev_view is not None:
            stable = (
                "✓"
                if all(view[i].content == prev_view[i].content for i in range(prev_prefix_len))
                else "✗ 破!"
            )

        print(f"{n:>4} | {raw_tok:>8} | {view_tok:>8} | {saved:>5} | {stable}")
        prev_view = view
        prev_prefix_len = compute_breakpoint(msgs, keep_recent=3)

    print(
        "\n看点：原始 token 随轮数线性涨，压缩 token 被「较旧区逐条截断」压住；"
        "缓存前缀逐轮恒为上一轮前缀的逐字延伸（全 ✓）→ Prompt Cache 可持续命中。"
    )


async def demo_middleware_mount() -> None:
    print("\n=== 下半场：中间件挂到 create_agent（需真实 LLM，缺失则跳过）===")
    try:
        from langchain.agents import create_agent

        from app.agent.llm import get_llm

        llm = get_llm()
    except Exception as e:  # noqa: BLE001
        print(f"  跳过（未配置 LLM 或导入失败）：{type(e).__name__}: {e}")
        return

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="你是购物助手，直接简短回答即可。",
        # 压缩不再是独立中间件：它是 Harness 的 pre_think Hook，随整个控制面一起挂上。
        middleware=build_agent_middleware(),
    )
    try:
        result = await agent.ainvoke({"messages": [("user", "用一句话介绍你能帮我做什么")]})
        print(f"  Agent（带压缩中间件）应答：{str(result['messages'][-1].content)[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"  跳过实际调用（LLM 不可达）：{type(e).__name__}: {e}")


async def main() -> None:
    demo_compression()
    await demo_middleware_mount()


if __name__ == "__main__":
    asyncio.run(main())
