"""M6 示例：Cache Breakpoint 上下文压缩。

上半场（确定性，不调 LLM）：造一段越来越长的对话（每轮一次返回大结果的 item_search），
逐轮跑 post_step_compress，打印：
  - 原始 vs 压缩后的 token 估算（看「不随轮数线性爆炸」）；
  - 缓存前缀是否逐轮稳定（看 Cache Breakpoint 的根本不变式：前缀只增不变）。

下半场（可选真实 LLM）：把 Harness 控制面（压缩是其中的 pre_think Hook）挂到一个真 Agent 上跑
一个查询，演示控制面对主 loop 透明——它只在每次请求模型前给「模型看到的历史」瘦身，
``state.context`` 里的原文不动。无 LLM 环境会自动跳过。

**断点是 block 粒度，不是消息下标**：AgentScope 里一轮 = **一条** assistant ``Msg``，该轮所有
tool_call / tool_result 都装在它的 content blocks 里。所以断点坐标是 ``(消息下标, block 下标)``
二元组，压缩也下沉到 block（见 app/compress/blocks.py 顶部）。

运行：uv run python examples/06_compress.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentscope.message import (  # noqa: E402
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)

from app.compress.blocks import compute_block_breakpoint, post_step_compress  # noqa: E402

_CHARS_PER_TOKEN = 4


def _est_tokens(messages: list[Msg]) -> int:
    """粗估 token：把每条消息的 content 拍平成字符串按字符数折算，够看趋势就行。"""
    return sum(len(str(m.content)) for m in messages) // _CHARS_PER_TOKEN


def _round(idx: int, payload_chars: int) -> list[Msg]:
    """造一轮对话：一条用户消息 + 一条含 tool_call/tool_result 的 assistant 消息。"""
    call_id = f"c{idx}"
    return [
        Msg(
            name="user",
            role="user",
            content=[TextBlock(type="text", text=f"第{idx}轮：再换个平台搜搜")],
        ),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ToolCallBlock(type="tool_call", id=call_id, name="item_search", input="{}"),
                ToolResultBlock(
                    type="tool_result",
                    id=call_id,
                    name="item_search",
                    output=[TextBlock(type="text", text="商品详情" * payload_chars)],
                ),
            ],
        ),
    ]


def _block_payload(block: object) -> str:
    """取一个 block 的**文本载荷**——只要真会发给模型的那部分。

    **别拿 ``str(block)`` 去比**：压缩会重建 block 对象，新对象带全新的 ``id`` 与
    ``created_at``，repr 每轮都不同，比出来必然「破」——而那是内存对象的元数据，
    ``CacheAwareOpenAIFormatter`` 序列化时根本不发（实测 payload 里没有 created_at）。
    判缓存前缀稳不稳，口径只能是**发出去的 payload**，不是内存对象长什么样。
    """
    if isinstance(block, dict):
        btype = block.get("type")
        if btype == "text":
            return f"text:{block.get('text', '')}"
        if btype == "tool_call":
            return f"call:{block.get('name')}:{block.get('input')}"
        if btype == "tool_result":
            output = block.get("output")
            if isinstance(output, list):
                inner = "".join(str(b.get("text", "")) for b in output if isinstance(b, dict))
            else:
                inner = str(output)
            return f"result:{block.get('name')}:{inner}"
        return str(block)
    btype = getattr(block, "type", None)
    if btype == "text":
        return f"text:{getattr(block, 'text', '')}"
    if btype == "tool_call":
        return f"call:{getattr(block, 'name', '')}:{getattr(block, 'input', '')}"
    if btype == "tool_result":
        output = getattr(block, "output", None)
        if isinstance(output, list):
            inner = "".join(str(getattr(b, "text", "") or "") for b in output)
        else:
            inner = str(output)
        return f"result:{getattr(block, 'name', '')}:{inner}"
    return str(block)


def _prefix_blocks(messages: list[Msg], bp: tuple[int, int]) -> list[str]:
    """把断点之前所有 block 的文本载荷拍成序列，用于逐字比对前缀是否稳定。"""
    out: list[str] = []
    for i, msg in enumerate(messages):
        content = msg.content if isinstance(msg.content, list) else []
        for j, block in enumerate(content):
            if (i, j) >= bp:
                return out
            out.append(_block_payload(block))
    return out


# 比对口径要用**上一轮的断点**，不能用本轮的：压缩改写的是断点**之前**的较旧区，所以断点每
# 往后挪一次，就有新的一批 block 从「保留区」进入「压缩区」并被截断——拿本轮断点去比，那批
# 刚被压的 block 必然与上一轮不同，会误报「破」。真正的不变式是：**上一轮已经压过的那段，
# 这一轮再压结果逐字相同**（post_step_compress 幂等），也就是缓存前缀只增不改。


def demo_compression() -> None:
    print("=== 上半场：逐轮压缩 + 前缀稳定（确定性）===")
    print(f"{'轮次':>4} | {'原始tok':>8} | {'压缩tok':>8} | {'省':>5} | 前缀稳定")
    print("-" * 52)

    msgs: list[Msg] = []
    prev_view: list[Msg] | None = None
    prev_bp: tuple[int, int] = (0, 0)
    for n in range(1, 13):
        msgs.extend(_round(n, payload_chars=800))  # 每条工具结果 ~3200 字符 ≈ 800 token
        view = post_step_compress(msgs, keep_recent=3, max_tool_tokens=300)

        raw_tok = _est_tokens(msgs)
        view_tok = _est_tokens(view)
        saved = f"{(1 - view_tok / raw_tok) * 100:.0f}%" if raw_tok else "-"

        stable = "—"
        if prev_view is not None:
            # 上一轮断点之前那段，本轮必须逐字不变（改一个字，缓存就从那里断掉）。
            stable = (
                "✓"
                if _prefix_blocks(view, prev_bp) == _prefix_blocks(prev_view, prev_bp)
                else "✗ 破!"
            )

        print(f"{n:>4} | {raw_tok:>8} | {view_tok:>8} | {saved:>5} | {stable}")
        prev_view = view
        prev_bp = compute_block_breakpoint(msgs, keep_recent=3)

    print(
        "\n看点：原始 token 随轮数线性涨，压缩 token 被「较旧区逐条截断」压住；"
        "缓存前缀逐轮恒为上一轮前缀的逐字延伸（全 ✓）→ Prompt Cache 可持续命中。"
    )
    print(
        "注：cache_control 标记不在这层打（content block 是强类型的），落在 harness/formatter.py。"
    )


async def demo_middleware_mount() -> None:
    print("\n=== 下半场：控制面挂到真 Agent（需真实 LLM，缺失则跳过）===")
    try:
        from agentscope.agent import Agent, ReActConfig
        from agentscope.tool import Toolkit

        from app.agent.llm import get_llm
        from app.harness.adapter import HarnessAgentAdapter, HarnessSession

        llm = get_llm()
    except Exception as e:  # noqa: BLE001
        print(f"  跳过（未配置 LLM 或导入失败）：{type(e).__name__}: {e}")
        return

    # 压缩不是独立中间件：它是 Harness 的 pre_think Hook，随整个控制面适配器一起挂上。
    session = HarnessSession(original_query="用一句话介绍你能帮我做什么")
    agent = Agent(
        name="compress_demo",
        system_prompt="你是购物助手，直接简短回答即可。",
        model=llm,
        toolkit=Toolkit(),
        middlewares=[HarnessAgentAdapter(session)],
        react_config=ReActConfig(max_iters=2),
    )
    try:
        query = session.original_query
        reply = await agent.reply(
            Msg(name="user", role="user", content=[TextBlock(type="text", text=query)])
        )
        print(f"  Agent（带控制面）应答：{str(reply.get_text_content() or '')[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"  跳过实际调用（LLM 不可达）：{type(e).__name__}: {e}")


async def main() -> None:
    demo_compression()
    await demo_middleware_mount()


if __name__ == "__main__":
    asyncio.run(main())
