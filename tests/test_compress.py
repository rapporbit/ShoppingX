"""Cache Breakpoint：block 级断点 + 较旧区压缩 + formatter 的缓存标记。

四个容易失效的点：

- 一整轮 = 一条 assistant ``Msg``，断点必须落进消息内部（消息级断点在这里非「全压」即「全不压」）。
- 压缩不得原地改 ``state.context`` 里的对象（改了就把持久化历史也截了）。
- ``tool_result`` 的 ``output`` 有 str 与 ``[TextBlock]`` 两种形态，压缩后形态不能变。
- cache_control 标记落在 formatter 的 dict 序列上，且 ``role="tool"`` 的条目不许被改成 block 列表
  （OpenAI 兼容端点要求它的 content 是字符串）。
"""

import json

import pytest
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock

from app.compress.blocks import (
    MIN_CACHE_PREFIX_TOKENS,
    _smart_compress_json,
    compute_block_breakpoint,
    post_step_compress,
)
from app.harness.adapter import HarnessAgentAdapter, HarnessSession
from app.harness.formatter import CacheAwareOpenAIFormatter
from app.utils.tokens import count_tokens

_UNIT = "想买便宜又抗造的旅行三件套，预算300，不要塑料的，喜欢小众品牌。"


def _payload(min_tokens: int) -> str:
    s = _UNIT
    while count_tokens(s) < min_tokens:
        s += _UNIT
    return s


def _tool_blocks(idx: int, payload: str, *, raw_str: bool = False) -> list:
    """一次 Act+Observe 的两个 block（tool_call 与它的 tool_result）。"""
    output = payload if raw_str else [TextBlock(type="text", text=payload)]
    return [
        ToolCallBlock(type="tool_call", id=f"c{idx}", name="item_search", input="{}"),
        ToolResultBlock(type="tool_result", id=f"c{idx}", name="item_search", output=output),
    ]


def _turn(n_tools: int, payload: str, *, raw_str: bool = False) -> list[Msg]:
    """造一轮完整交互：一条 user Msg + 一条把 n_tools 次工具调用全塞进去的 assistant Msg。"""
    content: list = []
    for i in range(n_tools):
        content.extend(_tool_blocks(i, payload, raw_str=raw_str))
    return [
        Msg(name="user", role="user", content=[TextBlock(type="text", text="需求")]),
        Msg(name="shoppingx", role="assistant", content=content),
    ]


def _fake_agent(*, context_size: int):
    """够 on_compress_context 用的最小 agent 替身（它只读 state / model / context_config）。"""
    from agentscope.state import AgentState

    class _Model:
        pass

    class _Cfg:
        trigger_ratio = 0.9

    class _Agent:
        def __init__(self) -> None:
            self.state = AgentState()
            self.model = _Model()
            self.model.context_size = context_size
            self.context_config = _Cfg()

    return _Agent()


# ---------- compute_block_breakpoint ----------
def test_breakpoint_is_inside_one_message() -> None:
    """一整轮就是一条消息，断点必须是消息内的 block 坐标——这正是消息级断点做不到的事。"""
    msgs = _turn(5, _UNIT)
    bp = compute_block_breakpoint(msgs, keep_recent=2)
    # 5 次工具结果在 msgs[1] 内的下标是 1,3,5,7,9；保留最近 2 次 → 断点落在第 4 次（下标 7）。
    assert bp == (1, 7)


def test_breakpoint_zero_when_too_few_tools() -> None:
    """工具结果数 ≤ keep_recent：还没有较旧区，断点 (0,0)，一个字都不压。"""
    msgs = _turn(3, _UNIT)
    assert compute_block_breakpoint(msgs, keep_recent=3) == (0, 0)


def test_breakpoint_keep_recent_zero_covers_all() -> None:
    """keep_recent<=0 → 整段可压缩（断点在末尾之后）。"""
    msgs = _turn(3, _UNIT)
    assert compute_block_breakpoint(msgs, keep_recent=0) == (len(msgs), 0)


# ---------- 压缩 ----------
def test_compress_only_before_breakpoint() -> None:
    """较旧区的大工具结果被压，最近 keep_recent 次原样。"""
    big = _payload(3000)
    msgs = _turn(4, big)
    out = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    blocks = out[1].content
    olds = [b for b in blocks[:4] if isinstance(b, ToolResultBlock)]
    recents = [b for b in blocks[4:] if isinstance(b, ToolResultBlock)]
    assert all(count_tokens(b.output[0].text) <= 500 + 50 for b in olds)
    assert all(b.output[0].text == big for b in recents)


def test_compress_does_not_mutate_input() -> None:
    """绝不原地改：入参那批对象就是 state.context，动了等于把持久化历史也截了。"""
    big = _payload(3000)
    msgs = _turn(4, big)
    original = [b.output[0].text for b in msgs[1].content if isinstance(b, ToolResultBlock)]
    post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    after = [b.output[0].text for b in msgs[1].content if isinstance(b, ToolResultBlock)]
    assert after == original


def test_untouched_messages_are_same_object() -> None:
    """没命中压缩的消息原对象返回（省拷贝，也让「视图未变」在 is 层面可验证）。"""
    msgs = _turn(4, _UNIT)  # 载荷很小，压不动
    out = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    assert all(a is b for a, b in zip(out, msgs, strict=True))


def test_output_shape_preserved_for_raw_string() -> None:
    """output 是裸字符串的 tool_result，压缩后仍是裸字符串（形态不能被压缩改写）。"""
    msgs = _turn(4, _payload(3000), raw_str=True)
    out = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    first = next(b for b in out[1].content if isinstance(b, ToolResultBlock))
    assert isinstance(first.output, str)
    assert count_tokens(first.output) <= 500 + 50


def test_smart_json_extraction_keeps_valid_json() -> None:
    """结构化工具结果走字段抽取：压完仍是合法 JSON，且保住了决策字段。"""
    candidates = [
        {
            "item_id": f"i{i}",
            "platform": "amazon",
            "title": _UNIT,
            "price_usd": 10.0 + i,
            "brand": "x",
            "description": _payload(200),
        }
        for i in range(30)
    ]
    payload = json.dumps({"platform": "amazon", "candidates": candidates}, ensure_ascii=False)
    msgs = _turn(4, payload)
    out = post_step_compress(msgs, keep_recent=2, max_tool_tokens=1500)
    text = next(b for b in out[1].content if isinstance(b, ToolResultBlock)).output[0].text
    data = json.loads(text)
    assert len(data["candidates"]) == 30
    assert data["candidates"][0]["item_id"] == "i0"
    assert "description" not in data["candidates"][0]


def test_compress_is_idempotent() -> None:
    """幂等：同一份历史压两次结果一致——缓存前缀稳定的前提。"""
    msgs = _turn(4, _payload(3000))
    once = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    twice = post_step_compress(once, keep_recent=2, max_tool_tokens=500)
    texts = [
        [b.output[0].text for b in m.content if isinstance(b, ToolResultBlock)]
        for m in (once[1], twice[1])
    ]
    assert texts[0] == texts[1]


def test_prefix_stable_across_turns() -> None:
    """再来一轮后，上一轮较旧区的视图仍是新视图的逐字前缀（滚动增量缓存的基础）。"""
    big = _payload(3000)
    msgs = _turn(4, big)
    view1 = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    prefix = [b.output[0].text for b in view1[1].content[:4] if isinstance(b, ToolResultBlock)]
    msgs[1].content.extend(_tool_blocks(9, big))
    view2 = post_step_compress(msgs, keep_recent=2, max_tool_tokens=500)
    grown = [b.output[0].text for b in view2[1].content[:4] if isinstance(b, ToolResultBlock)]
    assert grown == prefix


def test_token_growth_bounded() -> None:
    """50 次工具调用后体积有界：压缩视图远小于原文（不随轮数线性爆炸）。"""
    big = _payload(2000)
    msgs = _turn(50, big)
    out = post_step_compress(msgs, keep_recent=3, max_tool_tokens=500)
    total = sum(
        count_tokens(b.output[0].text) for b in out[1].content if isinstance(b, ToolResultBlock)
    )
    assert total < 50 * 2000 * 0.3


# ---------- formatter 的 cache_control ----------
@pytest.mark.asyncio
async def test_formatter_marks_only_system() -> None:
    """标记只落在 system 那条：位置与形态都不随轮次动，前缀才是逐字稳定的。"""
    big = _payload(MIN_CACHE_PREFIX_TOKENS + 500)
    msgs = [
        Msg(name="system", role="system", content=[TextBlock(type="text", text=big)]),
        *_turn(4, _UNIT),
    ]
    formatted = await CacheAwareOpenAIFormatter(keep_recent=2).format(msgs)
    marked = [
        i
        for i, e in enumerate(formatted)
        if isinstance(e.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in e["content"])
    ]
    assert marked == [0]
    assert all(isinstance(e["content"], str) for e in formatted if e.get("role") == "tool")


@pytest.mark.asyncio
async def test_formatter_marker_position_stable_across_turns() -> None:
    """再来几轮工具调用，标记仍钉在同一条上，且 system 之后的条目字节不受影响。"""
    big = _payload(MIN_CACHE_PREFIX_TOKENS + 500)
    system = Msg(name="system", role="system", content=[TextBlock(type="text", text=big)])
    fmt = CacheAwareOpenAIFormatter(keep_recent=2)
    short = await fmt.format([system, *_turn(2, _UNIT)])
    long = await fmt.format([system, *_turn(6, _UNIT)])
    assert short[0] == long[0]
    assert all("cache_control" not in json.dumps(e, default=str) for e in long[1:])


@pytest.mark.asyncio
async def test_formatter_skips_when_prefix_too_short() -> None:
    """system 段不足最小写入阈值不打标记——写了也不会被缓存，白占一个额度。"""
    msgs = [
        Msg(name="system", role="system", content=[TextBlock(type="text", text="短")]),
        Msg(name="user", role="user", content=[TextBlock(type="text", text="短")]),
    ]
    formatted = await CacheAwareOpenAIFormatter(keep_recent=2).format(msgs)
    assert all("cache_control" not in json.dumps(e, default=str) for e in formatted)


# ---------- 接线：pre_think Hook 与框架压缩的接管 ----------
@pytest.mark.asyncio
async def test_hook_dispatches_to_block_compression() -> None:
    """pre_think Hook 走的是 block 级压缩（消息级在这里一个字都压不掉）。"""
    from app.harness.hooks.context_compress import compress_context

    msgs = _turn(4, _payload(3000))
    ctx = {"messages": msgs}
    out = await compress_context(ctx)
    assert out is not None
    # keep_recent 取 .env 的 3，4 次工具调用里只有第 1 次落进较旧区（下标 1 的那个 block）。
    compressed = out["messages"][1].content[1]
    assert count_tokens(compressed.output[0].text) < 3000


@pytest.mark.asyncio
async def test_framework_summary_compression_is_taken_over() -> None:
    """超阈值时接管：context 就地压缩、条数不变，且**不调** next_handler（不跑框架的 LLM 摘要）。"""
    agent = _fake_agent(context_size=1000)  # 0.9 * 1000 → 12000 token 的历史必超阈值
    agent.state.context = _turn(4, _payload(3000))
    called = False

    async def next_handler(**_kwargs: object) -> None:
        nonlocal called
        called = True

    await HarnessAgentAdapter(HarnessSession()).on_compress_context(
        agent,  # type: ignore[arg-type]
        {"context_config": None, "instructions": None},
        next_handler,
    )
    assert called is False
    assert len(agent.state.context) == 2
    compressed = agent.state.context[1].content[1]
    assert count_tokens(compressed.output[0].text) < 3000


@pytest.mark.asyncio
async def test_framework_compression_noop_below_threshold() -> None:
    """未超阈值一个字都不许动——本钩子是**每轮**都被调的，不判阈值就等于每轮截历史原文。"""
    agent = _fake_agent(context_size=128000)
    agent.state.context = _turn(4, _payload(3000))
    before = [
        b.output[0].text for b in agent.state.context[1].content if isinstance(b, ToolResultBlock)
    ]

    async def next_handler(**_kwargs: object) -> None:
        raise AssertionError("不该放行框架的 LLM 摘要压缩")

    await HarnessAgentAdapter(HarnessSession()).on_compress_context(
        agent,  # type: ignore[arg-type]
        {"context_config": None, "instructions": None},
        next_handler,
    )
    after = [
        b.output[0].text for b in agent.state.context[1].content if isinstance(b, ToolResultBlock)
    ]
    assert after == before


# ---------- JSON 字段抽取 ----------
def _fake_item_search_json(n_candidates: int = 20) -> str:
    """构造一个逼真的 item_search 工具返回 JSON。"""
    candidates = []
    for i in range(n_candidates):
        candidates.append(
            {
                "item_id": f"ASIN{i:04d}",
                "platform": "amazon",
                "title": f"Travel Luggage Set {i} - Lightweight Durable Hardside Spinner",
                "brand": f"BrandName{i}",
                "price": 29.99 + i,
                "currency": "USD",
                "rating": round(3.5 + (i % 10) * 0.15, 2),
                "reviews_count": 100 + i * 50,
                "category": "Luggage > Luggage Sets > Hardside",
                "score": round(0.85 - i * 0.01, 4),
                "price_usd": 29.99 + i,
                "shipping_usd": None,
                "duty_usd": None,
                "landed_usd": None,
                "weight_kg": 3.2,
                "pick_reason": "",
            }
        )
    return json.dumps(
        {
            "platform": "amazon",
            "total_recall": n_candidates,
            "truncated": False,
            "candidates": candidates,
        },
        ensure_ascii=False,
    )


def test_smart_compress_json_extracts_fields() -> None:
    """JSON 字段抽取保留决策字段、丢弃冗余，结果是合法 JSON。"""
    raw = _fake_item_search_json(20)
    result = _smart_compress_json(raw, max_tokens=5000)
    assert result is not None
    data = json.loads(result)
    assert data["platform"] == "amazon"
    assert data["total_recall"] == 20
    assert len(data["candidates"]) == 20
    first = data["candidates"][0]
    assert "item_id" in first
    assert "title" in first
    assert "price_usd" in first
    assert "rating" in first
    # 冗余字段已丢弃
    assert "brand" not in first
    assert "score" not in first
    assert "reviews_count" not in first
    assert "weight_kg" not in first
    assert "currency" not in first


def test_smart_compress_json_smaller_than_raw() -> None:
    """字段抽取后 token 数显著低于原文。"""
    raw = _fake_item_search_json(20)
    result = _smart_compress_json(raw, max_tokens=5000)
    assert result is not None
    assert count_tokens(result) < count_tokens(raw) * 0.7


def test_smart_compress_json_returns_none_on_non_json() -> None:
    """非 JSON 文本返回 None，调用方回退到尾部截断。"""
    assert _smart_compress_json("这不是 JSON", max_tokens=100) is None


def test_smart_compress_json_returns_none_when_still_over_budget() -> None:
    """字段抽取后仍超预算时返回 None（回退到截断）。"""
    raw = _fake_item_search_json(200)
    result = _smart_compress_json(raw, max_tokens=50)
    assert result is None


def test_smart_compress_json_returns_none_for_no_candidate_keys() -> None:
    """JSON 不含候选列表字段名时返回 None。"""
    raw = json.dumps({"query": "test", "answer": "hello"})
    assert _smart_compress_json(raw, max_tokens=5000) is None


def test_smart_compress_json_skips_empty_values() -> None:
    """空值字段（None / 空串）不出现在抽取结果中。"""
    raw = _fake_item_search_json(5)
    result = _smart_compress_json(raw, max_tokens=5000)
    assert result is not None
    data = json.loads(result)
    for c in data["candidates"]:
        assert "pick_reason" not in c  # 空串被过滤
        assert "landed_usd" not in c  # None 被过滤


def test_compress_prefers_json_extraction_over_truncation() -> None:
    """较旧区的结构化工具结果走 JSON 字段抽取而非尾部截断——结果是合法 JSON，无截断提示。

    这条是压缩质量的关键：尾部截断会把 JSON 切成半截，模型既解析不了、也数不清有几件候选；
    字段抽取则丢冗余留决策字段，20 条候选一条不少。
    """
    raw = _fake_item_search_json(20)
    # 4 次调用、keep_recent=3 → 只有第 1 次（下标 1 的 block）落进较旧区被压
    msgs = _turn(4, raw)
    out = post_step_compress(msgs, keep_recent=3, max_tool_tokens=1500)
    compressed = out[1].content[1].output[0].text
    assert "已精简" not in compressed  # 没走尾部截断
    data = json.loads(compressed)  # 合法 JSON
    assert len(data["candidates"]) == 20  # 全部 20 条候选都保留了
    assert "brand" not in data["candidates"][0]  # 冗余字段丢弃


@pytest.mark.asyncio
async def test_compress_hook_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """三个压缩参数都能被 .env 覆盖（线上调参不必改代码）。"""
    from app.harness.hooks.context_compress import _compress_opts

    monkeypatch.setenv("COMPRESS_KEEP_RECENT", "2")
    monkeypatch.setenv("COMPRESS_MAX_TOOL_TOKENS", "500")
    monkeypatch.setenv("COMPRESS_CACHE_CONTROL", "true")
    assert _compress_opts() == (2, 500, True)


@pytest.mark.asyncio
async def test_compress_hook_noop_on_empty_messages() -> None:
    from app.harness.hooks.context_compress import compress_context

    assert await compress_context({"messages": []}) is None
