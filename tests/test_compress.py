"""M6 验收：Cache Breakpoint 上下文压缩（确定性，不依赖真实 LLM）。

覆盖 ROADMAP M6 验收点「构造长对话，断点前缀稳定、token 不随轮数线性爆炸」：
- compute_breakpoint：近 K 轮留全文、较旧区可压缩；轮次太少不压。
- compress_after_breakpoint：只截断较旧区超量工具结果，近 K 轮原样、其它消息不动。
- JSON 字段抽取：结构化工具结果优先做字段精简，保留合法 JSON；非结构化回退尾部截断。
- 缓存前缀稳定：对话每加一轮，上一轮的较旧区视图仍是这一轮视图的「逐字前缀」。
- token 不线性爆炸：50 轮大工具结果后，压缩视图体积有界。
- apply_cache_control：落实 ≤4 标记 / 最小阈值 / 前缀文本不变。

注：token 阈值统一用 ``app.utils.tokens.count_tokens`` 表达（Qwen 本地分词器为主、CJK 启发式
兜底），不再写死 char/4——后者对中文低估约 3 倍。这样真分词器/启发式两条后端都自洽。
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.compress.breakpoint import compute_breakpoint
from app.compress.compressor import (
    MAX_CACHE_MARKERS,
    MIN_CACHE_PREFIX_TOKENS,
    _smart_compress_json,
    apply_cache_control,
    compress_after_breakpoint,
    mark_system_cache,
)
from app.compress.pipeline import post_step_compress
from app.harness.hooks.context_compress import _compress_opts, compress_context
from app.utils.tokens import count_tokens

# 一段真实中文短句：用它重复堆体积，避免单字符（"商"*n）被 BPE 过度合并导致 token 数不可控。
_UNIT = "想买便宜又抗造的旅行三件套，预算300，不要塑料的，喜欢小众品牌。"
_SMALL = _UNIT  # 不关心体积的轮次用它（远小于任何 cap）
# 截到 cap 后允许的 token 余量：按本段平均 字符/token 比例换算的估算式截断会有微小漂移。
_CAP_MARGIN = 50


def _payload(min_tokens: int) -> str:
    """造一段 token 数 ≳ min_tokens 的真实中文文本（重复真实短句）。"""
    s = _UNIT
    while count_tokens(s) < min_tokens:
        s += _UNIT
    return s


def _round(idx: int, payload: str) -> list:
    """造一轮对话：用户问 → AI 调工具 → 工具返回 payload。"""
    return [
        HumanMessage(content=f"第{idx}轮需求"),
        AIMessage(content="", tool_calls=[{"name": "item_search", "args": {}, "id": f"c{idx}"}]),
        ToolMessage(content=payload, tool_call_id=f"c{idx}", name="item_search"),
    ]


def _conversation(n_rounds: int, payload: str) -> list:
    msgs: list = []
    for i in range(n_rounds):
        msgs.extend(_round(i, payload))
    return msgs


# ---------- compute_breakpoint ----------
def test_breakpoint_zero_when_too_few_tools() -> None:
    """工具调用数 ≤ keep_recent 时无较旧区，断点为 0（全部留全文）。"""
    msgs = _conversation(3, _SMALL)  # 3 个工具消息
    assert compute_breakpoint(msgs, keep_recent=3) == 0


def test_breakpoint_marks_start_of_recent_window() -> None:
    """断点落在「倒数第 keep_recent 个工具消息」的下标，使近 K 轮落在 [bp:]。"""
    msgs = _conversation(5, _SMALL)  # 工具消息在下标 2,5,8,11,14
    bp = compute_breakpoint(msgs, keep_recent=3)
    assert bp == 8  # 近 3 个工具消息(8,11,14)落在 [bp:]
    tail_tools = [m for m in msgs[bp:] if isinstance(m, ToolMessage)]
    assert len(tail_tools) == 3


def test_breakpoint_keep_recent_zero_compresses_all() -> None:
    msgs = _conversation(4, _SMALL)
    assert compute_breakpoint(msgs, keep_recent=0) == len(msgs)


# ---------- compress_after_breakpoint ----------
def test_compress_truncates_only_old_oversized_tools() -> None:
    big = _payload(3000)  # 远超 1500 token 上限，确保被截断
    msgs = _conversation(5, big)
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = compress_after_breakpoint(msgs, bp, max_tool_tokens=1500)

    # 较旧区(前 2 轮的工具消息, 下标 2 和 5)被截断到上限内；近 3 轮(8,11,14)保持全文。
    assert "已精简" in out[2].content and len(out[2].content) < len(msgs[2].content)
    assert "已精简" in out[5].content
    assert count_tokens(out[2].content) <= 1500 + _CAP_MARGIN
    for i in (8, 11, 14):
        assert out[i].content == msgs[i].content  # 近 K 轮原样
    # 非工具消息一律不动。
    for i, m in enumerate(msgs):
        if not isinstance(m, ToolMessage):
            assert out[i] is m


def test_compress_passthrough_when_under_cap() -> None:
    msgs = _conversation(5, _SMALL)  # 远小于上限
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = compress_after_breakpoint(msgs, bp, max_tool_tokens=1500)
    assert all(o.content == m.content for o, m in zip(out, msgs, strict=True))


def test_compress_is_deterministic_and_idempotent() -> None:
    msgs = _conversation(6, _payload(3000))
    bp = compute_breakpoint(msgs, keep_recent=3)
    once = compress_after_breakpoint(msgs, bp, max_tool_tokens=1500)
    twice = compress_after_breakpoint(once, bp, max_tool_tokens=1500)
    assert [m.content for m in once] == [m.content for m in twice]


def test_compress_tiny_cap_still_bounds_size() -> None:
    """极小的 max_tool_tokens 下也必须真的压短（预算比提示语还短时退化为硬切、不反而变长）。"""
    msgs = _conversation(5, _payload(3000))
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = compress_after_breakpoint(msgs, bp, max_tool_tokens=1)  # cap 比提示语还短
    # 较旧区工具结果应被压到「很短」，绝不接近原文。
    assert len(out[2].content) < 200
    assert len(out[2].content) < len(msgs[2].content)


def test_compress_skips_block_list_content() -> None:
    """非字符串 content（多模态 block 列表）不被 str() 化截断，结构原样保留。"""
    blocks = [{"type": "text", "text": "x" * 9000}]
    msgs = [
        HumanMessage(content="q0"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "a"}]),
        ToolMessage(content=blocks, tool_call_id="a", name="t"),
    ] + _conversation(3, _SMALL)  # 再凑够轮次让上面的 block 工具消息落入较旧区
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = compress_after_breakpoint(msgs, bp, max_tool_tokens=100)
    assert out[2].content == blocks  # list 结构未被破坏，未被 str() 化


def test_compress_clamps_out_of_range_breakpoint() -> None:
    msgs = _conversation(3, _payload(3000))
    # 越界断点不崩；负数 → 0（不压），超长 → len（全压）。
    assert compress_after_breakpoint(msgs, -5) is not None
    out = compress_after_breakpoint(msgs, 999, max_tool_tokens=1500)
    assert any("已精简" in m.content for m in out if isinstance(m, ToolMessage))


# ---------- 缓存前缀稳定（核心验收）----------
def test_cache_prefix_is_stable_across_rounds() -> None:
    """缓存前缀不变式：上一轮的较旧区 ``[:prev_bp]`` 逐字等于这一轮视图的同位置内容。

    断点会随对话前移，所以**只有** ``[:prev_bp]`` 这段（上一轮就已冻结的较旧区）保证逐字稳定；
    断点之间 ``[prev_bp, new_bp)`` 的跨界消息会从全文被截断一次（设计取舍，见 compressor 文档），
    故意不纳入断言——它们本就不属于「逐字稳定」的承诺范围。这正是滚动增量缓存能命中的那段。
    """
    payload = _payload(3000)  # 固定同一份 payload 保证确定性
    prev_view: list | None = None
    prev_prefix_len = 0
    for n in range(4, 12):
        msgs = _conversation(n, payload)
        view = post_step_compress(msgs, keep_recent=3, max_tool_tokens=1500)
        if prev_view is not None:
            for i in range(prev_prefix_len):
                assert view[i].content == prev_view[i].content
        prev_view = view
        prev_prefix_len = compute_breakpoint(msgs, keep_recent=3)


# ---------- token 不随轮数线性爆炸 ----------
def test_total_tokens_bounded_over_many_rounds() -> None:
    """50 轮、每轮一个超大工具结果：压缩后体积应远小于「不压缩」且增长受控。"""
    rounds = 50
    msgs = _conversation(rounds, _payload(10000))  # 每条 ~10000 token，远超 1500 上限

    raw_chars = sum(len(str(m.content)) for m in msgs)
    view = post_step_compress(msgs, keep_recent=3, max_tool_tokens=1500)
    view_chars = sum(len(str(m.content)) for m in view)

    # 较旧区每条工具结果被压到 ~1500 token 上限，整体应砍掉一大截。
    assert view_chars < raw_chars * 0.3
    # 较旧区工具消息条数 ≈ rounds - keep_recent，每条 ≤ 上限 token → 总量有上界、非线性于 payload。
    old_tool_msgs = [
        m for m in view[: compute_breakpoint(msgs, keep_recent=3)] if isinstance(m, ToolMessage)
    ]
    assert all(count_tokens(m.content) <= 1500 + _CAP_MARGIN for m in old_tool_msgs)


# ---------- apply_cache_control ----------
def _has_block_marker(content: object) -> bool:
    """content 是否是带 cache_control 的 content-block 列表（显式缓存的线上格式）。"""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("cache_control") == {"type": "ephemeral"} for b in content
    )


def test_cache_control_respects_min_prefix_threshold() -> None:
    """前缀 token 不足阈值时不打标记（打了也不会被缓存）。"""
    # 5 轮小 payload，前缀远不足 1024 token。
    msgs = _conversation(5, _SMALL)
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = apply_cache_control(msgs, bp)
    assert not any(_has_block_marker(m.content) for m in out)


def test_cache_control_marks_prefix_when_large_enough() -> None:
    # 较旧区有 2 条工具消息，每条 ~600 token → 前缀 >1024 阈值。
    msgs = _conversation(5, _payload(600))
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = apply_cache_control(msgs, bp)
    marked = [i for i, m in enumerate(out) if _has_block_marker(m.content)]
    # 只打一个标记，落在缓存前缀内（断点前最后一条可标记的消息——bp-1 常是空 content 的
    # AIMessage，会回退到更前一条非空消息）。
    assert len(marked) == 1
    mi = marked[0]
    assert mi < bp
    # 标记写进 content-block（DashScope/OpenAI 兼容端点真正消费的格式），且保留原文本。
    assert out[mi].content[-1]["text"] == msgs[mi].content
    assert out[mi].content[-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_preserves_marked_text() -> None:
    """打标记把 content 从 str 改写成 block，但 block 里的文本与原文逐字一致；其它消息不动。"""
    msgs = _conversation(5, _payload(600))
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = apply_cache_control(msgs, bp)
    marked = [i for i, m in enumerate(out) if _has_block_marker(m.content)]
    assert len(marked) == 1
    mi = marked[0]
    for i, (o, m) in enumerate(zip(out, msgs, strict=True)):
        if i == mi:
            assert o.content[-1]["text"] == m.content  # 被标记条：文本不变，仅包成 block
        else:
            assert o.content == m.content  # 其它条逐字不变


def test_cache_control_respects_max_markers() -> None:
    msgs = _conversation(5, _payload(600))
    bp = compute_breakpoint(msgs, keep_recent=3)
    # 预先塞满 MAX_CACHE_MARKERS 个 block 标记 → 不再追加。
    for m in msgs[:MAX_CACHE_MARKERS]:
        m.content = [
            {"type": "text", "text": str(m.content), "cache_control": {"type": "ephemeral"}}
        ]
    out = apply_cache_control(msgs, bp)
    assert sum(_has_block_marker(m.content) for m in out) == MAX_CACHE_MARKERS


# ---------- mark_system_cache（system 段独立缓存层，refdocs/05 §4.4） ----------
def test_mark_system_cache_marks_large_system() -> None:
    """够大的 system prompt 被打上 content-block cache_control，且文本逐字保留。"""
    text = _UNIT * 60  # 远超 MIN_CACHE_PREFIX_TOKENS
    sm = SystemMessage(content=text)
    out = mark_system_cache(sm)
    assert out is not None
    assert _has_block_marker(out.content)
    assert out.content[-1]["text"] == text
    assert out.content[-1]["cache_control"] == {"type": "ephemeral"}


def test_mark_system_cache_none_passthrough() -> None:
    """入参 None（无 system message）返回 None，调用方跳过 override。"""
    assert mark_system_cache(None) is None


def test_mark_system_cache_below_threshold_skipped() -> None:
    """system 段不足最小写入阈值时不打标记（写了也不会被缓存）。"""
    sm = SystemMessage(content="短")
    assert count_tokens("短") < MIN_CACHE_PREFIX_TOKENS
    assert mark_system_cache(sm) is None


def test_mark_system_cache_idempotent() -> None:
    """已带标记的 system message 原样返回，不二次包裹（前缀字节稳定的前提）。"""
    text = _UNIT * 60
    once = mark_system_cache(SystemMessage(content=text))
    assert once is not None
    twice = mark_system_cache(once)
    assert twice is once  # 幂等：已标记直接原样返回
    assert sum(1 for b in twice.content if "cache_control" in b) == 1


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
    """较旧区的结构化工具结果走 JSON 字段抽取而非尾部截断——结果是合法 JSON，无截断提示。"""
    raw = _fake_item_search_json(20)
    msgs = [
        HumanMessage(content="第0轮需求"),
        AIMessage(content="", tool_calls=[{"name": "item_search", "args": {}, "id": "c0"}]),
        ToolMessage(content=raw, tool_call_id="c0", name="item_search"),
    ] + _conversation(3, _SMALL)
    bp = compute_breakpoint(msgs, keep_recent=3)
    out = compress_after_breakpoint(msgs, bp, max_tool_tokens=1500)
    compressed = out[2].content
    assert "已精简" not in compressed  # 没走尾部截断
    data = json.loads(compressed)  # 合法 JSON
    assert len(data["candidates"]) == 20  # 全部 20 条候选都保留了
    assert "brand" not in data["candidates"][0]  # 冗余字段丢弃


# ---------- pre_think Hook 封装 ----------
# 压缩不再是独立中间件，而是 pre_think Hook（控制面统一走 Hook Pipeline）。


@pytest.mark.asyncio
async def test_compress_hook_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPRESS_KEEP_RECENT", "2")
    monkeypatch.setenv("COMPRESS_MAX_TOOL_TOKENS", "500")
    monkeypatch.setenv("COMPRESS_CACHE_CONTROL", "true")
    assert _compress_opts() == (2, 500, True)


@pytest.mark.asyncio
async def test_compress_hook_marks_system_message_when_cache_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 开启缓存标记时，Hook 应同时改写 messages 与 system_message，且 system 段被打标。
    monkeypatch.setenv("COMPRESS_KEEP_RECENT", "3")
    monkeypatch.setenv("COMPRESS_CACHE_CONTROL", "true")
    ctx = {
        "messages": _conversation(5, _payload(600)),
        "system_message": SystemMessage(content=_UNIT * 60),
    }
    out = await compress_context(ctx)
    assert out is not None
    assert _has_block_marker(out["system_message"].content)


@pytest.mark.asyncio
async def test_compress_hook_skips_system_marking_when_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 关闭缓存标记时只压缩 messages，不碰 system_message（保持同一对象）。
    monkeypatch.setenv("COMPRESS_KEEP_RECENT", "3")
    monkeypatch.setenv("COMPRESS_CACHE_CONTROL", "false")
    system = SystemMessage(content=_UNIT * 60)
    ctx = {"messages": _conversation(5, _payload(600)), "system_message": system}
    out = await compress_context(ctx)
    assert out is not None
    assert out["system_message"] is system


@pytest.mark.asyncio
async def test_compress_hook_noop_on_empty_messages() -> None:
    assert await compress_context({"messages": []}) is None
