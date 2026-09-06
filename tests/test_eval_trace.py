"""``app/eval/trace.py``：评测侧的轨迹解析契约。

钉三件事：
1. 当前形态（AgentScope）的内存对象能抽出工具序列——「一轮一条 assistant Msg、所有
   tool_call block 塞在它的 content 里」。写反了症状不是报错而是**轨迹恒为空**，judge 会把
   每条 query 都判成「一个工具没调」。
2. **历史落盘格式也要能读回**：线上还躺着迁移前用 ``messages_to_dict`` 落的 history.json，
   评测取样与 few-shot 蒸馏都会去读它们。这里用固化的 dump 字面量来验，不依赖旧框架包。
3. 「最终回复」取的是 assistant 的正文，不能把 tool_result 的 JSON 或注入的 HintBlock 当成回复。
"""

import json

from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock

from app.eval.trace import (
    extract_tool_calls,
    last_assistant_text,
    load_history,
    normalize_messages,
)


def _as_turn() -> list[Msg]:
    """一条真实形态的 AgentScope 轨迹：用户一条 + assistant 一条（一轮全在里面）。"""
    return [
        Msg(name="user", role="user", content=[TextBlock(type="text", text="买个旅行包")]),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                TextBlock(type="text", text="先拆一下需求"),
                # ``input`` 是 **JSON 字符串**，不是 dict——流式解析按文本增量拼出来的
                ToolCallBlock(id="1", name="planner", input='{"intent": "买包"}'),
                ToolResultBlock(
                    id="1",
                    name="planner",
                    output=[TextBlock(type="text", text='{"category": "背包"}')],
                ),
                ToolCallBlock(
                    id="2",
                    name="item_search",
                    input='{"query": "旅行包", "platform": "all"}',
                ),
                TextBlock(type="text", text="给你清单"),
            ],
        ),
    ]


# 迁移前落盘的 history.json 长这样（``messages_to_dict`` 的产物，此处已固化为字面量：
# 旧格式是既成事实、不会再变，为它保留一个框架依赖不值得）。
_LEGACY_DUMP: list[dict] = [
    {"type": "human", "data": {"content": "买个旅行包", "type": "human"}},
    {
        "type": "ai",
        "data": {
            "content": "",
            "type": "ai",
            "tool_calls": [
                {"name": "planner", "args": {"intent": "买包"}, "id": "1", "type": "tool_call"}
            ],
        },
    },
    {"type": "tool", "data": {"content": "{...}", "type": "tool", "tool_call_id": "1"}},
    {
        "type": "ai",
        "data": {
            "content": "",
            "type": "ai",
            "tool_calls": [
                {
                    "name": "item_search",
                    "args": {"query": "旅行包"},
                    "id": "2",
                    "type": "tool_call",
                }
            ],
        },
    },
    {"type": "ai", "data": {"content": "给你清单", "type": "ai"}},
]


def test_agentscope_blocks_yield_ordered_calls() -> None:
    calls = extract_tool_calls(_as_turn())
    assert [c["name"] for c in calls] == ["planner", "item_search"]
    # 入参从 block 的 ``input`` 取（不是 LangChain 的 ``args``），取错会让轨迹渲染成空括号
    assert calls[1]["args"]["query"] == "旅行包"


def test_current_and_legacy_formats_agree_on_tool_sequence() -> None:
    """同一段业务过程，新形态与历史落盘格式抽出的工具序列必须一致。

    这是拿迁移前后的评测报告做对照的前提：序列对不上，「迁移后少调了一个工具」这种结论就
    分不清是真退化还是解析口径变了。
    """
    assert [c["name"] for c in extract_tool_calls(_as_turn())] == [
        c["name"] for c in extract_tool_calls(normalize_messages(_LEGACY_DUMP))
    ]


def test_last_assistant_text_skips_tool_output() -> None:
    assert last_assistant_text(_as_turn()) == "先拆一下需求\n给你清单"
    assert last_assistant_text(normalize_messages(_LEGACY_DUMP)) == "给你清单"
    assert last_assistant_text([]) == ""


def test_load_history_agentscope_dump(tmp_path) -> None:
    path = tmp_path / "history.json"
    # ``default=str`` 与 orchestrator._save_trace 一致：枚举（ToolCallState）不可直接序列化
    path.write_text(
        json.dumps([m.model_dump() for m in _as_turn()], ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    msgs = load_history(path)
    assert [c["name"] for c in extract_tool_calls(msgs)] == ["planner", "item_search"]
    assert "给你清单" in last_assistant_text(msgs)


def test_load_history_legacy_dump(tmp_path) -> None:
    """迁移前落的 history.json 照样读得回——旧会话的评测取样不能因为换运行时就全废。"""
    path = tmp_path / "history.json"
    path.write_text(json.dumps(_LEGACY_DUMP, ensure_ascii=False), encoding="utf-8")
    msgs = load_history(path)
    assert [c["name"] for c in extract_tool_calls(msgs)] == ["planner", "item_search"]
    assert last_assistant_text(msgs) == "给你清单"


def test_normalize_is_idempotent() -> None:
    """``extract_tool_calls(load_history(...))`` 是蒸馏脚本的真实用法——归一两遍不许洗空。"""
    once = normalize_messages(_as_turn())
    twice = normalize_messages(once)
    assert twice == once
    assert [c["name"] for c in extract_tool_calls(once)] == ["planner", "item_search"]


def test_missing_history_is_empty_not_raise(tmp_path) -> None:
    """蒸馏脚本会对每条高分记录去猜 ``output/eval_<id>/history.json``，缺文件是常态不是错误。"""
    assert load_history(tmp_path / "nope.json") == []
    assert extract_tool_calls(None) == []
    assert normalize_messages([]) == []
