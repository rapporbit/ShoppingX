"""``app/eval/trace.py``：评测侧轨迹解析的**双运行时**契约（批 0 / L6）。

钉三件事：
1. 两套运行时的**内存对象**都能抽出同一串工具序列——AgentScope 是「一轮一条 assistant Msg、
   所有 tool_use block 塞在它的 content 里」，LangChain 是「一条 AIMessage 一组 tool_calls」。
   写反了症状不是报错而是**轨迹恒为空**，judge 会把每条 query 都判成「一个工具没调」。
2. 两套**落盘格式**都能读回（``Msg.model_dump()`` vs ``messages_to_dict``）。
3. 「最终回复」取的是 assistant 的正文，不能把 tool_result 的 JSON 或注入的 HintBlock 当成回复。
"""

import json

from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def _lc_turn() -> list:
    return [
        HumanMessage(content="买个旅行包"),
        AIMessage(
            content="", tool_calls=[{"name": "planner", "args": {"intent": "买包"}, "id": "1"}]
        ),
        ToolMessage(content="{...}", tool_call_id="1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "item_search", "args": {"query": "旅行包"}, "id": "2"}],
        ),
        AIMessage(content="给你清单"),
    ]


def test_agentscope_blocks_yield_ordered_calls() -> None:
    calls = extract_tool_calls(_as_turn())
    assert [c["name"] for c in calls] == ["planner", "item_search"]
    # 入参从 block 的 ``input`` 取（不是 LangChain 的 ``args``），取错会让轨迹渲染成空括号
    assert calls[1]["args"]["query"] == "旅行包"


def test_both_runtimes_agree_on_tool_sequence() -> None:
    """同一段业务过程，两套运行时抽出的工具序列必须一致——这是 L8 拿新旧链路对照的前提。"""
    assert [c["name"] for c in extract_tool_calls(_as_turn())] == [
        c["name"] for c in extract_tool_calls(_lc_turn())
    ]


def test_last_assistant_text_skips_tool_output() -> None:
    assert last_assistant_text(_as_turn()) == "先拆一下需求\n给你清单"
    assert last_assistant_text(_lc_turn()) == "给你清单"
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


def test_load_history_langchain_dump(tmp_path) -> None:
    from langchain_core.messages import messages_to_dict

    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(messages_to_dict(_lc_turn()), ensure_ascii=False), encoding="utf-8"
    )
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
