"""round3 刀 1：shopping_summary 的文案 / 逐件理由 / off_intent 由主模型在入参里给，工具不再起内部 LLM。

入参有 summary → 零模型调用（get_fast_llm 被换成会炸的桩，一调就红）；没给 summary → 仍走
旧的内部 LLM 路径（回归由 test_tools.py 的那批用例管）。
"""

from typing import Any

import pytest
from agentscope.message import ToolCallBlock
from agentscope.model import ChatResponse

import app.tools.shopping_summary as mod
from app.api import monitor
from app.harness import streaming
from app.tools.schemas import ItemCandidate


def _picks() -> list[dict]:
    return [
        ItemCandidate(item_id="A1", platform="amazon", title="canvas bag", landed_usd=30.0,
                      pick_reason="这几件里最便宜").model_dump(),
        ItemCandidate(item_id="A2", platform="amazon", title="nylon cubes", landed_usd=25.0,
                      pick_reason="7 件套").model_dump(),
        ItemCandidate(item_id="A3", platform="amazon", title="luggage tag", landed_usd=5.0).model_dump(),
    ]


def _boom() -> Any:
    raise AssertionError("入参已给 summary，不该再起内部 LLM")


@pytest.mark.asyncio
async def test_model_supplied_summary_skips_internal_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "get_fast_llm", _boom)
    msg = await mod.shopping_summary.ainvoke(
        {
            "name": "shopping_summary",
            "type": "tool_call",
            "id": "c1",
            "args": {
                "picks": _picks(),
                "summary": "两件收纳都对上你的要求。",
                "reasons": [{"item_id": "A1", "reason": "最便宜还耐造"}],
                "off_intent": ["A3"],
                "user_intent": "旅行收纳",
            },
        }
    )
    out = msg.artifact
    assert msg.content == "两件收纳都对上你的要求。"
    ids = [it.item_id for it in out.items]
    assert ids == ["A1", "A2"]  # off_intent 摘掉 A3（摘后仍 ≥2 件，护栏放行）
    assert out.items[0].reason == "最便宜还耐造"  # 模型写的
    assert out.items[1].reason == "7 件套"  # 漏写 → item_picker 的确定性理由兜底


@pytest.mark.asyncio
async def test_reasons_accepts_stringified_and_mapping_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "get_fast_llm", _boom)
    for reasons in ('[{"item_id":"A1","reason":"r1"}]', {"A1": "r1"}):
        out = await mod.shopping_summary.ainvoke(
            {"picks": _picks()[:2], "summary": "ok", "reasons": reasons, "user_intent": "x"}
        )
        assert out == "ok"


@pytest.mark.asyncio
async def test_stream_summary_delta_from_tool_call_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """主模型流式写 shopping_summary 入参时，summary 累计文本经 summary_delta 推给前端。"""
    sent: list[str] = []

    async def _fake_delta(text: str) -> None:
        sent.append(text)

    monkeypatch.setattr(monitor, "report_summary_delta", _fake_delta)
    partial = '{"summary": "这几件都对上你说的防泼水和 16 寸'
    chunk = ChatResponse(content=[ToolCallBlock(type="tool_call", id="t1", name="shopping_summary",
                                                input=partial)], is_last=False)
    emitted = await streaming.stream_summary_delta(chunk, 0)
    assert sent == ["这几件都对上你说的防泼水和 16 寸"] and emitted == len(sent[0])
    # 没长够 _DELTA_MIN_CHARS 不重发；别的工具不发。
    assert await streaming.stream_summary_delta(chunk, emitted) == emitted
    other = ChatResponse(content=[ToolCallBlock(type="tool_call", id="t2", name="item_search",
                                                input='{"summary": "not me, long enough text"}')], is_last=False)
    assert await streaming.stream_summary_delta(other, 0) == 0 and len(sent) == 1
