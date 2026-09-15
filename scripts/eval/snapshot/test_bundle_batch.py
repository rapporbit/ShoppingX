"""快照评测（真 LLM）：A3 之后套装轮由主环同轮 batch ``item_search(slot=…)``，不再派 worker。

守三件事：收尾正常不动单；不调 ``task_dispatch``；槽表 ≥2 时至少有一回合同时发出 ≥2 个
item_search（「同一回合」= session.json 里连续排列的 tool_call 块，框架据此并发）。
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.asyncio(loop_scope="session")]

TERMINAL = {"shopping_summary", "chat_fallback", "create_order", "cancel_order"}

QUERIES = {
    "q21": "旅行三件套：行李箱、旅行收纳袋、洗漱包，总预算 300，不要塑料的，喜欢耐用的",
    "q22": "新生入学一套，预算 1500",
}


def _batches(session_dir: Path) -> list[list[str]]:
    """按回合切 tool_call：连续的 tool_call 块是同一回合并发发出的一批。"""
    state = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    out: list[list[str]] = []
    for msg in state.get("context") or []:
        run: list[str] = []
        for b in msg.get("content") if isinstance(msg.get("content"), list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_call":
                run.append(str(b.get("name")))
                continue
            if run:
                out.append(run)
                run = []
        if run:
            out.append(run)
    return out


def _slots(session_dir: Path) -> list[dict[str, Any]]:
    p = session_dir / "bundle.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("slots", []) if isinstance(data, dict) else data


@pytest.mark.parametrize("rep", [1, 2])
@pytest.mark.parametrize("qid", sorted(QUERIES))
async def test_bundle_turn_batches_item_search(
    snap_run: Any, needs_qdrant: None, qid: str, rep: int
) -> None:
    r = await snap_run(QUERIES[qid])
    batches = _batches(r.session_dir)
    slots = _slots(r.session_dir)
    widest = max((b.count("item_search") for b in batches), default=0)
    summary = {
        "qid": qid,
        "rep": rep,
        "batches": batches,
        "slots": [s.get("name") for s in slots],
        "widest_item_search_batch": widest,
    }
    print("EVAL_BUNDLE " + json.dumps(summary, ensure_ascii=False))
    assert r.names and (r.names[-1] in TERMINAL or r.final_text), r.names
    assert not {"create_order", "cancel_order"} & set(r.names), r.names
    assert "task_dispatch" not in r.names, r.names
    if len(slots) >= 2 and "item_search" in r.names:
        assert widest >= 2, f"槽表 {len(slots)} 槽但没有同轮 batch item_search：{batches}"
