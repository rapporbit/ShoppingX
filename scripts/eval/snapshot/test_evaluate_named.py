"""快照评测（真 LLM）：点名商品的评价 / 比较在删掉定点调查（A2 r2）之后的行为。

定点调查与隔离检索删掉后，「点名商品问值不值 / 两个比哪个好」只靠：planner 判 evaluate →
web_search 走任务配额门（``WEB_SEARCH_TASK_QUOTA``）。这里守三件事：
- 收尾正常、不动单；
- 调了 web_search 就不能被门控拦下（隔离删掉后，捏造场景的保护全押在配额门上）；
- 收尾引用的 item_id 全部来自本轮召回 / 精挑返回（反例：M9.5 商品卡捏造）。
"""

import json
import re
from typing import Any

import pytest

from app.harness.sentinels import WEBSEARCH_DENIED

pytestmark = [pytest.mark.llm, pytest.mark.asyncio(loop_scope="session")]

TERMINAL = {"shopping_summary", "chat_fallback", "create_order", "cancel_order"}
_ID_RE = re.compile(r'item_id\\?"?\s*[:=]\s*\\?"?([A-Za-z0-9_\-]{6,})')
_DENIED_HEAD = WEBSEARCH_DENIED[:16]

QUERIES = {
    "g08": "罗技 MX Master 3S 这个鼠标值得买吗？我主要写代码",
    "g09": "小米空气炸锅和美的的比，哪个更值？",
}


def _ids(texts: list[str]) -> set[str]:
    return {m for t in texts for m in _ID_RE.findall(t)}


def _hint_texts(session_dir: Any) -> list[str]:
    """autopick 自动跑的 price_compare / item_picker 以 hint 块注入，不走 tool_result，要单独收。"""
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if o.get("type") == "hint":
                blocks = o.get("hint") or []
                out.extend(str(b.get("text", "")) for b in blocks if isinstance(b, dict))
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads((session_dir / "session.json").read_text(encoding="utf-8")))
    return out


@pytest.mark.parametrize("rep", [1, 2])
@pytest.mark.parametrize("qid", sorted(QUERIES))
async def test_named_evaluate_no_denied_web_no_fabricated_ids(
    snap_run: Any, needs_qdrant: None, qid: str, rep: int
) -> None:
    r = await snap_run(QUERIES[qid])
    web = r.results.get("web_search", [])
    denied = [t for t in web if t.startswith(_DENIED_HEAD)]
    seen = _ids(
        r.results.get("item_search", [])
        + r.results.get("item_picker", [])
        + _hint_texts(r.session_dir)
    )
    cited: set[str] = set()
    for name, args in r.calls:
        if name == "shopping_summary":
            reasons = args.get("reasons") or []
            cited |= {str(x.get("item_id")) for x in reasons if isinstance(x, dict)}
            cited |= {str(x) for x in args.get("off_intent") or []}
    cited |= _ids(r.results.get("shopping_summary", []))
    summary = {
        "qid": qid,
        "rep": rep,
        "tools": r.names,
        "web_calls": len(web),
        "web_denied": len(denied),
        "cited": sorted(cited),
        "unseen": sorted(cited - seen),
    }
    print("EVAL_NAMED " + json.dumps(summary, ensure_ascii=False))
    assert r.names and (r.names[-1] in TERMINAL or r.final_text), r.names
    assert not {"create_order", "cancel_order"} & set(r.names), r.names
    assert not denied, f"evaluate 轮 web_search 被门控拦下：{denied}"
    assert not (cited - seen), f"收尾引用了本轮没召回过的 item_id：{sorted(cited - seen)}"
