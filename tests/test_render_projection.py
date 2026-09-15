"""round3 刀 5：工具返回给模型的文本走 Output 的紧凑投影（``__str__``），没定义投影的照旧全量 JSON。

迁 AgentScope 后 ``_to_text`` 曾一律 ``model_dump_json``，item_search / item_picker / price_compare /
shipping_calc 的紧凑投影全成了死代码（一条 item_search 结果 12k 字符里近半是 url / null 字段）。
"""

import json

from app.tools._shell import _to_text
from app.tools.item_picker import ItemPickerOutput
from app.tools.item_search import ItemSearchOutput
from app.tools.schemas import ItemCandidate
from app.tools.shopping_summary import ShoppingSummaryOutput


def test_to_text_uses_compact_projection_when_defined() -> None:
    c = ItemCandidate(
        item_id="A1",
        platform="amazon",
        title="t" * 100,
        price=20,
        currency="USD",
        price_usd=20.0,
        url="http://x",
        image_url="http://img",
        rating=4.5,
        pick_reason="最便宜",
    )
    text = _to_text(ItemPickerOutput(picks=[c], excluded=[], over_budget=[]))
    data = json.loads(text)
    pick = data["picks"][0]
    assert "url" not in pick and "image_url" not in pick  # 前端字段不喂模型
    assert "shipping_usd" not in pick  # 未填充的 null 不烧 token
    assert pick["pick_reason"] == "最便宜" and pick["item_id"] == "A1"  # 决策要用的留着
    assert len(pick["title"]) < 100  # 回显标题截短


def test_to_text_falls_back_to_full_json_without_projection() -> None:
    out = ShoppingSummaryOutput(summary="ok", items=[])
    assert json.loads(_to_text(out)) == {"summary": "ok", "items": []}


def _cand(**kw) -> ItemCandidate:
    base = dict(
        item_id="A1", platform="amazon", title="t", price=20, currency="USD", price_usd=20.0
    )
    return ItemCandidate(**{**base, **kw})


def test_projections_round_trip_through_output_schema() -> None:
    """投影裁掉的只能是**可推导的冗余**：渲染串回填顶层 platform 后必须仍通过完整 *Output 验证。

    曾是运行时 schema 断言（harness validation hook，2026-09-15 删）：它验的是自家渲染器与
    schema 一致，模型改不了工具返回格式，该在这里当单测。
    """
    picker = ItemPickerOutput(picks=[_cand(pick_reason="r")], excluded=[], over_budget=[])
    ItemPickerOutput.model_validate(json.loads(_to_text(picker)))

    search = ItemSearchOutput(
        platform="amazon", candidates=[_cand()], total_recall=1, truncated=False
    )
    data = json.loads(_to_text(search))
    for c in data["candidates"]:
        c.setdefault("platform", data["platform"])  # 单平台渲染省略候选级 platform，顶层已写
    ItemSearchOutput.model_validate(data)
