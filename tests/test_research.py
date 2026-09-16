"""C2：research 有界研究函数。

四条验收对应四组测试：① 每 target 一条查询、aspects 合进同一条（不额外发搜索）；
② 主环返回**不含网页正文**、原始结果落盘可回取；③ url 对不上本次结果的 claim 被丢弃；
④ 全降级时不白花一次归纳解码。
"""

import json

import pytest

import app.tools.research as rs
from app.tools.research import ResearchClaim, ResearchFinding, _ResearchDraft
from app.tools.web_search import WebResult, WebSearchOutput

_BODY = "这段是网页正文，主环不该看到它。" * 20


def _fake_search(calls: list[str], *, empty: bool = False, degraded: bool = False):
    async def _search(query: str, max_results: int = 5) -> tuple[WebSearchOutput, bool]:
        calls.append(query)
        if empty:
            return WebSearchOutput(query=query, results=[], note="外部检索不可用"), degraded
        results = [
            WebResult(title=f"{query} 评测", url=f"https://ex.com/{len(calls)}", content=_BODY)
        ]
        return WebSearchOutput(query=query, results=results), degraded

    return _search


def _fake_draft(findings: list[ResearchFinding]):
    async def _call(model, prompt, schema, **kw) -> _ResearchDraft:  # noqa: ANN001
        return _ResearchDraft(findings=findings)

    return _call


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path):  # noqa: ANN001, ANN201
    monkeypatch.setattr(rs, "get_fast_llm", lambda: object())
    monkeypatch.setattr(rs, "get_session_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_one_query_per_target_aspects_merged(
    monkeypatch: pytest.MonkeyPatch, patched
) -> None:
    """每 target 一条查询、aspects 合进同一条 —— 搜索条数上界 = len(targets)，事前可知。"""
    calls: list[str] = []
    monkeypatch.setattr(rs, "search_web", _fake_search(calls))
    monkeypatch.setattr(rs, "call_structured", _fake_draft([]))

    out = await rs.research.ainvoke(
        {"targets": ["Sony XM5", "Bose QC45"], "aspects": ["降噪", "续航"]}
    )

    assert len(calls) == 2 == out.searched
    assert all("降噪" in q and "续航" in q for q in calls)  # aspects 进查询词，不额外发搜索
    assert "Sony XM5" in calls[0] and "Bose QC45" in calls[1]


@pytest.mark.asyncio
async def test_targets_truncated_not_rejected(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    """超出 RESEARCH_MAX_TARGETS 截断而非整条失败 —— 回 3 个比报错有用。"""
    calls: list[str] = []
    monkeypatch.setattr(rs, "search_web", _fake_search(calls))
    monkeypatch.setattr(rs, "call_structured", _fake_draft([]))

    out = await rs.research.ainvoke({"targets": ["a", "b", "c", "d", "e"]})

    assert len(calls) == rs.RESEARCH_MAX_TARGETS == len(out.targets)


@pytest.mark.asyncio
async def test_body_not_in_output_but_on_disk(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    """C2 主验收：正文不进主环，但盘上有原文 —— 两边都断言。"""
    calls: list[str] = []
    monkeypatch.setattr(rs, "search_web", _fake_search(calls))
    finding = ResearchFinding(
        target="Sony XM5",
        pros=["降噪强"],
        claims=[ResearchClaim(text="降噪业界第一", url="https://ex.com/1")],
    )
    monkeypatch.setattr(rs, "call_structured", _fake_draft([finding]))

    out = await rs.research.ainvoke({"targets": ["Sony XM5"]})

    assert _BODY not in out.model_dump_json()  # 上下文里没有
    saved = json.loads((patched / out.raw_path).read_text(encoding="utf-8"))
    assert _BODY in saved["packs"][0]["results"][0]["content"]  # 盘上有原文


@pytest.mark.asyncio
async def test_ungrounded_claim_dropped(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    """url 不在本次搜索结果里的 claim 丢弃（防归纳模型凑出处），但 finding 与 pros 保留。"""
    calls: list[str] = []
    monkeypatch.setattr(rs, "search_web", _fake_search(calls))
    finding = ResearchFinding(
        target="Sony XM5",
        pros=["降噪强"],
        claims=[
            ResearchClaim(text="真实出处", url="https://ex.com/1"),
            ResearchClaim(text="编的出处", url="https://fake.com/x"),
            ResearchClaim(text="没有出处", url=""),
        ],
    )
    monkeypatch.setattr(rs, "call_structured", _fake_draft([finding]))

    out = await rs.research.ainvoke({"targets": ["Sony XM5"]})

    assert [c.text for c in out.findings[0].claims] == ["真实出处"]
    assert out.findings[0].pros == ["降噪强"]  # 只删 claim 不删 finding
    assert "丢弃 2 条" in out.note


@pytest.mark.asyncio
async def test_all_empty_skips_summarizer(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    """全降级 / 全空时不调归纳模型 —— 没资料还解码一次是白花钱。"""
    calls: list[str] = []
    called = False

    async def _boom(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        nonlocal called
        called = True
        raise AssertionError("不该调归纳模型")

    monkeypatch.setattr(rs, "search_web", _fake_search(calls, empty=True, degraded=True))
    monkeypatch.setattr(rs, "call_structured", _boom)

    out = await rs.research.ainvoke({"targets": ["Sony XM5"]})

    assert called is False and out.findings == []
    assert out.note == "外部检索不可用"


@pytest.mark.asyncio
async def test_empty_targets_returns_note(patched) -> None:  # noqa: ANN001
    out = await rs.research.ainvoke({"targets": []})
    assert out.findings == [] and "targets" in out.note


@pytest.mark.asyncio
async def test_empty_findings_carries_a_note(monkeypatch: pytest.MonkeyPatch, patched) -> None:
    """搜到了、归纳没报错、但模型回了空 findings —— 必须带 note，不能回一个沉默的空壳。

    这是 C2/C3 快照验收（r02_versus 首次调用）撞到的真实形态：额度照扣、findings 为空、note
    也为空，主 agent 只能猜「是没资料还是没归纳出来」，实测它原样重调了一次，两条额度白花。
    """
    calls: list[str] = []
    monkeypatch.setattr(rs, "search_web", _fake_search(calls))
    monkeypatch.setattr(rs, "call_structured", _fake_draft([]))

    out = await rs.research.ainvoke({"targets": ["Sony XM5", "Bose QC45"]})

    assert out.searched == 2 and out.findings == []
    assert "findings 为空" in out.note and "raw_path" in out.note
