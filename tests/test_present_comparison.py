"""C4：present_comparison 横向对比工具 + 两条入口共用的实现。

主验收是**护栏**：归纳模型回的 item_id 只要不在入参集合里就整条丢弃、推荐 id 对不上就置空。
前端按 id 找列、按推荐 id 高亮，错一个就是用户可见的错位——这比没有对比更糟。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.tools.present_comparison as pc
from app.tools._candidates import _REGISTRY, register
from app.tools.present_comparison import ComparisonItem, _ComparisonDraft, compare_items
from app.tools.schemas import ItemCandidate
from app.utils.thread_ctx import thread_scope

SESSION_DIR = Path("/tmp/shoppingx-test-compare-session")


def _cand(item_id: str, **kw: object) -> ItemCandidate:
    base = {
        "item_id": item_id,
        "platform": "amazon",
        "title": f"{item_id} 背包",
        "price_usd": 100.0,
    }
    base.update(kw)
    return ItemCandidate(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clean() -> None:
    _REGISTRY.pop(str(SESSION_DIR), None)
    yield
    _REGISTRY.pop(str(SESSION_DIR), None)


def _draft(items: list[ComparisonItem], rec: str = "", reason: str = "为什么推荐它"):
    async def _call(model, prompt, schema, **kw):  # noqa: ANN001, ANN202
        return _ComparisonDraft(items=items, recommended_item_id=rec, recommendation_reason=reason)

    return _call


class TestGrounding:
    @pytest.mark.asyncio
    async def test_drops_hallucinated_ids_and_keeps_column_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """编出来的 id 整条丢；用户勾了几件就还是几列（缺席的补空壳，不少列）。"""
        monkeypatch.setattr(
            pc,
            "call_structured",
            _draft(
                [
                    ComparisonItem(item_id="a1", pros=["轻"]),
                    ComparisonItem(item_id="编造的id", pros=["不存在"]),
                ]
            ),
        )
        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1"), _cand("a2")])
            out = await compare_items(["a1", "a2"])

        assert [i.item_id for i in out.items] == ["a1", "a2"]  # 保序、列数不变
        assert out.items[0].pros == ["轻"]
        assert out.items[1].pros == []  # 模型没给 a2 → 空壳，不是把编造的那条顶上

    @pytest.mark.asyncio
    async def test_recommendation_outside_input_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """推荐 id 不在入参里 → 置空，连带理由一起撤（宁可不推荐，也不推一件表里没有的）。"""
        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        monkeypatch.setattr(
            pc, "call_structured", _draft([ComparisonItem(item_id="a1")], rec="别的商品")
        )
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1"), _cand("a2")])
            out = await compare_items(["a1", "a2"])
        assert out.recommended_item_id == ""
        assert out.recommendation_reason == ""

    @pytest.mark.asyncio
    async def test_valid_recommendation_survives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        monkeypatch.setattr(
            pc, "call_structured", _draft([ComparisonItem(item_id="a2")], rec="a2", reason="更耐造")
        )
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1"), _cand("a2")])
            out = await compare_items(["a1", "a2"])
        assert out.recommended_item_id == "a2"
        assert out.recommendation_reason == "更耐造"


class TestDeterministicParts:
    @pytest.mark.asyncio
    async def test_mixed_price_kind_note_is_computed_not_modeled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """价格口径提醒由函数侧算（看 landed_usd 在不在），不进模型的自由发挥范围。"""
        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        monkeypatch.setattr(pc, "call_structured", _draft([]))
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1", landed_usd=130.0), _cand("a2")])
            out = await compare_items(["a1", "a2"])
        assert "口径不一致" in out.note

        with thread_scope("t1", SESSION_DIR):
            register([_cand("b1", landed_usd=130.0), _cand("b2", landed_usd=90.0)])
            out = await compare_items(["b1", "b2"])
        assert out.note == ""  # 两件都是到手价 → 不提醒

    @pytest.mark.asyncio
    async def test_needs_two_known_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """取不到的 id 直接忽略；剩不到 2 件就不调模型，回一句可读的 note。"""
        called = False

        async def _never(*a, **kw):  # noqa: ANN002, ANN003, ANN202
            nonlocal called
            called = True

        monkeypatch.setattr(pc, "call_structured", _never)
        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1")])
            out = await compare_items(["a1", "查无此物"])
        assert called is False
        assert "至少 2 件" in out.note

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_not_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """归纳失败回空壳 + note，不抛——用户勾的商品还在，表照样看得见。"""

        async def _boom(*a, **kw):  # noqa: ANN002, ANN003, ANN202
            raise RuntimeError("模型挂了")

        monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
        monkeypatch.setattr(pc, "call_structured", _boom)
        with thread_scope("t1", SESSION_DIR):
            register([_cand("a1"), _cand("a2")])
            out = await compare_items(["a1", "a2"])
        assert [i.item_id for i in out.items] == ["a1", "a2"]
        assert "没能比出结论" in out.note


def test_http_entry_is_registered() -> None:
    """按钮那条确定性路径必须真在路由表上——前端只认这一个路径，写错了是上线才发现的哑火。"""
    from app.api.server import app

    routes = {
        (r.path, tuple(sorted(r.methods)))  # type: ignore[attr-defined]
        for r in app.routes
        if "compare" in getattr(r, "path", "")
    }
    assert ("/api/threads/{thread_id}/compare", ("POST",)) in routes


@pytest.fixture
async def _client(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """开着鉴权的 ASGI 客户端，输出根钉到 tmp（与 test_confirmations 同一套路）。"""
    from httpx import ASGITransport, AsyncClient

    import app.api.server as server

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path / "output")
    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test") as c:
        yield c


async def test_http_compare_roundtrip_and_ownership(  # type: ignore[no-untyped-def]
    _client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """按钮那条路走通一遍：出对比、别人碰不到、没登录进不来。

    ``test_http_entry_is_registered`` 只证路径在路由表上——路由在、归属校验漏了照样是洞。
    这条把 ``_guard_thread`` 真正打一遍：thread 属于 A，B 拿自己的 token 打同一个 thread 必须 403。
    """
    import app.tools._candidates as candidates
    from app.db.accounts import claim_thread
    from app.db.session import init_db, session_factory

    # 任务已结束时登记表为空，compare 按 id 回源商品库——这里把回源换成测试商品池。
    pool = {"c1": _cand("c1"), "c2": _cand("c2")}
    monkeypatch.setattr(
        candidates, "_fetch_from_store", lambda ids: [pool[i] for i in ids if i in pool]
    )
    monkeypatch.setattr(pc, "get_fast_llm", lambda: object())
    monkeypatch.setattr(
        pc,
        "call_structured",
        _draft([ComparisonItem(item_id="c1", pros=["轻"]), ComparisonItem(item_id="c2")], rec="c1"),
    )

    await init_db()
    resp = await _client.post(
        "/api/auth/register", json={"username": "cmp-a", "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    uid = resp.json()["user_id"]
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    other = await _client.post(
        "/api/auth/register", json={"username": "cmp-b", "password": "sup3r-secret"}
    )
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
    async with session_factory()() as db:
        await claim_thread(db, "t-cmp", uid, "对比")

    body = {"item_ids": ["c1", "c2"]}
    ok = await _client.post("/api/threads/t-cmp/compare", headers=headers, json=body)
    assert ok.status_code == 200, ok.text
    data = ok.json()
    assert [i["item_id"] for i in data["items"]] == ["c1", "c2"]
    assert data["recommended_item_id"] == "c1"

    assert (await _client.post("/api/threads/t-cmp/compare", json=body)).status_code == 401
    forbidden = await _client.post("/api/threads/t-cmp/compare", headers=other_headers, json=body)
    assert forbidden.status_code == 403, forbidden.text


def test_is_terminal_tool() -> None:
    """终结性：调完即收尾，不让模型再调 shopping_summary 把同一份判断用散文重讲。"""
    from app.agent.constants import TERMINAL_TOOLS

    assert "present_comparison" in TERMINAL_TOOLS
