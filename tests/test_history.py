"""对话历史持久化的确定性测试（不依赖真实 LLM / redis）。

覆盖：
- messages 表累加写 + 回看读出 + 逐轮顺序。
- load_prior_turns 转成 (role, content) 元组 + 按 HISTORY_MAX_TURNS 截尾 + 设 0 关闭续聊。
- 惰性迁移：库上线前的 turns.json 第一次被读 / 写时整段迁进库，且不重复迁、续号不撞。
- 容错：旧 turns.json 损坏 / 单条结构异常一律降级（空 / 跳过坏条），不抛。
- save_full_trace 对 ToolMessage.artifact（自定义 Pydantic）能落盘不崩。
- run_agent 续聊接缝：同一 thread_id 第二次跑会把上一轮回喂进开局 messages。
- GET /api/history/{tid} 读出逐轮对话。

thread_id 各用例互不相同（库是共享的，靠 conftest 的 _clean_memory_tables 兜底清表）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.api.server as server
import app.memory.history as history
from app.memory.history import (
    append_turn,
    load_prior_turns,
    read_turns,
    save_full_trace,
)
from app.tools.shopping_summary import ShoppingSummaryOutput, SummaryItem


def _write_legacy(session_dir: Path, raw: Any) -> None:
    """造一份库上线前的 turns.json（惰性迁移 / 容错用例的输入）。"""
    (session_dir / "turns.json").write_text(json.dumps(raw), encoding="utf-8")


# ---------- append / read 往返 ----------
async def test_append_and_read_roundtrip() -> None:
    await append_turn("t-rt", "买个旅行包", "为你精选 1 件。")
    await append_turn("t-rt", "再便宜点的", "这有 3 件更便宜的。")
    turns = await read_turns("t-rt")
    # 两轮 = 四条消息，按 user→assistant 交替、且保留追加顺序（靠 seq，不靠时间戳）。
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "买个旅行包"
    assert turns[3]["content"] == "这有 3 件更便宜的。"


async def test_read_turns_empty_when_never_ran() -> None:
    # 从未跑过的 thread：返回空列表，不抛。
    assert await read_turns("t-never") == []


# ---------- 回看专用字段：items（商品卡） + activity（思考过程） ----------
async def test_append_persists_items_and_activity_on_assistant_turn() -> None:
    # 商品卡 + 思考过程事件挂在 assistant 轮上，read_turns 原样还原，供前端回看画卡 + 思考行。
    items = [{"item_id": "x1", "platform": "amazon", "title": "旅行包", "landed_usd": 42.0}]
    activity = [
        {"type": "monitor_event", "event": "tool_start", "data": {"tool": "item_search"}},
        {"type": "monitor_event", "event": "tool_end", "data": {"tool": "item_search"}},
    ]
    await append_turn("t-items", "买个旅行包", "为你精选 1 件。", items=items, activity=activity)
    turns = await read_turns("t-items")
    assert turns[0] == {"role": "user", "content": "买个旅行包"}
    assert turns[1]["items"] == items
    assert turns[1]["activity"] == activity


async def test_append_omits_empty_items_and_activity() -> None:
    # 闲聊兜底 / 无连接：不塞空数组，assistant 轮只剩 role+content（保持精简、契约与旧版逐字同形）。
    await append_turn("t-empty-fields", "你好", "你好呀～", items=[], activity=[])
    turns = await read_turns("t-empty-fields")
    assert turns[1] == {"role": "assistant", "content": "你好呀～"}


async def test_append_persists_elapsed_ms_and_tokens() -> None:
    # 本轮耗时 / token 用量挂在 assistant 轮上，供前端在该轮右下角显示「用时」与「token 消耗」。
    tokens = {"input": 100, "output": 20, "total": 120, "cost_usd": 0.001}
    await append_turn("t-elapsed", "买个旅行包", "为你精选 1 件。", elapsed_ms=12345, tokens=tokens)
    turns = await read_turns("t-elapsed")
    assert turns[1]["elapsed_ms"] == 12345
    assert turns[1]["tokens"] == tokens
    # 续聊回喂对它们透明，不进上下文、不增 token。
    assert await load_prior_turns("t-elapsed") == [
        ("user", "买个旅行包"),
        ("assistant", "为你精选 1 件。"),
    ]


async def test_optional_fields_omitted_when_none() -> None:
    await append_turn("t-none", "你好", "你好呀～")
    turn = (await read_turns("t-none"))[1]
    assert turn == {"role": "assistant", "content": "你好呀～"}


# ---------- load_prior_turns：续聊回喂 ----------
async def test_load_prior_turns_returns_role_content_tuples() -> None:
    await append_turn("t-prior", "q1", "a1")
    assert await load_prior_turns("t-prior") == [("user", "q1"), ("assistant", "a1")]


async def test_load_prior_turns_caps_to_recent(monkeypatch: Any) -> None:
    # 只回喂最近 N 轮：造 5 轮、上限设 2 → 取最后 2 轮（4 条消息）。
    monkeypatch.setattr(history, "HISTORY_MAX_TURNS", 2)
    for i in range(5):
        await append_turn("t-cap", f"q{i}", f"a{i}")
    prior = await load_prior_turns("t-cap")
    assert prior == [("user", "q3"), ("assistant", "a3"), ("user", "q4"), ("assistant", "a4")]


async def test_load_prior_turns_disabled_when_zero(monkeypatch: Any) -> None:
    # 上限 ≤0 等价关闭续聊：不回喂任何历史（每次全新开局）。
    monkeypatch.setattr(history, "HISTORY_MAX_TURNS", 0)
    await append_turn("t-off", "q", "a")
    assert await load_prior_turns("t-off") == []


# ---------- 惰性迁移：库上线前的 turns.json ----------
async def test_legacy_turns_json_migrated_on_read(tmp_path: Path) -> None:
    # 老会话第一次被点开：磁盘上的正文整段迁进库，顺序与可选字段都不丢。
    _write_legacy(
        tmp_path,
        [
            {"role": "user", "content": "老问题"},
            {"role": "assistant", "content": "老答案", "items": [{"item_id": "x1"}]},
        ],
    )
    turns = await read_turns("t-legacy", tmp_path)
    assert [t["content"] for t in turns] == ["老问题", "老答案"]
    assert turns[1]["items"] == [{"item_id": "x1"}]
    # 迁完只认库：这时把旧文件删掉，照样读得出来（正文的家已经换了）。
    (tmp_path / "turns.json").unlink()
    assert len(await read_turns("t-legacy", tmp_path)) == 2


async def test_legacy_migration_is_idempotent_and_new_turns_append_after(tmp_path: Path) -> None:
    # 反复读不会把老轮次导第二遍；老会话续聊时，新一轮接在老轮次之后（续号，不从 0 撞唯一约束）。
    _write_legacy(
        tmp_path,
        [{"role": "user", "content": "老问题"}, {"role": "assistant", "content": "老答案"}],
    )
    await read_turns("t-legacy-append", tmp_path)
    await read_turns("t-legacy-append", tmp_path)
    await append_turn("t-legacy-append", "新问题", "新答案", session_dir=tmp_path)
    turns = await read_turns("t-legacy-append", tmp_path)
    assert [t["content"] for t in turns] == ["老问题", "老答案", "新问题", "新答案"]


async def test_legacy_not_migrated_without_session_dir(tmp_path: Path) -> None:
    # 不给会话目录就没有迁移这回事（库里没有 = 空），不会去猜路径。
    _write_legacy(tmp_path, [{"role": "user", "content": "老问题"}])
    assert await read_turns("t-legacy-nodir") == []


# ---------- 容错：旧文件损坏不崩（迁移路径的入口，坏数据不能反噬新会话）----------
async def test_corrupt_legacy_json_degrades_to_empty(tmp_path: Path) -> None:
    (tmp_path / "turns.json").write_text("{ not json", encoding="utf-8")
    assert await read_turns("t-corrupt", tmp_path) == []
    assert await load_prior_turns("t-corrupt", tmp_path) == []


async def test_non_list_top_level_degrades_to_empty(tmp_path: Path) -> None:
    _write_legacy(tmp_path, {"role": "user"})
    assert await read_turns("t-nonlist", tmp_path) == []


async def test_skips_malformed_entries_keeps_good_ones(tmp_path: Path) -> None:
    # 一条好、三条坏（缺 content / role 非法 / content 非串）→ 只迁好的那条，不连坐整段。
    _write_legacy(
        tmp_path,
        [
            {"role": "user", "content": "ok"},
            {"role": "user"},
            {"role": "system", "content": "x"},
            {"role": "assistant", "content": 123},
        ],
    )
    assert await read_turns("t-malformed", tmp_path) == [{"role": "user", "content": "ok"}]


async def test_malformed_items_activity_ignored_not_fatal(tmp_path: Path) -> None:
    # items/activity 结构不对（非 list）静默忽略，不连坐这一轮的文本。
    _write_legacy(
        tmp_path,
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "items": "oops", "activity": {"bad": 1}},
        ],
    )
    turns = await read_turns("t-bad-fields", tmp_path)
    assert turns[1] == {"role": "assistant", "content": "a"}


async def test_elapsed_ms_bool_rejected_on_migration(tmp_path: Path) -> None:
    # 坏数据（bool 是 int 子类）不被当耗时收进来。
    _write_legacy(
        tmp_path,
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "elapsed_ms": True},
        ],
    )
    assert "elapsed_ms" not in (await read_turns("t-bad-elapsed", tmp_path))[1]


# ---------- save_full_trace：仍是文件（排障产物，刻意不进库）----------
def test_save_full_trace_handles_custom_artifact(tmp_path: Path) -> None:
    out = ShoppingSummaryOutput(
        summary="为你精选 1 件。",
        items=[SummaryItem(item_id="A1", platform="amazon", title="canvas bag")],
    )
    messages = [
        HumanMessage(content="买个旅行包"),
        ToolMessage(content=out.summary, name="shopping_summary", tool_call_id="c1", artifact=out),
        AIMessage(content="已为你整理好清单。"),
    ]
    save_full_trace(tmp_path, messages)
    data = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    # 标准 messages_to_dict 结构：列表、逐条带 type；artifact 经 _json_default 降级成 dict，不抛。
    assert isinstance(data, list) and len(data) == 3
    assert data[0]["type"] == "human"


# ---------- run_agent 续聊接缝（假 agent，无真实 LLM）----------
async def test_run_agent_continues_thread_with_prior_turns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import app.agent.main_agent as main_agent

    # 会话目录仍要重定向到 tmp（完整轨迹 history.json 还落文件），避免污染真实 output/。
    session_dir = tmp_path / "conv"
    session_dir.mkdir()
    monkeypatch.setattr(main_agent, "ensure_session_dir", lambda _tid: session_dir)
    monkeypatch.setattr(main_agent.monitor, "report_task_result", _noop)
    monkeypatch.setattr(main_agent.monitor, "report_session_created", _noop)

    captured: dict[str, Any] = {}

    def _fake_build(system_prompt: str, **_kw: Any) -> Any:
        class _FakeAgent:
            async def ainvoke(self, payload: Any, config: Any) -> dict[str, Any]:
                captured["messages"] = payload["messages"]
                # 末条是「运行时上下文（启用平台等）+ 本轮 query」，取回原始问题原样回一句。
                raw = payload["messages"][-1][1].split("用户本轮消息：\n")[-1]
                return {"messages": [AIMessage(content="answer-" + raw)]}

        return _FakeAgent()

    monkeypatch.setattr(main_agent, "_build_main_agent", _fake_build)

    # 第一轮：无历史，开局只有当前 query（外加当轮注入的运行时上下文）。
    await main_agent.run_agent("第一个问题", thread_id="t-conv", user_id=None)
    first = captured["messages"]
    assert len(first) == 1 and first[0][0] == "user"
    assert "<enabled_platforms>" in first[0][1] and first[0][1].endswith("第一个问题")

    # 第二轮（同 thread_id）：开局回喂上一轮 user→assistant，再接当前 query。
    # 回喂的历史是**干净原始** (q,a) 对——不含运行时注入，故跨轮前缀逐字稳定、能命中 prompt cache。
    await main_agent.run_agent("第二个问题", thread_id="t-conv", user_id=None)
    msgs = captured["messages"]
    assert msgs[:2] == [("user", "第一个问题"), ("assistant", "answer-第一个问题")]
    assert msgs[2][0] == "user" and msgs[2][1].endswith("第二个问题")
    # 库里已累加两轮。
    assert len(await read_turns("t-conv")) == 4


# ---------- GET /api/history/{tid} ----------
@pytest.fixture
async def client(monkeypatch: Any, tmp_path: Path) -> AsyncIterator[AsyncClient]:
    # 惰性迁移要读会话目录，把根目录重定向到 tmp，隔离真实 output/。
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_get_history_returns_turns(client: AsyncClient) -> None:
    await append_turn("t-http", "买个旅行包", "为你精选 1 件。")
    resp = await client.get("/api/history/t-http")
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == "t-http"
    assert body["turns"][0]["content"] == "买个旅行包"


async def test_get_history_migrates_legacy_session(client: AsyncClient, tmp_path: Path) -> None:
    # 老会话（正文只在 turns.json 里）点开就能看：端点顺带把它迁进库。
    sess = tmp_path / "t-http-legacy"
    sess.mkdir()
    _write_legacy(sess, [{"role": "user", "content": "老问题"}])
    resp = await client.get("/api/history/t-http-legacy")
    assert resp.status_code == 200
    assert resp.json()["turns"] == [{"role": "user", "content": "老问题"}]
    assert await read_turns("t-http-legacy") == [{"role": "user", "content": "老问题"}]


async def test_get_history_unknown_thread_returns_empty(client: AsyncClient) -> None:
    # 从未跑过的 thread：空列表而非 404（对前端是「新会话」）。
    resp = await client.get("/api/history/never-ran")
    assert resp.status_code == 200
    assert resp.json()["turns"] == []


# thread_id 的路径穿越防护复用 _safe_session_dir，已在 test_server 的 upload/download 覆盖；
# 本路由 thread_id 是单段路径参，分隔符过不了 Starlette 路由（编码斜杠会 404 而非进 handler），
# 故不在 HTTP 层重复断言，只保留 handler 里的 _safe_session_dir 守卫做纵深防御。


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


# ---------- 回看专用字段：images（用户轮的参考图）----------
async def test_append_persists_images_on_user_turn() -> None:
    """参考图文件名挂 **user** 轮：它是用户这次提问的一部分（「找这个同款」的「这个」）。

    存的是文件名不是图本体——图在 uploaded/<thread_id>/ 下，回看时前端拿名去 /api/uploads 取。
    """
    await append_turn(
        "t-img",
        "找这个同款",
        "为你精选 3 件。",
        images=["ref.png", "ref2.jpg"],
    )
    turns = await read_turns("t-img")
    assert turns[0]["role"] == "user"
    assert turns[0]["images"] == ["ref.png", "ref2.jpg"]
    # 不该渗到 assistant 轮：那轮讲的是 Agent 的产出，参考图不是它产的。
    assert "images" not in turns[1]


async def test_append_without_images_omits_key() -> None:
    """没传图的轮次不写该键——与 items/activity 同规矩：可选字段仅非空才出现，
    前端 `h.images ?? []` 直接吃空，旧数据（迁移前落的轮次）也照样读得回来。"""
    await append_turn("t-noimg", "买个旅行包", "为你精选 1 件。")
    turns = await read_turns("t-noimg")
    assert "images" not in turns[0]
