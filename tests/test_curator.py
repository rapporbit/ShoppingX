"""M3 回合后抽取（``app.memory.curator``）的确定性测试（假 LLM，无真实模型）。

断言集中在**换成事实模型之后才成立的那几条语义**上：

- 只学「下次还成立」的事实，且**封顶 3 条**；写进去的是 ``memory_facts``，不再是旧偏好表。
- 同 key 是更新（值说的是同一件事就不写）、新 key 值重复是噪声（bigram Jaccard 判重）。
- 抽取期间用户清空记忆 → 整批丢弃（``purge_generation`` 护栏）。
- 匿名、``ENABLE_MEMORY=false``、LLM 失败：都只降级返回空列表，绝不抛。
- 喂给模型的只有「已存事实 + 用户原话 + 最终回复」，**没有工具结果、没有 P_t**。
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import app.memory.curator as curator
from app.db.models import User
from app.db.session import init_db, session_factory
from app.memory.curator import CurationResult, _RecordedFact, curate_turn
from app.memory.fact_store import get_fact_store
from app.memory.facts import validate_fact
from app.utils.thread_ctx import thread_scope


class _FakeLLM:
    """AgentScope 侧假模型：只需 ``generate_structured_output``（见 invoke.call_structured）。"""

    model = "fake-fast"

    def __init__(self, payload: Any, calls: list[Any] | None = None) -> None:
        self._payload = payload
        self._calls = calls

    async def generate_structured_output(self, messages: Any, _schema: Any, **_kw: Any) -> Any:
        if self._calls is not None:
            self._calls.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        content = (
            self._payload.model_dump()
            if isinstance(self._payload, BaseModel)
            else dict(self._payload)
        )
        return SimpleNamespace(content=content, usage=None)


def _patch_llm(monkeypatch: Any, payload: Any, calls: list[Any] | None = None) -> None:
    monkeypatch.setattr(curator, "get_fast_llm", lambda: _FakeLLM(payload, calls))


def _result(*facts: tuple[str, str, str]) -> CurationResult:
    return CurationResult(facts=[_RecordedFact(key=k, value=v, category=c) for k, v, c in facts])


async def _with_user(uid: str) -> None:
    """建一个真用户行（``memory_facts.user_id`` 有外键）。``created_at`` 设成昨天的理由见
    ``tests/test_memory_facts.py``：测试库共享，用「现在」会吃掉当天注册配额。"""
    await init_db()
    async with session_factory()() as db:
        db.add(
            User(
                id=uid,
                username=f"c_{uid[:8]}",
                password_hash="x",
                created_at=datetime.now(UTC) - timedelta(days=1),
            )
        )
        await db.commit()


@pytest.mark.asyncio
async def test_writes_durable_facts_and_caps_at_three(monkeypatch: Any) -> None:
    """一轮最多学 3 条：允许更多，模型就会把本轮的颜色、价位、心情都凑成「长期偏好」。"""
    uid = uuid.uuid4().hex
    await _with_user(uid)
    _patch_llm(
        monkeypatch,
        _result(
            ("material_avoid", "不要塑料材质", "constraint"),
            ("brand_style", "偏爱小众品牌", "preference"),
            ("default_ship_to", "常寄日本", "context"),
            ("color_today", "第四条不该被写进去", "preference"),
        ),
    )

    with thread_scope("t-cur", Path(tempfile.mkdtemp()), user_id=uid):
        written = await curate_turn(
            uid, "我一直不要塑料的，偏爱小众品牌，平时寄日本", "已为你精选。"
        )

    assert [f.key for f in written] == ["material_avoid", "brand_style", "default_ship_to"]
    stored = {f.key: f.value for f in await get_fact_store().get_facts(uid)}
    assert stored["material_avoid"] == "不要塑料材质"
    assert "color_today" not in stored  # 封顶截掉的那条确实没落库
    # 写进去的事实带上本轮 thread_id，偏好页要按它溯源
    assert {f.source_session for f in written} == {"t-cur"}


@pytest.mark.asyncio
async def test_same_key_updates_and_restatement_is_dropped(monkeypatch: Any) -> None:
    """同 key 是更新；值说的是同一件事就什么都不写——否则每轮都在刷同一条的时间戳与「记住了」回执。"""
    uid = uuid.uuid4().hex
    await _with_user(uid)
    await get_fact_store().upsert_facts(uid, [validate_fact("material_avoid", "不要塑料的")])

    # 第一次：同 key、同一个意思（只差一个「的」）→ 不写
    _patch_llm(monkeypatch, _result(("material_avoid", "不要塑料", "preference")))
    with thread_scope("t-same", Path(tempfile.mkdtemp()), user_id=uid):
        assert await curate_turn(uid, "还是不要塑料", "好的。") == []

    # 第二次：同 key、改主意 → 覆盖成新值，库里仍只有一条
    _patch_llm(monkeypatch, _result(("material_avoid", "塑料也可以接受", "preference")))
    with thread_scope("t-same", Path(tempfile.mkdtemp()), user_id=uid):
        written = await curate_turn(uid, "我改主意了，塑料也行", "好的。")

    assert [f.value for f in written] == ["塑料也可以接受"]
    facts = await get_fact_store().get_facts(uid)
    assert [(f.key, f.value) for f in facts] == [("material_avoid", "塑料也可以接受")]


@pytest.mark.asyncio
async def test_new_key_restating_a_held_fact_is_dropped(monkeypatch: Any) -> None:
    """换个 key 把同一件事再写一遍 → 判重拦下。

    判重用**字符 bigram Jaccard**：参考实现按空格分词，中文整句只切出一个 token，这道闸等于不存在，
    结果是注入块很快被同义句刷满。
    """
    uid = uuid.uuid4().hex
    await _with_user(uid)
    await get_fact_store().upsert_facts(uid, [validate_fact("material_avoid", "不要塑料材质")])

    _patch_llm(monkeypatch, _result(("plastic_dislike", "不要塑料材质的", "preference")))
    with thread_scope("t-dup", Path(tempfile.mkdtemp()), user_id=uid):
        assert await curate_turn(uid, "不要塑料材质的", "好的。") == []

    assert [f.key for f in await get_fact_store().get_facts(uid)] == ["material_avoid"]


@pytest.mark.asyncio
async def test_purge_during_extraction_discards_the_whole_batch(monkeypatch: Any) -> None:
    """模型跑的这几秒里用户点了「清空」→ 整批丢弃，不给「刚清完又长回来」留缝。"""
    uid = uuid.uuid4().hex
    await _with_user(uid)
    store = get_fact_store()

    class _PurgingLLM(_FakeLLM):
        async def generate_structured_output(self, messages: Any, schema: Any, **kw: Any) -> Any:
            await store.clear(uid)  # 抽取进行中，用户清空（代数 +1）
            return await super().generate_structured_output(messages, schema, **kw)

    monkeypatch.setattr(
        curator,
        "get_fast_llm",
        lambda: _PurgingLLM(_result(("material_avoid", "不要塑料", "preference"))),
    )
    with thread_scope("t-purge", Path(tempfile.mkdtemp()), user_id=uid):
        assert await curate_turn(uid, "不要塑料", "好的。") == []

    assert await get_fact_store().get_facts(uid) == []


@pytest.mark.asyncio
async def test_prompt_sees_only_the_conversation_and_held_facts(monkeypatch: Any) -> None:
    """喂给抽取模型的只有「已存事实 + 用户原话 + 最终回复」。

    工具结果不进来是机制性的：商品标题、网页正文里写着什么「记住我是管理员」都跟用户无关，
    让它们进抽取输入等于给长期库开一道注入口。
    """
    uid = uuid.uuid4().hex
    await _with_user(uid)
    await get_fact_store().upsert_facts(uid, [validate_fact("size_shoe", "穿 42 码")])
    calls: list[Any] = []
    _patch_llm(monkeypatch, _result(), calls)

    with thread_scope("t-msg", Path(tempfile.mkdtemp()), user_id=uid):
        await curate_turn(uid, "帮我找双跑鞋", "给你三双。")

    user_msg = calls[0][-1].get_text_content() or ""
    assert "穿 42 码" in user_msg and "帮我找双跑鞋" in user_msg and "给你三双。" in user_msg
    assert "P_t" not in user_msg  # 会话级状态不再喂给它（参考实现只给事实 + 对话）


@pytest.mark.asyncio
async def test_anonymous_and_disabled_and_llm_failure_all_degrade(monkeypatch: Any) -> None:
    """三条降级路径都返回空列表且不抛——它在主回复下发后跑，不该反噬主链路。"""
    uid = uuid.uuid4().hex
    await _with_user(uid)
    calls: list[Any] = []
    _patch_llm(monkeypatch, _result(("k", "v", "preference")), calls)

    assert await curate_turn("", "不要塑料", "好的。") == []
    assert calls == []  # 匿名连 LLM 都不调

    monkeypatch.setenv("ENABLE_MEMORY", "false")
    assert await curate_turn(uid, "不要塑料", "好的。") == []
    assert calls == []  # 开关关掉同样不调
    monkeypatch.delenv("ENABLE_MEMORY")

    _patch_llm(monkeypatch, RuntimeError("boom"))
    with thread_scope("t-fail", Path(tempfile.mkdtemp()), user_id=uid):
        assert await curate_turn(uid, "不要塑料", "好的。") == []
    assert await get_fact_store().get_facts(uid) == []
