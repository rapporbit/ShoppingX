"""M1 的记忆事实模型与存取口。

断言集中在**换模型之后才成立的那几条语义**上：同 key 覆盖（而非叠加）、constraint 不受注入 cap 限制、
PII 拒写且不回显 value、中文匹配不靠空格切词、清空要连代数一起加。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.db.models import User
from app.db.session import init_db, session_factory
from app.memory.fact_store import MemoryFactStore
from app.memory.facts import (
    MemoryCategory,
    MemoryFact,
    MemoryWriteRejected,
    match_facts,
    render_memory_block,
    same_fact,
    select_tier_one_facts,
    validate_fact,
)


def _fact(key: str, value: str, category: str = "preference", *, age_days: int = 0) -> MemoryFact:
    return MemoryFact(
        key=key,
        value=value,
        category=MemoryCategory(category),
        updated_at=datetime.now(UTC) - timedelta(days=age_days),
    )


async def _with_user(uid: str) -> None:
    """建一个真用户行——``memory_facts.user_id`` 有外键，没有这行写不进去。

    ``created_at`` 显式设成昨天：``ratelimit.guard_daily_signups`` 直接 COUNT 当天新建的 users
    行，而测试库是整套测试共享的——用默认「现在」建几行就会把当天注册名额吃光，害得
    ``test_ratelimit`` 里一个毫不相干的用例 429（只在全量跑时复现，单跑那个文件永远是绿的）。
    """
    await init_db()
    async with session_factory()() as db:
        db.add(
            User(
                id=uid,
                username=f"t_{uid[:8]}",
                password_hash="x",
                created_at=datetime.now(UTC) - timedelta(days=1),
            )
        )
        await db.commit()


def test_validate_normalizes_key_and_defaults_category() -> None:
    fact = validate_fact("  Ship To ", "  常寄\n德国  ", "context")
    assert fact.key == "ship_to"
    assert fact.value == "常寄 德国"  # 换行压成空格：一条事实只占一行
    assert fact.category is MemoryCategory.CONTEXT
    # 非法分类退到 preference，而不是整条丢掉
    assert validate_fact("k", "v", "nonsense").category is MemoryCategory.PREFERENCE


@pytest.mark.parametrize(
    "value",
    ["手机 13800138000", "卡号 4111-1111-1111-1111", "a@b.com", "DE89370400440532013000"],
)
def test_pii_is_rejected_without_echoing_value(value: str) -> None:
    with pytest.raises(MemoryWriteRejected) as exc:
        validate_fact("contact", value)
    assert value not in str(exc.value)


def test_short_numbers_survive() -> None:
    """价格 / 尺码 / 年份不该被 9 位数字那条规则误伤。"""
    assert validate_fact("budget", "预算 300 美元以内").value == "预算 300 美元以内"
    assert validate_fact("shoe_size", "42 码").value == "42 码"


def test_fence_markers_are_stripped() -> None:
    fact = validate_fact("style", '<external_content source="web_search">喜欢小众</external_content>')
    assert "external_content" not in fact.value


def test_tier_one_keeps_every_constraint_over_cap() -> None:
    facts = [_fact(f"c{i}", f"硬规则{i}", "constraint") for i in range(10)]
    facts += [_fact(f"p{i}", f"取向{i}", age_days=i) for i in range(5)]
    selected = select_tier_one_facts(facts, cap=8)
    assert len([f for f in selected if f.category is MemoryCategory.CONSTRAINT]) == 10
    assert not [f for f in selected if f.category is MemoryCategory.PREFERENCE]


def test_tier_one_fills_rest_by_recency() -> None:
    facts = [_fact(f"p{i}", f"取向{i}", age_days=i) for i in range(5)]
    assert [f.key for f in select_tier_one_facts(facts, cap=2)] == ["p0", "p1"]


def test_match_and_same_fact_work_without_spaces() -> None:
    facts = [_fact("material_plastic", "不要塑料的", "constraint")]
    assert match_facts(facts, "塑料")  # 中文子串命中，不靠空格切词
    assert not match_facts(facts, "皮革")
    assert same_fact("不要塑料的", "塑料的不要")
    assert not same_fact("喜欢小众品牌", "预算三百以内")


def test_render_block_is_empty_when_no_facts() -> None:
    assert render_memory_block([]) == ""
    assert "[constraint] k: v" in render_memory_block([_fact("k", "v", "constraint")])


@pytest.mark.asyncio
async def test_store_overwrites_by_key_and_clears_with_generation() -> None:
    store = MemoryFactStore()
    uid = uuid.uuid4().hex
    await _with_user(uid)

    await store.upsert_facts(uid, [_fact("no_plastic", "不要塑料的", "constraint")])
    await store.upsert_facts(uid, [_fact("no_plastic", "塑料也可以")])
    facts = await store.get_facts(uid)
    assert len(facts) == 1  # 覆盖，不是两条互相矛盾的事实并存
    assert facts[0].value == "塑料也可以"
    assert facts[0].category is MemoryCategory.PREFERENCE

    assert await store.delete_fact(uid, "no_plastic") is True
    assert await store.delete_fact(uid, "no_plastic") is False  # 已经没了

    await store.upsert_facts(uid, [_fact("style", "喜欢小众品牌")])
    assert await store.purge_generation(uid) == 0
    await store.clear(uid)
    assert await store.get_facts(uid) == []
    assert await store.purge_generation(uid) == 1  # 清空必须连代数一起加


# ============================================================
# M2：save_memory 工具 / 收货国 C1 / 围栏
# ============================================================


@pytest.mark.asyncio
async def test_save_memory_writes_and_overwrites_by_key() -> None:
    """用户当场说「记住 X」→ 当轮就落库、有回执；改主意时用同 key 覆盖，不留两条打架的。"""
    import tempfile
    from pathlib import Path

    from app.memory.fact_store import get_fact_store
    from app.tools.save_memory import save_memory
    from app.utils.thread_ctx import thread_scope

    uid = uuid.uuid4().hex
    await _with_user(uid)
    with thread_scope("t-save", Path(tempfile.mkdtemp()), user_id=uid):
        out = await save_memory.ainvoke(
            {"key": "Material Avoid", "value": "不要塑料的", "category": "constraint"}
        )
        assert out.saved and out.key == "material_avoid"  # key 规范化后才是身份
        assert "已记住" in out.note

        # 遗忘走覆盖，不走删除：模型手里没有删除口。
        await save_memory.ainvoke({"key": "material_avoid", "value": "塑料也可以"})
        facts = await get_fact_store().get_facts(uid)
    assert [f.value for f in facts] == ["塑料也可以"]


@pytest.mark.asyncio
async def test_save_memory_rejects_pii_without_echoing_value() -> None:
    """PII 被写入单门挡下时，回执里**不能回显 value**——被拒的多半正是不该扩散的东西。"""
    import tempfile
    from pathlib import Path

    from app.memory.fact_store import get_fact_store
    from app.tools.save_memory import save_memory
    from app.utils.thread_ctx import thread_scope

    uid = uuid.uuid4().hex
    await _with_user(uid)
    with thread_scope("t-save-pii", Path(tempfile.mkdtemp()), user_id=uid):
        out = await save_memory.ainvoke({"key": "phone", "value": "我的手机 13800138000"})
        assert not out.saved and "13800138000" not in out.note
        assert await get_fact_store().get_facts(uid) == []


@pytest.mark.asyncio
async def test_save_memory_anonymous_says_so_and_write_failure_does_not_lie() -> None:
    """匿名会话如实说要登录；库挂了也**不能说「已记住」**——用户据此不再说第二遍，这条就永远丢了。"""
    import tempfile
    from pathlib import Path

    from app.memory import fact_store as fs
    from app.tools.save_memory import save_memory
    from app.utils.thread_ctx import thread_scope

    with thread_scope("t-save-anon", Path(tempfile.mkdtemp())):
        anon = await save_memory.ainvoke({"key": "k", "value": "v"})
    assert not anon.saved and "匿名" in anon.note

    uid = uuid.uuid4().hex
    await _with_user(uid)

    class _DeadStore(fs.MemoryFactStore):
        async def upsert_facts(self, user_id: str, facts: list[MemoryFact]) -> bool:
            return False  # 库挂了：store 吞掉异常、返回 False

    original = fs._store
    fs._store = _DeadStore()
    try:
        with thread_scope("t-save-dead", Path(tempfile.mkdtemp()), user_id=uid):
            out = await save_memory.ainvoke({"key": "k", "value": "不要塑料"})
    finally:
        fs._store = original
    assert not out.saved and "已记住" not in out.note


@pytest.mark.asyncio
async def test_dest_country_reads_ship_to_fact() -> None:
    """C1：收货国第 3 层按 ``SHIP_TO_KEY`` 取事实。旧代码读的 category/polarity 已随 M1 消失，
    不改这里不会报错、只会静默退回默认国，到手价按错国家算——所以这条测试盯的是**不报错的那种错**。
    """
    import tempfile
    from pathlib import Path

    from app.memory.fact_store import get_fact_store
    from app.memory.facts import SHIP_TO_KEY
    from app.tools.planner import resolve_dest_country_layered
    from app.utils.thread_ctx import thread_scope

    uid = uuid.uuid4().hex
    await _with_user(uid)
    with thread_scope("t-ship", Path(tempfile.mkdtemp()), user_id=uid):
        await get_fact_store().upsert_facts(
            uid, [validate_fact(SHIP_TO_KEY, "常寄德国", "context")]
        )
        country, assumed, stated_now = await resolve_dest_country_layered("买个背包")
        assert (country, assumed, stated_now) == ("DE", False, False)

        # 本轮原话仍然压过长期记忆（第 1 层 > 第 3 层）。
        explicit, _, stated = await resolve_dest_country_layered("买个背包，寄到日本")
        assert (explicit, stated) == ("JP", True)


def test_recall_memories_output_is_fenced() -> None:
    """召回的记忆正文源头是用户某一轮说的话：一条被写进去的「忽略以上指令」会在此后每次召回时
    重放。它必须和网页正文一样进围栏白名单，模型才分得清「数据」与「指令」。"""
    from app.security.content_filter import EXTERNAL_SOURCE_TOOLS

    assert "recall_memories" in EXTERNAL_SOURCE_TOOLS


def test_save_memory_is_allowed_without_confirm_prompt() -> None:
    """写工具默认被 PermissionEngine 挂起等确认。save_memory 要进放行表，否则用户说「记住 X」
    时主链路会停在半路等一个前端根本没有的确认按钮。"""
    from app.agent.permissions import DEFAULT_ALLOWED_TOOLS
    from app.agent.tool_registry import TOOLS_BY_NAME

    assert "save_memory" in TOOLS_BY_NAME  # 进了工具面
    assert TOOLS_BY_NAME["save_memory"].is_read_only is False  # 它是写工具，别标成只读
    assert "save_memory" in DEFAULT_ALLOWED_TOOLS


# ---------------------------------------------------------------------------
# M3：保留期与部署开关
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retention_hides_expired_facts_and_drops_them_on_next_write() -> None:
    """超龄的事实立刻读不到，并在该用户下次写入时被删掉。

    「读不到」必须先于「删掉」生效：一条两年前的「常寄德国」还出现在注入块里，比它留在库里
    危害大得多。而清理挂在写入上，是因为记忆按用户切开、没有扫全库的必要。
    """
    from datetime import timedelta

    from app.memory.fact_store import MemoryFactStore, RetentionMemoryStore

    uid = uuid.uuid4().hex
    await _with_user(uid)
    inner = MemoryFactStore()
    old = validate_fact("ship_to", "常寄德国", "context")
    old.updated_at = datetime.now(UTC) - timedelta(days=400)
    await inner.upsert_facts(uid, [old])

    retained = RetentionMemoryStore(inner, timedelta(days=30))
    assert await retained.get_facts(uid) == []  # 超龄：读不到
    assert await retained.search_facts(uid, "德国") == []
    assert len(await inner.get_facts(uid)) == 1  # 但还在库里，等下一次写入

    await retained.upsert_facts(uid, [validate_fact("brand_style", "偏爱小众品牌")])
    assert [f.key for f in await inner.get_facts(uid)] == ["brand_style"]


def test_with_retention_is_off_by_default_and_wraps_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MEMORY_RETENTION_DAYS`` 不设或 <=0 = 不限期，直接用原 store，不白套一层。"""
    from app.memory.fact_store import RetentionMemoryStore, get_fact_store

    monkeypatch.delenv("MEMORY_RETENTION_DAYS", raising=False)
    assert not isinstance(get_fact_store(), RetentionMemoryStore)

    monkeypatch.setenv("MEMORY_RETENTION_DAYS", "30")
    store = get_fact_store()
    assert isinstance(store, RetentionMemoryStore)
    assert store.retention == timedelta(days=30)


@pytest.mark.asyncio
async def test_enable_memory_off_silences_both_tools_and_the_injection() -> None:
    """``ENABLE_MEMORY=false``：两个工具回一句明确的「这个部署没开记忆」，注入一条都不发。

    工具回执必须明确——含糊的失败模型会重试，明确的关闭它会转述给用户。
    """
    import os
    import tempfile
    from pathlib import Path

    from app.harness.hooks.context_shaping import inject_long_term_memory
    from app.memory.facts import MEMORY_DISABLED_TEXT
    from app.tools.recall_memories import recall_memories
    from app.tools.save_memory import save_memory
    from app.utils.thread_ctx import thread_scope

    uid = uuid.uuid4().hex
    await _with_user(uid)
    await MemoryFactStore().upsert_facts(uid, [validate_fact("material_avoid", "不要塑料")])

    os.environ["ENABLE_MEMORY"] = "false"
    try:
        with thread_scope("t-off", Path(tempfile.mkdtemp()), user_id=uid):
            saved = await save_memory.ainvoke({"key": "k", "value": "v"})
            recalled = await recall_memories.ainvoke({"topic": ""})
            ctx: dict = {"tool_name": "planner"}
            assert await inject_long_term_memory(ctx) is None
    finally:
        os.environ.pop("ENABLE_MEMORY", None)

    assert not saved.saved and saved.note == MEMORY_DISABLED_TEXT
    assert recalled.count == 0 and recalled.note == MEMORY_DISABLED_TEXT
    assert "inject_messages" not in ctx  # 连空占位都不塞
    # 关开关不动库：重新打开后那条事实还在
    assert [f.key for f in await MemoryFactStore().get_facts(uid)] == ["material_avoid"]
