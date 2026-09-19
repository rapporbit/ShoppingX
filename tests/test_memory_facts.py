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
    await init_db()
    store = MemoryFactStore()
    uid = uuid.uuid4().hex
    async with session_factory()() as db:
        db.add(User(id=uid, username=f"t_{uid[:8]}", password_hash="x"))
        await db.commit()

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
