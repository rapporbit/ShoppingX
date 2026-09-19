"""长期记忆的事实模型与写入单门。

对照 `commerce-agents`（`commerce_common/memory.py`）重建，取代 `PreferenceEntry` 那套
polarity / domain / slug / blocking 的建模。三处不同是本仓主动改的，不是照抄漏了：

1. **匹配口径不用 `split()`**。参考实现按空格切词再判包含，中文没有词间空格，一句「不要塑料的」
   会切成一个 token，与 key `material_plastic` 永远匹配不上。这里保留子串匹配。
2. **同一事实的判重用字符 bigram Jaccard**，理由同上（M3 会用到，先放这里）。
3. **围栏剥离复用 `security.content_filter`**，不自建一套标记。

**记忆只经模型上下文生效**：事实注入进 system 消息后，由主模型自己写进 `item_search` 的入参。
本模块不提供任何「直接改检索结果」的接口——那条腿（`memory/assemble.py` 的硬淘汰与减分）随 M4 删。
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field

from app.security.content_filter import FENCE_TAG

KEY_MAX = 64
VALUE_MAX = 200
# 每轮注入的事实条数上限。constraint 全进，剩下的名额按 updated_at 倒序补。
TIER_ONE_CAP = 8

#: **唯一一个被代码按名字读的 key**：收货国解析的第 3 层（``planner.resolve_dest_country_layered``）
#: 和旧数据迁移都认它。其余 key 都由模型自拟，只经上下文生效、没有哪段代码按名字取。
#: 定在这里而不是各自写字面量——写岔一个字符就是静默退回默认国、到手价按错国家算（计划 §4.1 C1）。
SHIP_TO_KEY = "default_ship_to"


class MemoryCategory(StrEnum):
    """事实的三分类。决定的是**注入优先级**，不是杀伤力。

    - ``constraint``：一直成立的硬规则（「不吃坚果」「只收欧盟境内发货」），每轮全量注入。
    - ``preference``：取向（「喜欢小众品牌」），按新鲜度补位。
    - ``context``：身份类背景（「常寄德国」「家里两只猫」），同样按新鲜度补位。
    """

    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    CONTEXT = "context"


class MemoryFact(BaseModel):
    """一条长期记忆。``key`` 就是身份——同 key 覆盖写，不做合并。"""

    key: str = Field(max_length=KEY_MAX)
    value: str = Field(max_length=VALUE_MAX)
    category: MemoryCategory = MemoryCategory.PREFERENCE
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # 写下它的会话标记，只作偏好页溯源展示，不参与判定。
    source_session: str = ""


class MemoryWriteRejected(ValueError):
    """候选事实被写入过滤器拒绝。**异常消息里绝不回显 value**——被拒的多半正是不该扩散的东西。"""


# 三条默认模式，对应「账号 / 证件 / 联系方式」这一类绝不该进长期记忆的标识符。
DEFAULT_BLOCKED_PATTERNS: tuple[str, ...] = (
    # 连续 9 位以上数字，允许卡号 / 证件号 / 手机号常见的空格、横杠、点、括号分隔。
    # 日期、价格、尺码都短于这个长度，不会被误伤。
    r"(?:\d[ .()\-]{0,2}){8}\d",
    # IBAN 形态的账户标识。
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    # 邮箱。
    r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}",
)

MEMORY_WRITE_REJECTED_TEXT = (
    "没有保存：长期记忆只放偏好和长期成立的规则，不放账号、卡号、证件或联系方式。"
)

_FENCE_MARK_RE = re.compile(r"(?i)<\s*/?\s*" + FENCE_TAG + r"\b[^>]*>")


@lru_cache(maxsize=8)
def _blocked_patterns(extra: str) -> tuple[re.Pattern[str], ...]:
    """默认模式 + ``MEMORY_BLOCKED_PATTERNS`` 里按 ``|||`` 分隔的追加模式，编译一次。

    分隔符不用逗号：正则里 ``{0,2}`` 这类量词自带逗号，用逗号切会把模式本身切断。
    """
    patterns = [*DEFAULT_BLOCKED_PATTERNS, *(p for p in extra.split("|||") if p.strip())]
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            # 坏正则只跳过它自己，不能让整个写入通道失效（默认三条仍然生效）。
            continue
    return tuple(compiled)


def write_filter_rejects(key: str, value: str) -> bool:
    """key 或 value 命中任一模式即拒绝。"""
    patterns = _blocked_patterns(os.environ.get("MEMORY_BLOCKED_PATTERNS", ""))
    return any(p.search(text) for text in (key, value) for p in patterns)


def _clean(text: str, limit: int) -> str:
    """剥围栏标记 → 压掉换行与连续空白 → 截长。

    换行要压掉：一条事实是一行，留着换行等于让被记住的内容能在注入块里伪造出新的结构。
    """
    stripped = _FENCE_MARK_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", stripped).strip()[:limit]


def validate_fact(
    key: str,
    value: str,
    category: str | None = None,
    *,
    source_session: str = "",
    apply_filter: bool = True,
) -> MemoryFact:
    """规范化一条候选事实并过写入过滤器。

    **三条写路径（工具 / 回合后抽取 / 偏好页 API）都必须过它。**

    key 统一小写、空格转下划线：同一主题在不同轮被写成 "Ship To" 和 "ship_to" 时，覆盖写才认得出
    它们是一条。category 给不出合法值时退到 ``preference``——分类错了只是注入优先级低一档，
    比整条事实丢掉划算。

    ``apply_filter=False`` 只给旧数据迁移用（迁移要逐条报告哪些被拒，自己调
    :func:`write_filter_rejects`）。
    """
    fact_key = _clean(key, KEY_MAX).lower().replace(" ", "_")
    fact_value = _clean(value, VALUE_MAX)
    if not fact_key or not fact_value:
        raise MemoryWriteRejected("没有保存：记忆的 key 和内容都不能为空。")
    if apply_filter and write_filter_rejects(fact_key, fact_value):
        raise MemoryWriteRejected(MEMORY_WRITE_REJECTED_TEXT)
    try:
        cat = MemoryCategory(category) if category else MemoryCategory.PREFERENCE
    except ValueError:
        cat = MemoryCategory.PREFERENCE
    return MemoryFact(
        key=fact_key,
        value=fact_value,
        category=cat,
        updated_at=datetime.now(UTC),
        source_session=_clean(source_session, 80),
    )


# ---------------------------------------------------------------------------
# 读侧：注入选择 / 召回匹配 / 判重
# ---------------------------------------------------------------------------


def select_tier_one_facts(
    facts: list[MemoryFact], cap: int = TIER_ONE_CAP
) -> list[MemoryFact]:
    """每轮注入的那一批：**全部 constraint**，再按 ``updated_at`` 倒序补到 ``cap``。

    constraint 不受 cap 限制是有意的：硬规则漏一条就是推荐里出现用户明说过不要的东西，而取向漏一条
    只是这次不够贴合。剩下的事实并非不可见——模型需要时调 ``recall_memories`` 按主题捞。
    """
    oldest = datetime.min.replace(tzinfo=UTC)

    def recency(fact: MemoryFact) -> datetime:
        # 库里可能读回不带时区的时间戳（SQLite 存过的老行），按 UTC 比较，别在这里抛。
        updated = fact.updated_at or oldest
        return updated if updated.tzinfo else updated.replace(tzinfo=UTC)

    constraints = [f for f in facts if f.category is MemoryCategory.CONSTRAINT]
    others = sorted(
        (f for f in facts if f.category is not MemoryCategory.CONSTRAINT),
        key=recency,
        reverse=True,
    )
    return constraints + others[: max(0, cap - len(constraints))]


def match_facts(facts: list[MemoryFact], query: str) -> list[MemoryFact]:
    """``recall_memories`` 的匹配口径：查询词出现在 key / value / category 任一处。

    **按字符子串匹配，不按空格切词**——参考实现的 ``query.split()`` 对中文等于没切，「塑料 材质」
    这样的查询会整串去找，一条都命中不了。空格仍然当分隔符用（英文查询照常切开），
    切完的每个词再走子串。
    """
    terms = [t for t in query.lower().split() if t] if query else []
    if not terms:
        return list(facts)
    return [
        f
        for f in facts
        if any(t in f"{f.key} {f.value} {f.category.value}".lower() for t in terms)
    ]


def _bigrams(text: str) -> set[str]:
    """字符 bigram。中文按字切，英文顺带也能比。"""
    cleaned = re.sub(r"\s+", "", text.lower())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def same_fact(a: str, b: str, threshold: float = 0.6) -> bool:
    """两条事实的**值**是否说的是同一件事（Jaccard ≥ 阈值）。

    参考实现按空格分词算 Jaccard，对中文失效（整句一个 token，两条不同的话相似度恒为 0，
    结果是同一件事被反复写成新 key）。这里改字符 bigram。
    """
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return sa == sb
    return len(sa & sb) / len(sa | sb) >= threshold


def render_memory_block(facts: list[MemoryFact]) -> str:
    """把事实渲染成注入用的 XML 块。空列表返回空串（调用方据此决定不注入这条 system 消息）。

    每条一行 ``[category] key: value``。分类摆在最前，是为了让模型一眼看出哪几条是硬规则。
    """
    if not facts:
        return ""
    lines = [f"[{f.category.value}] {f.key}: {f.value}" for f in facts]
    body = "\n".join(lines)
    return f"<user_long_term_memory>\n{body}\n</user_long_term_memory>"
