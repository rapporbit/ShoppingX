"""召回层检索文本塑形：category 截尾 + 描述边界截断。

建索引时拼 dense 编码文本用（``item_search`` 精排也复用 ``tail_category``）。
词法/稀疏分词已随选型修订移除——召回改 dense + filter（见 `docs/plans/召回引擎选型思路.md` §4）。
"""

from __future__ import annotations

import re

# category 取尾 N 层：泛词集中在顶 1-2 层，尾 3 层保留「叶+父+祖」上下文又不引泛词噪声。
CATEGORY_TAIL = 3
# 描述截断：句末标点用于「防半句」。
_SENT_END_RE = re.compile(r"[.!?]")


def tail_category(category: str, n: int = CATEGORY_TAIL) -> str:
    """归一面包屑 ``A > B > C`` 取末 n 层重拼；不足 n 层原样；空返空。"""
    if not category:
        return ""
    parts = [p.strip() for p in category.split(">") if p.strip()]
    return " > ".join(parts[-n:])


def clip_sentence(text: str, limit: int = 300, min_keep: int = 200) -> str:
    """截到 ≤limit 字符，防半句：优先末句标点(≥min_keep)，否则词边界(≥min_keep)，否则硬截。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in _SENT_END_RE.finditer(head) if m.end() >= min_keep]
    if ends:
        return head[: ends[-1]]
    space = head.rfind(" ")
    return head[:space] if space >= min_keep else head
