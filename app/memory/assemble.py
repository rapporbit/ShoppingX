"""本轮**会话级**约束的装配：item_search / item_picker 读到的唯一一份词表。

**长期记忆不在这里**（M4 删）。它现在只经模型上下文生效：每轮注入的 `<user_long_term_memory>`
里带 `[constraint]` 标记，由**主模型自己**把它写进 `item_search` 的入参（exclude / 关键词 /
价格区间），走现有的 payload filter。本模块原来那条「长期库 → 硬淘汰 / 减分」的腿整条摘掉。

**为什么摘**：一条长期偏好要生效，原来有两条并行通路——注入给模型看的文本，和机制侧按域过滤的词表。
两份来源不同、判据不同（域闸只管机制侧），于是「模型看到的」和「机制执行的」长期处在不一致里，
还得靠 fail-closed 的域闸兜着：planner 判不出域的轮次，长期硬规则整轮静默消失。收敛成一条通路后，
记忆生效与否只取决于模型有没有把它写进入参——写没写进去，在工具入参里**看得见**。
代价也写在这里：普通轮的 `item_picker` 由 autopick 自动跑、没有模型入参，长期 constraint 不再
在精挑层生效，只在检索层和收尾由模型把关（计划 §3.2 第 1 条，已知并接受）。

剩下两路都不是长期记忆：

- **会话级 P_t**：用户**本轮**亲口说的「不要 X」/「要 Y」。它本来就只活在这次会话里，
  由模型之外的机制维护（`memory/session_state.py`），不存在「跨轮跨品类误杀」的问题。
- **行为亲和**：从**收藏**聚合出的弱正向证据（零 LLM，见 :mod:`app.memory.affinity`），
  只进 item_picker 的弱加分。因为是推断，档位压到最低：不淘汰、不进检索词、冲突时让位于显式表达。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.context import get_session_pt
from app.memory.affinity import affinity_terms
from app.memory.store import PreferenceStore
from app.utils.terms import normalize_terms, term_hits


class MemoryBundle(BaseModel):
    """本轮会话级约束的装配结果。下游工具**只读这个**。"""

    exclude: list[str] = Field(default_factory=list)  # → item_picker 命中即淘汰（P_t 的「不要 X」）
    penalty: list[str] = Field(default_factory=list)  # → item_picker 命中减分（P_t 的弱表达）
    must: list[str] = Field(default_factory=list)  # → item_picker 强加分（不淘汰）
    # 行为亲和（收藏聚合，见 app.memory.affinity）→ item_picker 弱加分。**不进检索词**：
    # 检索词决定「捞哪一池」，把一个用户从没提过的词（canvas）塞进 query 会把整池带偏，代价是
    # 全局的；而它作为打分项只在池内微调排序，代价是局部的。行为是弱证据，只配后者。
    affinity: list[str] = Field(default_factory=list)
    budget_usd: float | None = None


def _merge(*groups: list[str]) -> list[str]:
    """小写、保序去重地并列表——词表是拿去和商品标题做子串匹配的。"""
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for t in g:
            low = t.strip().lower()
            if low and low not in seen:
                seen.add(low)
                out.append(low)
    return out


async def assemble(user_id: str, store: PreferenceStore | None = None) -> MemoryBundle:
    """装配本轮约束：会话级 P_t（本轮亲口说的）+ 行为亲和（收藏聚合）。"""
    pt = get_session_pt()

    exclude = _merge(pt.dislike_terms() if pt else [])
    penalty = _merge(pt.soft_dislike_terms() if pt else [])
    must = _merge(pt.like_terms() if pt else [])

    # **说过的话压过做过的事**：用户本轮排斥过的词，即便在收藏里高频出现也一律不进亲和——否则会
    # 出现「他嘴上说不要皮革，可他收藏过三双皮鞋」这种一边减分一边加分的自相矛盾，净效果取决于
    # 两个权重谁大，不可解释。显式是强证据、行为是弱证据，冲突时强的赢。
    #
    # **压制判定必须归一 + 走 term_hits，不能拿原始词做 `in set()` 的精确相等**——初版就是这么
    # 写的，在最常见的真实形态下静默失效，两处叠加：① 中文对话抽出的原子词（'皮革'）与亲和词是
    # 英文标题 token（'leather'），不归一两边永远对不上（item_picker 对 penalty 也是**归一后**
    # 才拿去匹标题的，压制这一路却漏了同一道工序）；② 归一后 blocked 里是 'leather'，而亲和
    # token 可能是更长的变体 'genuine leather'，精确相等照样穿过去。得用「这条排斥词命中了这个
    # 亲和词吗」来判，即 term_hits。失效的后果是把这条纪律整个架空。
    blocked = normalize_terms(_merge(exclude, penalty))
    affinity = [
        t for t in await affinity_terms(user_id, store) if not any(term_hits(b, t) for b in blocked)
    ]

    return MemoryBundle(
        exclude=exclude,
        penalty=penalty,
        must=must,
        affinity=_merge(affinity),
        budget_usd=pt.budget_usd if pt else None,
    )
