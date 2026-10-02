"""本轮生效约束 P_t —— planner 每轮从「前几轮原话 + 本轮原话」整体重算出的结构化约束。

形状沿用 RecBot 论文原型（arXiv:2509.21317）：品类 / 预算 / 三个词表（硬淘汰 / 软减分 /
加分）。与另两层的分界（务必别混）:
- **长期库**（:mod:`app.memory.store`）：跨会话的一贯取向，按用户聚合、语义去重、带半衰期。
- **本模块 P_t**：**本轮**仍生效的约束（「这次预算 300」「不要塑料」）。唯一写者是 planner，
  item_picker / assemble / drift / signals 只读。
- **行为历史**（``HistoryEntry``）：上次搜 / 买了什么的事实快照，与偏好正交。

**无状态（2026-09-25 重构）**：P_t 曾跨轮累积、随 session.json 落盘，撤回要靠「模型抄词 +
逐词核验原话」、换品类要靠 ``topic_switch`` 清表——一整套合并机制只为让 planner 不必重读上文。
现在反过来：跨轮只存用户原话（``AgentState.middle_context["prior_queries"]``，见 orchestrator），
planner 每轮读最近几轮原话、**整体重算**本轮仍生效的约束。撤回与换品类随重算自然生效，
不再有合并逻辑。代价是约束的存续要过 LLM 的手：早几轮说的约束能否被重新抽出来，需多轮用例对照。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field


def _atoms(words: Iterable[str]) -> list[str]:
    """小写、去空、保序去重——三个词表的统一形态（拿去和商品标题做字符串匹配）。"""
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = (w or "").strip().lower()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return out


# 英文品类名里这些修饰词会把精排的路径分 / 标题分压塌（2026-10-02 rank-eval badcase：
# 「running shoes」路跑路径 0.929，「adidas men's running shoes black」0.151）。
# 品牌靠 planner 字段说明约束；性别 / 颜色是封闭小词表，机械剔掉。
_EN_MODIFIERS = re.compile(
    r"\b(?:men|women|man|woman|boy|girl|kid|unisex|male|female|ladies|lady|"
    r"black|white|red|blue|green|pink|grey|gray|brown|beige|navy|purple|yellow|"
    r"silver|gold)(?:'?s|')?\b",
    re.IGNORECASE,
)


def clean_category_en(text: str) -> str:
    """英文品类名去性别 / 颜色修饰、压空白、小写；剔完为空就返回空（调用方退回中文品类）。"""
    return " ".join(_EN_MODIFIERS.sub(" ", text or "").split()).lower()


class TurnConstraints(BaseModel):
    """本轮生效约束 P_t（planner 写，run 内有效）。

    - ``exclude_terms``：硬淘汰（用户说「不要 X」）→ item_picker 命中即淘汰。
    - ``avoid_terms``：软减分（「尽量别」「不太喜欢」）→ 命中减分、不淘汰。硬淘汰匹不准的词
      就是拿误杀去赌——「花哨」这种词一旦匹上（「塑料感」连坐 plastic），杀掉的可能正是用户要的。
    - ``prefer_terms``：加分。正向**不做二值淘汰**（数据没有可靠的材质 / 风格字段，keep-only 会
      误杀一大片），所以即便「必须金属」也只作强加分。
    - ``category``：本轮主品类（中文）；``category_en``：同一品类的英文名，精排路径分用它；
      ``keywords``：planner 检索词，精排标题分用它（标题要判具体属性，路径只判品类）。

    三个词表在构造时统一成 :func:`_atoms` 形态。消费接口沿用旧名（``dislike_terms`` /
    ``soft_dislike_terms`` / ``like_terms``），下游 assemble / signals / drift / item_picker
    零改动。
    """

    category: str = Field(default="", description="本轮主品类")
    category_en: str = Field(default="", description="本轮主品类英文名（已去性别 / 颜色修饰）")
    keywords: list[str] = Field(
        default_factory=list, description="planner 检索词（精排给商品标题打分用，路径分不用）"
    )
    budget_usd: float | None = Field(default=None, description="本轮生效的预算上限 USD，无则 None")
    exclude_terms: list[str] = Field(default_factory=list, description="硬淘汰词")
    avoid_terms: list[str] = Field(default_factory=list, description="软减分词")
    prefer_terms: list[str] = Field(default_factory=list, description="加分词")

    @classmethod
    def build(
        cls,
        *,
        category: str = "",
        category_en: str = "",
        keywords: Iterable[str] = (),
        budget_usd: float | None = None,
        exclude: Iterable[str] = (),
        avoid: Iterable[str] = (),
        prefer: Iterable[str] = (),
    ) -> TurnConstraints:
        """从 planner 的产出构造。一个词同时进了多个表时按「硬淘汰 > 软减分 > 加分」只留一处。"""
        ex = _atoms(exclude)
        av = [w for w in _atoms(avoid) if w not in ex]
        pf = [w for w in _atoms(prefer) if w not in ex and w not in av]
        return cls(
            category=category,
            category_en=clean_category_en(category_en),
            keywords=[k.strip() for k in keywords if k and k.strip()],
            budget_usd=budget_usd,
            exclude_terms=ex,
            avoid_terms=av,
            prefer_terms=pf,
        )

    def is_empty(self) -> bool:
        return (
            not self.category
            and not self.keywords
            and self.budget_usd is None
            and not self.exclude_terms
            and not self.avoid_terms
            and not self.prefer_terms
        )

    def dislike_terms(self) -> list[str]:
        """硬 dislike 词 → item_picker 命中即淘汰。"""
        return list(self.exclude_terms)

    def soft_dislike_terms(self) -> list[str]:
        """软 dislike 词 → item_picker 命中减分、不淘汰。"""
        return list(self.avoid_terms)

    def like_terms(self) -> list[str]:
        """like 词 → item_picker 加分。"""
        return list(self.prefer_terms)
