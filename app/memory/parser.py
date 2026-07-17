"""把用户手填的一句话拆成结构化偏好条目（偏好页面「添加」入口的后端）。

**为什么要 LLM 这一步：** 长期库里的一条偏好不只是一句文本——它需要 ``slug``（去重身份）、
``keywords``（dislike 硬过滤的落地黑名单）、``category``/``polarity``/``strength``。这些字段
不可能让用户填（谁会知道 slug 是什么），但让用户只写一句自然语言、后端补齐结构，两边都舒服。

**为什么不做「确认卡」：** 解析结果会立刻出现在偏好列表里，不对就点一下删掉——写入是廉价可逆的，
所以不值得为它加一道审批（同 ChatGPT 的 saved memories）。这里唯一的硬要求是 **不引申**：
用户明说什么就存什么，prompt 里已写死；curator 那种「从对话里推断一贯取向」的判断力，
在这条路径上是负资产。

失败（LLM 报错 / 解析不出）一律返回空列表，由 API 层转成 400——不静默写脏数据。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, model_validator

from app.agent.llm import get_fast_llm
from app.agent.prompts import get_preference_parse_prompt
from app.memory.domains import DOMAIN_OTHER, PrefDomain, domain_menu
from app.memory.store import Polarity, PrefCategory
from app.tools._args import drop_none_values

logger = logging.getLogger("shoppingx.memory")

# 用户一句话最多拆出几条——防一段长文本炸出十几条噪声偏好。超出部分丢弃（截断即可，
# 用户看到列表里只进了前 N 条，可以再补一句）。
MAX_PREFS_PER_TEXT = 5


class UserPrefDraft(BaseModel):
    """用户手填偏好经 LLM 结构化后的一条。

    字段满足 :class:`app.memory.injector.NewPreferenceLike`，故可直接进同一个落库口。

    与 curator 的 ``_PersistentPref`` 字段一致，但**没有 keys_to_supersede**——手填不做矛盾消解，
    用户要改哪条自己删哪条，别让 LLM 替他删东西。

    **这里是 ``blocking`` 唯一的合法来源。** 长期库里的硬淘汰权只由用户授予：他在偏好页面写下
    「绝对不要皮革」并勾选「绝不推荐」，那条才有权把商品从结果里删掉。curator 从对话里学到的
    一律只减分（``persist_new_preferences`` 里那道闸挡着，source="agent" 拿不到 blocking）。
    """

    content: str = Field(description="偏好内容，如「不接受皮革材质」")
    category: PrefCategory = Field(default="other")
    domain: PrefDomain = Field(
        default=DOMAIN_OTHER,
        description=(
            "这条偏好管哪个品类域。用户说「买鞋不要皮革」→ footwear；说「我素食，任何皮革制品都"
            "不要」→ global（跨品类底线）。判不出填 other。可选值：\n" + domain_menu()
        ),
    )
    slug: str = Field(description="原子标识（规范化英文，如 leather/brand_nike）")
    polarity: Polarity = Field(default="like")
    blocking: bool = Field(
        default=False,
        description=(
            "用户表达的是**绝对排除**（「绝对不要」「永远别给我推」「过敏，一定不能有」）→ true，"
            "命中即从结果里删掉。只是「不太喜欢 / 尽量避免 / 不太想要」→ false（减分不淘汰）。"
            "**拿不准一律给 false**：误判为 true 会让用户再也搜不到某类商品且归因不了，"
            "误判为 false 只是排序偏一点。"
        ),
    )
    keywords: list[str] = Field(default_factory=list)


class _ParseResult(BaseModel):
    """LLM 的结构化输出：一句话可能含多条偏好，也可能一条都不含（空列表）。"""

    # 显式 null 归一为缺席（badcase cdee1d6d 同族，见 drop_none_values）。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    preferences: list[UserPrefDraft] = Field(default_factory=list)


async def parse_user_preference(text: str) -> list[UserPrefDraft]:
    """把一句自然语言拆成若干结构化偏好草稿；解析不出或出错时返回空列表（不抛）。"""
    if not text.strip():
        return []
    try:
        # method 钉死的理由见 planner.py（默认值随模型能力画像浮动，qwen 系会 400）。
        structured = get_fast_llm().with_structured_output(_ParseResult, method="function_calling")
        result = await structured.ainvoke(
            [("system", get_preference_parse_prompt()), ("user", text.strip())]
        )
    except Exception as exc:  # noqa: BLE001 —— 解析失败降级为空，由 API 层告诉用户「没听懂」
        logger.warning("偏好解析失败（text=%r）：%s", text[:60], exc)
        return []
    if not isinstance(result, _ParseResult):
        return []
    # slug 是去重身份，空 slug 的条目会让 dedup_key 退化成同一个键、互相覆盖——直接丢弃。
    drafts = [p for p in result.preferences if p.slug.strip() and p.content.strip()]
    return drafts[:MAX_PREFS_PER_TEXT]
