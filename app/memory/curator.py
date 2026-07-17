"""记忆管家 curator —— 会话结束后判定「一贯取向」并写入**长期库**的唯一入口。

**单写者纪律（P_t 重构）：curator 不碰 P_t。** 会话级短期状态 P_t 的唯一写者是 planner
（每轮跑、看用户原话、当轮落盘当轮生效，见 ``app.tools.planner._sync_session_pt``）。curator
曾拥有 P_t 的「收口权」（current_intent / 预算撤销 / slots / open_questions / 约束撤回），实践
证明双写者结构就是 bug 温床：两个写者的时序要靠 main_agent 收尾前重新快照来协调；curator 的
输出质量（fast 档、收尾后跑、看的是转述）系统性低于 planner（当轮、看原话、带确定性回填闸），
历史事故全是它覆盖 planner 的正确结果（无 blocking 的 draft 覆盖软硬档位、脑内换汇覆盖折算好
的预算）。收口权因此整体挪给 planner，curator 退回它唯一做得好的事：

- **persistent_preferences** → 经 injector 的 ``persist_new_preferences`` 这**单一落库口**
  提升为长期偏好（带 slug/domain + keys_to_supersede 做去重 / 冲突消解）。

**时机（后处理异步）:** 在 ``run_agent`` 收尾、``report_task_result`` **之后**调用——用户已拿到
回复，curator 的 LLM 调用对用户零感知延迟。``prev_pt`` 只是**只读参考**（让 LLM 看见会话层已经
记了什么，避免把本轮约束错升成长期），不再据此写任何东西。

**容错（降级不崩）:** curator 在主回复已下发后跑，任何异常都**只记日志、返回 None**，绝不反噬
主链路（不让「记忆没记上」演变成「这次任务失败」）。
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.agent.llm import get_fast_llm
from app.agent.prompts import get_memory_curator_prompt
from app.api.context import get_session_domains
from app.memory.domains import DOMAIN_OTHER, PrefDomain, domain_menu
from app.memory.injector import persist_new_preferences
from app.memory.session_state import SessionPrefState
from app.memory.store import PrefCategory, PreferenceEntry, get_store
from app.tools._args import drop_none_values

logger = logging.getLogger("shoppingx.curator")

Polarity = Literal["like", "dislike"]


class _PersistentPref(BaseModel):
    """curator 判定为「跨会话一贯取向」、应提升为长期偏好的一条。

    **没有 strength / blocking**：curator 学到的偏好**一律只减分**，拿不到硬淘汰权——那要用户
    去偏好页面亲手勾「绝不推荐」。这道闸设在 ``persist_new_preferences`` 这个唯一落库口上
    （source="agent" 一律 blocking=False），curator 就算填了也没用。

    ``domain`` 是封闭枚举 :data:`app.memory.domains.PrefDomain`，决定这条偏好在哪些轮次生效。
    """

    content: str = Field(description="偏好内容，如「不接受皮革材质」")
    category: PrefCategory = Field(default="other")
    domain: PrefDomain = Field(
        default=DOMAIN_OTHER,
        description=(
            "这条偏好管哪个品类域。**买鞋时说的「不喜欢皮革」填 footwear，不要填 global**——"
            "global 只留给真正跨品类的底线（「我素食，任何动物皮革都不要」这种伦理/安全/过敏）。"
            "判不出填 other。可选值：\n" + domain_menu()
        ),
    )
    slug: str = Field(description="原子标识（规范化英文，如 leather/brand_nike）")
    polarity: Polarity = Field(default="like")
    keywords: list[str] = Field(
        default_factory=list,
        description="可直接匹配商品标题的原子词。dislike **必须给**——它是下游匹配的唯一抓手",
    )
    keys_to_supersede: list[str] = Field(
        default_factory=list, description="引用要顶替的旧长期偏好的 dedup_key（从已有偏好清单里选）"
    )


class CurationResult(BaseModel):
    """curator 的结构化输出：只判长期偏好。

    曾经还有 current_intent / clear_budget / category / slots_patch / supersede_session_keys /
    new_open_questions / resolved_questions 一整排会话态字段——全删。会话态的产生**与撤销**都归
    planner（唯一写者）：撤回走「LLM 抄 id 提议 + 代码词面核验」（见 session_state.merge_pt），
    预算撤销走 ``PlanOutput.clear_budget``。给 curator 留这些字段，就是给双写者留还魂的门。
    """

    # 显式 null 归一为缺席（badcase cdee1d6d 同族，见 drop_none_values）。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    persistent_preferences: list[_PersistentPref] = Field(default_factory=list)


def _format_long_term(entries: list[PreferenceEntry]) -> str:
    """把已有长期偏好渲染给 curator 看，供它引用 dedup_key 判 keys_to_supersede（冲突消解）。"""
    if not entries:
        return "（该用户暂无长期偏好）"
    return "\n".join(f"- [{e.dedup_key}] {e.content}（{e.polarity}）" for e in entries)


def _scope_to_session_domain(prefs: list[_PersistentPref], domains: list[str]) -> None:
    """curator 漏填 domain（落到 ``other``）时，用 planner 本轮判定的域兜底（就地改 prefs）。

    **为什么需要这道兜底**：``other`` 是保守档——它谁也匹配不上，等于这条偏好几乎不会生效。而
    LLM 漏判是常态（域有 20 个选项）。本轮 planner 已经判过「在买什么」了（``footwear``），那就
    是这条偏好最可能归属的域，比 ``other`` 强得多。

    **为什么不兜到 global**：那是激进档（全局生效、跨品类杀商品）。漏填的默认值绝不能是它——
    这正是改造前的坑：``domain`` 留空即全局，于是「买鞋时说的不喜欢皮革」在买沙发时也生效。
    失效方向必须是「这条偏好范围偏窄」（用户再说一遍即可），而不是「它在所有品类里静默杀商品」。

    本轮域判不出（闲聊轮 / planner 没跑）时保持 ``other`` 不动——没有依据就别猜。
    """
    if not domains:
        return
    fallback = domains[0]  # 多域时取第一个（planner 按主次给），够用且不必猜
    for p in prefs:
        if p.domain == DOMAIN_OTHER:
            p.domain = fallback  # type: ignore[assignment]
            logger.info("偏好 slug=%s 未判出 domain，兜底到本轮域 %s", p.slug, fallback)


async def curate_turn(
    user_id: str,
    query: str,
    final_text: str,
    prev_pt: SessionPrefState,
    session_domains: list[str] | None = None,
) -> CurationResult | None:
    """会话结束后跑一次：读本轮对话，判「一贯取向」→ 提升长期偏好。**不写 P_t**。

    参数:
      - user_id:登录用户。**匿名（``""``）直接跳过**——长期库不落匿名偏好，而 P_t 已不归
        curator 管，匿名轮次没有任何它能做的事，白烧一次 LLM 调用。
      - query / final_text:本轮用户原话 + 助手最终回复。
      - prev_pt:本轮 P_t 的**只读参考**（渲染给 LLM 看「会话层已经记了什么」，帮它把本轮
        约束挡在长期库外）——不据此写任何东西，传开局那份或 planner 更新后那份皆可。
      - session_domains:planner 本轮判定的品类域，**由调用方在清理会话状态之前快照好传进来**。
        不传则回退到 ContextVar——那只在测试 / 离线脚本这类没走 run_agent 收尾清理的场景下
        才是对的（``run_agent`` 里 finally 早已把它清空，直接读会永远拿到空值，域兜底会静默失效）。

    返回 :class:`CurationResult`（供日志 / 测试断言）;匿名或 LLM 调用失败时返回 ``None`` 并已
    降级（长期库不写）——**绝不抛**，因为本函数在主回复下发之后跑，不该反噬主链路。
    """
    if not user_id:
        return None  # 匿名：长期库不写、P_t 不归它管——没有可做的事，连 LLM 都不必调

    # 已有长期偏好（供 curator 判冲突消解）。
    long_term: list[PreferenceEntry] = await get_store().read(user_id)

    user_msg = (
        f"【本轮用户原话】\n{query}\n\n"
        f"【助手最终回复（摘要判断本轮做了什么）】\n{final_text[:1500]}\n\n"
        f"【当前会话状态 P_t（只读参考：会话层已记住的部分，不必也不能由你再记）】\n"
        f"{prev_pt.render()}\n\n"
        f"【该用户已有长期偏好（判 keys_to_supersede 用）】\n{_format_long_term(long_term)}"
    )
    try:
        # method 钉死的理由见 planner.py（默认值随模型能力画像浮动，qwen 系会 400）。
        structured = get_fast_llm().with_structured_output(
            CurationResult, method="function_calling"
        )
        result = await structured.ainvoke(
            [("system", get_memory_curator_prompt()), ("user", user_msg)],
        )
        curation = (
            result if isinstance(result, CurationResult) else CurationResult.model_validate(result)
        )
    except Exception:
        logger.warning("curator LLM 调用失败，本轮记忆判定降级跳过（user=%s）", user_id)
        return None

    # 提升长期偏好（经单一落库口）。keys_to_supersede 逐条聚合后传入。
    domains = session_domains if session_domains is not None else get_session_domains()
    _scope_to_session_domain(curation.persistent_preferences, domains)
    if curation.persistent_preferences:
        supersede: list[str] = []
        for p in curation.persistent_preferences:
            supersede.extend(p.keys_to_supersede)
        await persist_new_preferences(
            user_id,
            curation.persistent_preferences,
            keys_to_supersede=supersede or None,
        )

    logger.info("curator user=%s persistent=%d", user_id, len(curation.persistent_preferences))
    return curation
