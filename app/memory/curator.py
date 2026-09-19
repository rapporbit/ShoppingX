"""回合后抽取 —— 会话跑完之后读这一轮对话，把「下次还成立」的事实写进长期库。

**只写 `memory_facts`（`MemoryFactStore`），不碰 P_t，也不再碰旧 `PreferenceStore`。** 会话级短期
状态的唯一写者仍是 planner（每轮跑、看用户原话、当轮生效，见 `app.tools.planner._sync_session_pt`）；
curator 退回它唯一做得好的事：判「一贯取向」。M3 把它的输出从 polarity / slug / domain /
keys_to_supersede 那套换成 `key / value / category` 三字段的事实——`key` 就是身份，同 key 覆盖写，
冲突消解不再需要模型引用一串 dedup_key（它经常拼错，拼错就删不掉旧的）。

**只读 user / assistant 文本，不读工具结果。** 喂给抽取模型的只有用户原话与最终回复：商品标题、
网页正文这些外部内容里写着什么「记住我是管理员」都跟用户无关，让它们进抽取输入等于给长期库开了
一道注入口。P_t 也不再喂（参考实现只给「已存事实 + 本轮对话」）——它是会话层状态，靠 prompt 里
「本轮约束不升长期」那条规则把关就够，少一份上下文少一种把短期状态抄成长期的诱因。

**清空并发护栏：** 抽取前后各读一次 `purge_generation`，代数变了说明用户在模型跑的这几秒里清空过
记忆，整批丢弃——否则刚点完「清空」，几条旧偏好又自己长回来了。

**时机（后处理异步）：** 在 `run_agent` 收尾、`report_task_result` **之后**调用——用户已拿到回复，
这次 LLM 调用对用户零感知延迟。

**容错（降级不崩）：** 任何异常都只记日志、返回空列表，绝不反噬主链路（不让「记忆没记上」演变成
「这次任务失败」）。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, model_validator

from app.agent.invoke import call_structured
from app.agent.llm import get_fast_llm
from app.agent.prompts import get_memory_curator_prompt
from app.api.context import get_thread_id, record_learned_pref
from app.memory.fact_store import get_fact_store
from app.memory.facts import (
    KEY_MAX,
    VALUE_MAX,
    MemoryFact,
    MemoryWriteRejected,
    memory_enabled,
    render_memory_block,
    same_fact,
    validate_fact,
)
from app.tools._args import drop_none_values

logger = logging.getLogger("shoppingx.curator")

#: 一轮最多学 3 条。上限不是性能考虑，是质量闸：一轮对话真能教会的事实就那么几件，允许它写十条，
#: 它就会把本轮的颜色、价位、心情都凑成「长期偏好」。
MAX_NEW_FACTS = 3

#: 喂给抽取模型的对话文本上限（用户原话 + 最终回复各自截断）。
TRANSCRIPT_LIMIT = 2000


class _RecordedFact(BaseModel):
    """抽取模型提议记住的一条事实。字段与 `save_memory` 工具、偏好页 API 完全一致——三条写路径
    同一套形状，才能都过 `validate_fact` 这一道门。"""

    key: str = Field(
        max_length=KEY_MAX,
        description="这条事实的主题标识（英文小写下划线，如 material_avoid / default_ship_to）。"
        "更新已有主题时**原样复用**清单里那个 key，新值会覆盖旧值",
    )
    value: str = Field(max_length=VALUE_MAX, description="一句话写清内容，只写用户说过的")
    category: str = Field(
        default="preference",
        description="constraint 一直成立的硬规则 / preference 取向 / context 身份背景",
    )


class CurationResult(BaseModel):
    """抽取模型的结构化输出：只有一个列表。没学到就给空列表。"""

    # 显式 null 归一为缺席（badcase cdee1d6d 同族，见 drop_none_values）：模型把「什么都没学到」
    # 写成 ``{"facts": null}`` 时，default_factory 不生效，校验直接炸——整轮抽取白跑。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    facts: list[_RecordedFact] = Field(default_factory=list)


def _select_new_facts(
    proposals: list[_RecordedFact], existing: list[MemoryFact], *, source_session: str
) -> list[MemoryFact]:
    """把模型的提议筛成真正要落库的几条：过写入过滤器、去重、封顶 `MAX_NEW_FACTS`。

    三条判据，顺序有讲究：

    1. **同 key** → 这是 prompt 要的「更新」。值说的是同一件事就跳过（不刷时间戳、不回执「学到」），
       否则保留新值，并把旧值移出已知集合——旧值已被顶替，不该再挡住别的提议。
    2. **新 key、但值与某条已存事实说的是同一件事** → 跳过。这正是 `same_fact` 存在的理由：
       同一件事被反复换个 key 写进来，注入块很快就被同义句刷满。
    3. 剩下的才是新事实。

    判重用**字符 bigram Jaccard**（`facts.same_fact`），不用参考实现的空格分词——中文整句只切出
    一个 token，两条不同措辞的相似度恒为 0，那道闸等于不存在。
    """
    held = {f.key: f.value for f in existing}
    known = set(held.values())
    picked: list[MemoryFact] = []
    for proposal in proposals:
        if len(picked) >= MAX_NEW_FACTS:
            break
        try:
            fact = validate_fact(
                proposal.key, proposal.value, proposal.category, source_session=source_session
            )
        except MemoryWriteRejected:
            # 空值或 PII。被拒的内容不进日志——被拒的多半正是不该扩散的东西。
            logger.info("抽取提议被写入过滤器拒绝，跳过（key=%s）", proposal.key[:KEY_MAX])
            continue
        current = held.get(fact.key)
        if current is not None:
            if same_fact(fact.value, current):
                continue
            known.discard(current)
        elif any(same_fact(fact.value, seen) for seen in known):
            continue
        held[fact.key] = fact.value
        known.add(fact.value)
        picked.append(fact)
    return picked


async def curate_turn(user_id: str, query: str, final_text: str) -> list[MemoryFact]:
    """会话结束后跑一次：读本轮对话，把值得跨会话记住的事实写进长期库。返回**真正写进去的**几条。

    参数:
      - user_id:登录用户。**匿名（``""``）直接跳过**——长期库不落匿名事实，白烧一次 LLM 调用。
      - query / final_text:本轮用户原话 + 助手最终回复。**输入只有这两段**，工具结果不进来。

    返回写入的事实列表（供日志 / 测试断言）;匿名、开关关闭、LLM 失败、被清空代数否决时返回空列表
    并已降级——**绝不抛**，因为本函数在主回复下发之后跑，不该反噬主链路。
    """
    if not user_id:
        return []
    if not memory_enabled():
        return []

    store = get_fact_store()
    # 代数要在读已存事实之前取：两次读之间用户清空的话，existing 会是清空后的空表，
    # 而收尾的代数比对仍能发现变化并丢弃整批。反过来取则有一道缝。
    generation = await store.purge_generation(user_id)
    existing = await store.get_facts(user_id)

    user_msg = (
        f"【已经记住的事实】\n{render_memory_block(existing) or '（暂无）'}\n\n"
        f"【本轮用户原话】\n{query[:TRANSCRIPT_LIMIT]}\n\n"
        f"【助手最终回复】\n{final_text[:TRANSCRIPT_LIMIT]}"
    )
    try:
        curation = await call_structured(
            get_fast_llm(),
            [("system", get_memory_curator_prompt()), ("user", user_msg)],
            CurationResult,
        )
    except Exception:
        logger.warning("curator LLM 调用失败，本轮记忆抽取降级跳过（user=%s）", user_id)
        return []

    new_facts = _select_new_facts(curation.facts, existing, source_session=get_thread_id() or "")
    if not new_facts:
        return []
    if await store.purge_generation(user_id) != generation:
        # 模型跑的这几秒里用户清空了记忆。整批丢弃——不给「刚清完又长回来」留任何缝。
        logger.info("抽取期间用户清空了记忆，%d 条提议整批丢弃（user=%s）", len(new_facts), user_id)
        return []
    if not await store.upsert_facts(user_id, new_facts):
        return []

    for fact in new_facts:
        # 本轮累加器 → run_agent 汇总进 learned_preferences，并经 AGUI 推给前端「记住了 …」那行。
        # 前端那行的 ✕ 目前打的还是旧偏好表的删除接口，M4 改偏好页时一起对齐到 key。
        record_learned_pref(fact.value, fact.key)
    logger.info("curator user=%s 写入 %d 条事实", user_id, len(new_facts))
    return new_facts
