"""用户级数据接口：长期记忆（偏好面板）/ 会话级约束 / 收藏 / 找相似。从 ``server.py`` 拆出。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import BaseModel

from app.agent.orchestrator import load_session_state, save_session_state
from app.api import (
    monitor,
)
from app.api.auth import (
    auth_enabled,
    get_current_user_id,
)
from app.api.guards import guard_thread, safe_session_dir
from app.memory.fact_store import get_fact_store
from app.memory.facts import MemoryFact, MemoryWriteRejected, validate_fact
from app.memory.session_state import (
    SessionPrefState,
    constraint_rows,
    drop_constraint,
    pt_from_state,
    pt_into_state,
)
from app.memory.store import FavoriteItem, get_store
from app.recall import get_recall_client
from app.utils.path_utils import (
    OUTPUT_ROOT,
)

router = APIRouter()


def _fact_json(fact: MemoryFact) -> dict[str, Any]:
    """一条事实的 JSON。字段就是模型看到的那四个，不多不少。

    页面上给用户看的，必须**和注入给模型的是同一份东西**——上一版偏好页回吐 polarity /
    blocking / domain / keywords 七八个字段，用户改了其中一个却看不出行为会怎么变，而模型
    根本没见过这些字段。现在两边都只有 ``key / value / category``：用户看到什么，模型就读到什么。

    ``updated_at`` 给前端显示「这条多久没更新了」——它只参与 tier-one 的补位排序（见
    ``facts.select_tier_one_facts``）与保留期，不参与任何打分。
    """
    return {
        "key": fact.key,
        "value": fact.value,
        "category": fact.category.value,
        "updated_at": fact.updated_at.isoformat(),
        "source_session": fact.source_session,
    }


def _assert_own(user_id: str, auth_uid: str | None) -> None:
    """开启鉴权后只能读写**自己**的记忆（同 GET 的口径，写口尤其不能漏）。"""
    if auth_enabled() and auth_uid != user_id:
        raise HTTPException(403, "无权访问他人偏好")


class FactWrite(BaseModel):
    """偏好页手填 / 修改一条事实的请求体（POST 与 PUT 共用）。

    三个字段与 ``save_memory`` 工具、回合后抽取**完全一致**，且同样过 ``validate_fact`` 这道门
    （PII 过滤、长度、key 规范化）——三条写路径一个门，页面不是特权入口。
    """

    key: str
    value: str
    category: str = "preference"


@router.get("/api/preferences/{user_id}")
async def get_preferences(
    user_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读取某用户的长期记忆，供前端「偏好面板」展示。

    **鉴权（堵越权读）：** 开启 ``AUTH_ENABLED`` 后，只能读**自己**的记忆——token 的 sub 与 URL
    段 user_id 不一致即 403。关闭时退回现状（任意读）。

    store 已按 ``updated_at`` 倒序返回，前端看到的第一条就是最近被写过的那条；保留期
    （``MEMORY_RETENTION_DAYS``）内的才返回，与注入给模型的口径一致——页面上看得见的，
    就是模型读得到的。
    """
    _assert_own(user_id, auth_uid)
    facts = await get_fact_store().get_facts(user_id)
    return {"user_id": user_id, "preferences": [_fact_json(f) for f in facts]}


@router.post("/api/preferences/{user_id}")
async def add_preference(
    user_id: str,
    body: FactWrite,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """手填一条长期记忆（同 key 覆盖）。

    **没有「先解析成结构化草稿」那一步了**：上一版要 LLM 把一句自然语言拆成 polarity /
    category / keywords 再让用户确认，是因为那套模型有七八个字段、用户填不出来。事实只有
    key / value / category 三个，直接填即可——省掉一次 LLM 调用，也省掉「解析不出来」的 400。

    过 ``validate_fact``（PII 过滤 + 长度 + key 规范化）：被拒时回 400，**但不回显 value**
    （被拒的多半正是不该扩散的东西，错误消息由 ``MemoryWriteRejected`` 给）。
    """
    _assert_own(user_id, auth_uid)
    if not user_id:
        raise HTTPException(400, "匿名用户无法沉淀记忆")
    try:
        fact = validate_fact(body.key, body.value, body.category, source_session="")
    except MemoryWriteRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    if not await get_fact_store().upsert_facts(user_id, [fact]):
        raise HTTPException(503, "记忆库暂时写不进去，请稍后再试")
    return {"added": [_fact_json(fact)]}


@router.put("/api/preferences/{user_id}/entry/{key:path}")
async def update_preference(
    user_id: str,
    key: str,
    body: FactWrite,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """就地修改一条记忆。改了 key 就是**换一条**：先删旧 key，再按新 key 写。

    URL 里的 ``key`` 是**旧**的（前端本来就有），body 里的是改完的。两者相同时等价于覆盖写，
    无副作用。``:path`` 转换器是历史沿用——``validate_fact`` 规范化后的 key 不含 ``/``，
    但让路由宽容一点，免得前端传了脏 key 时拿到 404 而不是 400。
    """
    _assert_own(user_id, auth_uid)
    try:
        fact = validate_fact(body.key, body.value, body.category, source_session="")
    except MemoryWriteRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    store = get_fact_store()
    if key != fact.key:
        await store.delete_fact(user_id, key)
    if not await store.upsert_facts(user_id, [fact]):
        raise HTTPException(503, "记忆库暂时写不进去，请稍后再试")
    return {"updated": [_fact_json(fact)]}


@router.delete("/api/preferences/{user_id}")
async def clear_preferences(
    user_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, str]:
    """清空该用户全部长期记忆（偏好页的「全部清除」）。

    **同一个事务里把 ``memory_purge_gen`` 加一**：正在跑的回合后抽取会在写库前后各读一次代数，
    发现变了就整批丢弃——否则用户刚点完清空，上一轮的抽取结果转头又落回空库里，看起来就是
    「清了个寂寞」（见 ``fact_store.clear`` 与 ``curator``）。

    幂等：没有记忆的用户照样返回 ok，连点两次不报错。
    """
    _assert_own(user_id, auth_uid)
    await get_fact_store().clear(user_id)
    return {"status": "ok"}


@router.get("/api/session/{thread_id}/constraints")
async def get_session_constraints(
    thread_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读本次会话累积的 P_t 约束集（偏好面板「本次会话」区；打开面板 / 断线重连时主动拉）。

    可见可纠（P_t 重构步骤三①）：约束录入过 LLM 的手（极性判反 / keywords 抽漏照样进 P_t 且无
    自愈性），抽错时唯一的兜底是用户看得见、点得掉。每条带 ``id``（删除按它打 DELETE）与
    （``<词表>:<词>``，lite P_t 没有 source_quote）。会话无 session.json / 读坏 → 空列表（同
    run_agent 开局的容错口径）。
    """
    await guard_thread(thread_id, auth_uid)
    pt = _read_session_pt(safe_session_dir(OUTPUT_ROOT, thread_id))
    return {
        "thread_id": thread_id,
        "epoch": 0,  # lite P_t 无代际；字段保留给前端契约
        "budget_usd": pt.budget_usd,
        "category": pt.category,
        "constraints": constraint_rows(pt),
    }


@router.delete("/api/session/{thread_id}/constraints/{constraint_id}")
async def delete_session_constraint(
    thread_id: str, constraint_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, str]:
    """从本次会话的 P_t 里删一条约束（面板每行的 ×）——抽取出错时的人纠错入口。

    **不走撤回词面核验**：那道闸挡的是 LLM 幻觉 / 抄错 id，用户亲手点的就是那一条，他的删除
    是最高权威（识别 / 授权分离里的「授权」端）。直接按 id 从 active 集移除、写回 session.json
    的 middle_context；下一轮 run_agent 开局读回的就是删除后的状态。不存在的 id / 无 session.json
    静默成功（幂等，连点两次不报错）。删完把新快照推给该 thread 的 WS 连接，面板不必自己再拉一次。

    **与 run_agent 的写点不冲突**：任务在跑时 session.json 只在成功收尾那一刻被整体覆盖，
    这里的删改若与之交错会被那次覆盖冲掉（用户再点一次即可）——不为这个极窄的窗口加锁。
    """
    await guard_thread(thread_id, auth_uid)
    session_dir = safe_session_dir(OUTPUT_ROOT, thread_id)
    state = load_session_state(session_dir)
    if state is None:
        return {"status": "ok"}
    pt = pt_from_state(state.middle_context)
    if drop_constraint(pt, constraint_id):
        pt_into_state(state.middle_context, pt)
        save_session_state(session_dir, state)
        await monitor.report_session_constraints(pt, thread_id=thread_id)
    return {"status": "ok"}


def _read_session_pt(session_dir: Path) -> SessionPrefState:
    """偏好面板读 P_t：从 session.json 的 middle_context 取；无文件 / 读坏 → 空。"""
    state = load_session_state(session_dir)
    return pt_from_state(state.middle_context) if state is not None else SessionPrefState()


@router.delete("/api/preferences/{user_id}/{key:path}")
async def delete_preference(
    user_id: str,
    key: str,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, str]:
    """删除一条记忆（页面上每行的 ×，以及回复下方「记住了 …」的撤销）。

    **这是唯一的删除口，且只有用户能走**：模型侧的「忘掉 X」走 ``save_memory`` 用原 key 覆盖写
    （计划 §3.2 第 2 条）——识别交给模型，授权留给用户。不存在的 key 静默成功（幂等，连点两次
    不该报错）。

    **删了会不会被 Agent 学回来？** 会，但只在用户重新提起同一件事时——那时他本来就是又说了
    一遍。为此加一张 tombstone 表（删除记录 + TTL + 写入前查禁）不值，真被抱怨了再加。
    """
    _assert_own(user_id, auth_uid)
    await get_fact_store().delete_fact(user_id, key)
    return {"status": "ok"}


@router.get("/api/favorites/{user_id}")
async def get_favorites(
    user_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """读取某用户收藏（♡）的商品，供前端「收藏抽屉」展示。新→旧。

    **收藏是纯展示数据**：它不注入 prompt、不进长期偏好库、不影响检索与精挑——刻意如此。
    收藏一件商品并不能可靠地推出任何偏好（可能只是想再比比价），拿它去改 Agent 行为是过度解读。
    这跟同在 Store 里的偏好 / 行为历史是两码事，那两个都会被喂进上下文。
    """
    _assert_own(user_id, auth_uid)
    return {
        "user_id": user_id,
        "favorites": [i.model_dump() for i in await get_store().read_favorites(user_id)],
    }


@router.post("/api/favorites/{user_id}")
async def add_favorite(
    user_id: str,
    body: FavoriteItem,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """收藏一件商品（点 ♡）。同 ``item_id`` 覆盖 → 重复点幂等。

    存的是**商品快照**而非只存 id：收藏跨会话长期留着，而候选登记表（``tools._candidates``）
    随会话清理，换个会话按 id 早捞不回商品了。前端点 ♡ 时手上正好有整张卡的数据，直接送来。
    """
    _assert_own(user_id, auth_uid)
    await get_store().write_favorite(user_id, body)
    return {"user_id": user_id, "item_id": body.item_id, "status": "ok"}


@router.delete("/api/favorites/{user_id}/{item_id}")
async def remove_favorite(
    user_id: str, item_id: str, auth_uid: str | None = Depends(get_current_user_id)
) -> dict[str, Any]:
    """取消收藏。``item_id`` 不存在则静默成功（幂等）。"""
    _assert_own(user_id, auth_uid)
    await get_store().delete_favorite(user_id, item_id)
    return {"user_id": user_id, "item_id": item_id, "status": "ok"}


@router.get("/api/similar/{item_id}")
async def get_similar(
    item_id: str,
    top_k: int = 8,
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, Any]:
    """「搜同款」：拿这件商品的向量在全库找近邻，同步返回一组相似商品。

    **刻意不走 AgentLoop**：这是一次纯向量检索（0 次 LLM 调用、亚秒级），塞进 Agent 只会换来
    几十秒的规划-工具-收尾开销，换不到任何东西。故它不进 ``FULL_TOOL_SET``，就是个 REST 端点。

    **不再按长期记忆过滤**（M4）：原来这里会拿用户亲手勾的「绝不推荐」黑名单挡一遍同款。那条腿
    随长期记忆改成「只经模型上下文生效」一并删了——记忆现在只有一种生效方式，就是模型把它写进
    工具入参，而这条通路根本没有模型。留着它就等于留一条谁也看不见的第二生效通路，正是这次
    重构要消灭的东西。代价：同款列表里可能出现用户说过不喜欢的东西，他可以照样不点。

    返回形状直接对齐前端 ``ProductItem``：只有货价（``price_usd``，建库时预折算），**没有到手价**
    ——那要跑 ``shipping_calc``，不是这条通路该做的事，前端照实标「货价」即可。
    """
    top_k = max(1, min(top_k, 24))
    cands = await asyncio.to_thread(get_recall_client().similar, item_id, top_k)
    return {
        "item_id": item_id,
        "items": [
            {
                "item_id": c.item_id,
                "platform": c.platform,
                "title": c.title,
                "price_usd": c.price_usd,
                "image_url": c.image_url,
                "url": c.url,
                "score": round(c.score, 4),
            }
            for c in cands
        ],
    }
