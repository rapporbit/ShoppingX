"""对话历史持久化 —— 把每段 thread 的多轮对话落盘，支撑前端「回看」。

和 :mod:`app.memory.store` 的长期记忆是两回事，别混：
- store.py 存的是**跨会话的偏好结论**（「不要塑料」），按用户聚合、只在识别到新偏好时写。
- 本模块存的是**单个 thread_id 内的逐轮对话**，按 thread 隔离、随轮数增长。

两份产物各司其职，**存放位置刻意不同**：
- **逐轮 ``user → assistant`` 对 → 关系库的 ``messages`` 表**（累加写）。它**只给前端「回看本段
  对话」**，Agent 不读它（续聊靠 ``session.json`` 恢复完整上下文，不回放精简 (q,a)）。
  **为什么从 turns.json 搬进库**：``threads`` 表已经
  把「这段会话归谁」搬进库了，正文却还在文件系统上——两者生命周期一旦不一致（换机器 / 卷没挂上 /
  多副本各写各的盘），侧栏就会列出一段点进去空白的会话。元信息与正文必须同生共死，见
  :class:`app.db.models.Message`。
- ``session.json`` —— ``AgentState`` 全量（模型看得见的完整上下文 + P_t），写在会话目录
  ``output/<thread_id>/``，由 ``orchestrator`` 维护。它是**续聊的唯一读源**；本模块不碰。

**旧会话怎么办（惰性迁移）。** 库上线前的会话，正文还在各自的 ``turns.json`` 里。读 / 写任一路径
发现「库里没有这个 thread 但磁盘上有旧文件」，就把旧文件整段导进库、之后只认库——不需要停机跑
迁移脚本，也不会漏掉那些再也没人点开的老会话（它们本就不必迁）。旧文件读完保留不删：迁移出岔子
时那是唯一的后路。

**容错口径（与 store.py 同一套「降级不崩」）：** 历史是附带产物，读 / 写任一环出问题都只记
日志降级——回看返回空列表，绝不反噬主链路（任务该跑还跑）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message
from app.db.session import session_factory

logger = logging.getLogger("shoppingx.history")

# turns.json 里合法 role 的白名单（惰性迁移旧文件时校验用）。
_VALID_ROLES = {"user", "assistant"}


def _turns_path(session_dir: Path) -> Path:
    """旧会话的正文文件（库上线前的产物）。只读、不再写——见模块 docstring 的惰性迁移。"""
    return session_dir / "turns.json"


def _new_id() -> str:
    return uuid4().hex


def _row_to_turn(row: Message) -> dict[str, Any]:
    """一行 ``messages`` → 前端认得的轮次 dict。可选字段仅在非空时出现，与旧 turns.json 逐字同形，
    故前端不必改（搬家只搬存储，不改契约）。"""
    rec: dict[str, Any] = {"role": row.role, "content": row.content}
    if row.role == "user" and row.images:
        rec["images"] = row.images
    if row.role == "assistant":
        if row.items:
            rec["items"] = row.items
        if row.activity:
            rec["activity"] = row.activity
        if row.elapsed_ms is not None:
            rec["elapsed_ms"] = row.elapsed_ms
        if row.tokens is not None:
            rec["tokens"] = row.tokens
    return rec


async def _fetch(db: AsyncSession, thread_id: str) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            select(Message).where(Message.thread_id == thread_id).order_by(Message.seq)
        )
    ).scalars()
    return [_row_to_turn(r) for r in rows]


async def _count(db: AsyncSession, thread_id: str) -> int:
    n = (await db.execute(select(func.count()).where(Message.thread_id == thread_id))).scalar_one()
    return int(n or 0)


def _to_rows(thread_id: str, turns: list[dict[str, Any]], start_seq: int) -> list[Message]:
    return [
        Message(
            id=_new_id(),
            thread_id=thread_id,
            seq=start_seq + i,
            role=t["role"],
            content=t["content"],
            items=t.get("items"),
            activity=t.get("activity"),
            tokens=t.get("tokens"),
            elapsed_ms=t.get("elapsed_ms"),
            images=t.get("images"),
        )
        for i, t in enumerate(turns)
    ]


async def _backfill_legacy(db: AsyncSession, thread_id: str, session_dir: Path | None) -> int:
    """把旧 turns.json 整段导进库（仅当库里这个 thread 一条都没有）。返回导入的条数。

    **只在库为空时导**：库里已有内容说明这个 thread 早就迁过（或本就是库时代新建的），再导一遍
    就是把老轮次插到新轮次之后，把一段对话的顺序搅乱。这也让本函数天然幂等——反复调用只有第一次
    真的写。
    """
    if session_dir is None or await _count(db, thread_id) > 0:
        return 0
    legacy = _load_turns_raw(_turns_path(session_dir))
    if not legacy:
        return 0
    db.add_all(_to_rows(thread_id, legacy, 0))
    await db.commit()
    logger.info("旧会话正文已迁入库（thread=%s，%d 条）", thread_id, len(legacy))
    return len(legacy)


def _load_turns_raw(path: Path) -> list[dict[str, Any]]:
    """读出 turns.json 并逐条校验，损坏 / 缺失一律按空处理（记日志，不抛）。

    单条结构不对（缺 role / content 非串 / role 非白名单）就跳过那条，保住其余完好的轮次——
    和 store.py「单条坏掉跳过、不连坐整文件」同一手法。

    assistant 轮额外携带两个**回看专用**的可选字段（只在结构合法时保留，缺/坏则静默忽略，
    不影响这一轮的文本）：``items``（精选商品卡）、``activity``（思考过程的 AGUI 事件流）。
    Agent 不读这张表，这两个字段只服务回看。
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("对话历史无法读取/解析，按空处理（path=%s）：%s", path, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("对话历史结构异常（顶层非列表），按空处理（path=%s）", path)
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not (
            isinstance(item, dict)
            and item.get("role") in _VALID_ROLES
            and isinstance(item.get("content"), str)
        ):
            logger.warning("跳过损坏的对话轮次（path=%s）：%r", path, item)
            continue
        rec: dict[str, Any] = {"role": item["role"], "content": item["content"]}
        if item["role"] == "assistant":
            items = item.get("items")
            if isinstance(items, list):
                rec["items"] = items
            activity = item.get("activity")
            if isinstance(activity, list):
                rec["activity"] = activity
            # bool 是 int 子类，得显式排除，免得把 True 当耗时存进去。
            elapsed = item.get("elapsed_ms")
            if isinstance(elapsed, int) and not isinstance(elapsed, bool):
                rec["elapsed_ms"] = elapsed
            # 本轮 token 用量（input/output/total/cost_usd），供回看在右下角还原「token 消耗」。
            tokens = item.get("tokens")
            if isinstance(tokens, dict):
                rec["tokens"] = tokens
        out.append(rec)
    return out


async def read_turns(thread_id: str, session_dir: Path | None = None) -> list[dict[str, Any]]:
    """读出本段会话的全部逐轮对话（前端「回看」用）。无历史则返回空列表。

    assistant 轮可能带 ``items``（商品卡）/ ``activity``（思考过程）供前端原样还原渲染。
    给了 ``session_dir`` 就顺带把旧 turns.json 惰性迁进库（老会话第一次被点开时自动补上）。

    库故障返回空而非抛：回看退化为「这段会话没有历史」，与旧版读文件失败时同一口径。
    """
    try:
        async with session_factory()() as db:
            await _backfill_legacy(db, thread_id, session_dir)
            return await _fetch(db, thread_id)
    except SQLAlchemyError as exc:
        logger.warning("读取对话历史失败，按空处理（thread=%s）：%s", thread_id, exc)
        return []


async def append_turn(
    thread_id: str,
    query: str,
    final_text: str,
    items: list[dict[str, Any]] | None = None,
    activity: list[dict[str, Any]] | None = None,
    elapsed_ms: int | None = None,
    tokens: dict[str, Any] | None = None,
    session_dir: Path | None = None,
    images: list[str] | None = None,
    experiment: dict[str, Any] | None = None,
) -> None:
    """把本轮 ``user → assistant`` 这一对追加进 ``messages`` 表（累加写，供前端回看）。

    ``items``（精选商品卡）/ ``activity``（思考过程 AGUI 事件流）/ ``elapsed_ms``（本轮耗时）/
    ``tokens``（本轮全树 token 用量）挂在 assistant 轮上，是**回看**的还原源——让历史轮也能画出
    商品卡、「思考过程」折叠区、用时与 token 消耗，而非只剩一段结论文本。皆可选、仅非空才写（闲聊
    兜底无商品卡、无连接时无事件流、未记账时无 tokens）。

    在 ``run_agent`` 收尾调用。写失败只记日志降级——一轮没落库，前端少一轮回看，但绝不该
    拖垮主链路的偏好写回与 task_result 上报（用户已经拿到答案了）。
    """
    assistant: dict[str, Any] = {"role": "assistant", "content": final_text}
    if items:
        assistant["items"] = items
    if activity:
        assistant["activity"] = activity
    if elapsed_ms is not None:
        assistant["elapsed_ms"] = elapsed_ms
    if tokens is not None:
        assistant["tokens"] = tokens
    if experiment:
        assistant["experiment"] = experiment  # 提示词版本 / A/B 桶 / 策略 / skill（回看那行 chip）
    user: dict[str, Any] = {"role": "user", "content": query}
    if images:
        # 参考图挂 user 轮：它是用户这次「说的话」的一部分，回看时该跟 query 一起显示。
        user["images"] = list(images)
    turn = [user, assistant]
    try:
        async with session_factory()() as db:
            # 先补迁旧文件再续号：老会话在库时代的第一轮，得接在它那些老轮次之后，而不是从 0 开始
            # 把顺序（和唯一约束）撞了。
            await _backfill_legacy(db, thread_id, session_dir)
            db.add_all(_to_rows(thread_id, turn, await _count(db, thread_id)))
            await db.commit()
    except SQLAlchemyError as exc:
        logger.warning("追加对话轮次失败，本轮未落库（thread=%s）：%s", thread_id, exc)
