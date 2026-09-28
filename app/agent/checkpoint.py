"""按步检查点：worker 崩溃后，接管方从最近一步接着跑，而不是整轮重跑。

**存什么。** 一步 = 框架的一次 reasoning-acting 迭代。续跑需要的是三样东西在同一时刻的快照：
``AgentState``（模型视野 + 最后一条消息里「有调用无结果」的工具，框架续跑时会直接补跑它们，
见 ``tests/test_resume_from_checkpoint.py``）、run 状态表（候选登记表 / 槽表 / 检索预算 / 用量…，
见 :mod:`app.api.run_state`）、控制面 :class:`~app.harness.session.HarnessSession`
（prefill 做没做过、autopick 武装位、GuardState…）。外加 ``cur_iter``（框架续跑时会清零，要写回）
和本轮起点 ``turn_start``（收尾只在本轮消息里找产物）。

**什么时候存。** 两处，都在 :class:`~app.harness.adapter.HarnessAgentAdapter`：模型调用前（上一步
的工具结果已全部落进 context）、reasoning 刚结束（模型决定了调哪些工具、工具还没跑——崩在工具
里时省一次模型调用）。

**怎么编码。** ``AgentState`` 走框架自己的 JSON；另外两样是嵌套的普通类，逐个手写编解码既啰嗦
又容易在加字段时漏，走 pickle（RQ 的任务体默认也是 pickle 进 Redis）。数据只从我们自己的 Redis
读，与队列消息同一个信任边界。pickle 的真正风险是**版本漂移**：新代码给某个类加了字段，旧检查点
解出来的对象缺这个属性，跑到半路才 ``AttributeError``。所以 :func:`load` 逐个比对属性集合，对不上
就当没有检查点——退回整轮重跑，也就是没有这个模块时的行为，只会更好不会更坏。

**放哪、活多久。** 队列那个 Redis（本项目的硬依赖），键 ``globex:ckpt:<run_id>``，TTL 默认 30 分钟
（远大于接管窗口与整轮超时）。成功收尾即删；失败的留着给重投用，到期自然消失。
所有操作**从不抛**：检查点是加速续跑的手段，存不上只是退回整轮重跑，不能反过来拖垮本轮。
"""

import logging
import pickle
import zlib
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.utils.env import env_int

logger = logging.getLogger("shoppingx.checkpoint")

KEY_PREFIX = "globex:ckpt:"
CHECKPOINT_TTL_SEC = env_int("CHECKPOINT_TTL_SEC", 1800)
_FORMAT = 1  # 负载格式本身的版本；改了 _Payload 的键就 +1，旧检查点整批作废

_UNSET: Any = object()
_client: Any = _UNSET  # _UNSET = 按环境变量懒建；None = 关闭


@dataclass
class Checkpoint:
    state_json: str
    cur_iter: int
    turn_start: int
    run_slots: dict[type, object]
    harness: Any  # HarnessSession；不在模块顶层 import，免得 app.agent → app.harness 成环


def _get_client() -> Any:
    global _client
    if _client is _UNSET:
        try:
            import redis.asyncio as aredis

            from app.queue import queue_redis_url

            _client = aredis.from_url(
                queue_redis_url(), socket_connect_timeout=1.0, socket_timeout=2.0
            )
        except Exception as exc:  # 包没装 / URL 非法：关掉，不每步都试一次
            logger.warning("检查点 Redis 客户端建不起来，本进程不写检查点：%s", exc)
            _client = None
    return _client


def set_client(client: Any) -> None:
    """注入客户端（测试用）。传 ``None`` = 关闭检查点。"""
    global _client
    _client = client


def reset() -> None:
    """复位为「按环境变量懒建」。"""
    set_client(_UNSET)


def _key(run_id: str) -> str:
    return f"{KEY_PREFIX}{run_id}"


def _shape_ok(obj: Any, depth: int = 0) -> bool:
    """解出来的对象与当前代码的同类实例属性集合一致吗（只查本仓自己的类，容器看首个元素）。"""
    if depth > 4:
        return True
    if isinstance(obj, BaseModel):
        return set(obj.__dict__) == set(type(obj).model_fields)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return not obj or _shape_ok(next(iter(obj)), depth + 1)
    if isinstance(obj, dict):
        return not obj or _shape_ok(next(iter(obj.values())), depth + 1)
    if not type(obj).__module__.startswith("app.") or not hasattr(obj, "__dict__"):
        return True
    try:
        fresh = type(obj)()
    except Exception:
        return False
    if set(vars(obj)) != set(vars(fresh)):
        return False
    return all(_shape_ok(v, depth + 1) for v in vars(obj).values())


async def save(run_id: str, agent: Any, session: Any, turn_start: int) -> None:
    """把此刻的续跑所需状态写进 Redis。``run_id`` 为空（非队列入口）或已关闭时什么都不做。"""
    client = _get_client()
    if client is None or not run_id:
        return
    try:
        from app.api.run_state import snapshot_run_state

        # 编码是同步的一整段：中间不让出事件循环，三样东西必然是同一时刻的快照。
        payload = {
            "format": _FORMAT,
            "state": agent.state.model_dump_json(),
            "cur_iter": agent.state.cur_iter,
            "turn_start": turn_start,
            "slots": snapshot_run_state(),
            "harness": session,
        }
        blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), 1)
        await client.set(_key(run_id), blob, ex=CHECKPOINT_TTL_SEC)
    except Exception as exc:
        logger.warning("写检查点失败（run_id=%s），本步照常继续：%s", run_id, exc)


async def load(run_id: str) -> Checkpoint | None:
    """读回检查点。没有 / 解不开 / 与当前代码结构对不上都返回 ``None``（= 整轮重跑）。"""
    client = _get_client()
    if client is None or not run_id:
        return None
    try:
        raw = await client.get(_key(run_id))
        if not raw:
            return None
        payload = pickle.loads(zlib.decompress(raw))
        if payload.get("format") != _FORMAT:
            return None
        slots, harness = payload["slots"], payload["harness"]
        if not all(_shape_ok(o) for o in (*slots.values(), harness)):
            logger.warning("检查点与当前代码结构不一致（发版前留下的？），整轮重跑：%s", run_id)
            return None
        return Checkpoint(
            state_json=payload["state"],
            cur_iter=int(payload["cur_iter"]),
            turn_start=int(payload["turn_start"]),
            run_slots=slots,
            harness=harness,
        )
    except Exception as exc:
        logger.warning("读检查点失败（run_id=%s），整轮重跑：%s", run_id, exc)
        return None


async def discard(run_id: str) -> None:
    client = _get_client()
    if client is None or not run_id:
        return
    try:
        await client.delete(_key(run_id))
    except Exception as exc:
        logger.warning("删检查点失败（run_id=%s），等 TTL 过期：%s", run_id, exc)
