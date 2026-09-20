"""参数覆盖的持久化：``config_overrides`` 表 ↔ 覆盖层内存。

**库是唯一真相，内存（``overrides._OVERRIDES``）只是本进程的缓存**（阶段 1 条 8）。这个方向在
2026-09-20 反过来了：在那之前是「内存权威、库是持久影子」，成立的前提写在 overrides.py 头上——
「部署是 ``--workers 1`` 单进程」。API 与 worker 拆成两个进程之后，那个前提没了：后台改参数打在
API 进程上，而 AgentLoop 在 worker 里跑，worker 的 ``os.environ`` 纹丝不动。坏得还很安静——前端
读的是 API 进程的 ``current_value()``，显示「已生效」，实际推理照旧用旧值。

所以每个进程都按 :data:`SYNC_INTERVAL_SEC` 轮询这张表对账（:func:`refresh`）。

**为什么轮询而不是走 control.py 那条广播**：配置是**状态**，不是事件。广播是 fire-and-forget，
``publish`` 失败只打 warning，worker 重连期间错过一条就永远是旧值、不会自愈；轮询则是每一轮都重新
对账，任何一次丢失下一轮补回来。取消指令必须实时所以走广播，参数晚一轮生效没有代价。

分成两个模块而非塞进 overrides.py，是为了让覆盖层本身不依赖数据库——测试与脚本可以只 import
overrides 做纯内存的参数改动，不用起库。
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import delete, select

from app.config import overrides
from app.config.registry import BY_KEY
from app.db.models import ConfigOverride
from app.db.session import session_factory
from app.utils.env import env_int

logger = logging.getLogger(__name__)

# 每个进程多久与库对一次账。30s 是「改完等得起」与「别把库当心跳打」之间的折中：一次全表 select，
# 表里最多几十行。不进 registry（后台可改参数表）——把轮询间隔做成轮询才能改的参数是个圈。
SYNC_INTERVAL_SEC = 30


async def _read_desired(*, quiet: bool) -> dict[str, str]:
    """读全表，返回「库希望生效的」key → **规范化后**的值。

    **绝不因为一条脏数据让服务起不来**——库里可能留着已从注册表删掉的 key，或范围收紧后变得
    非法的旧值。这类行逐条跳过（值继续跟随 .env / 代码默认），而不是让 lifespan 抛异常。

    值必须在这里就规范化：库里存的是 ``normalize`` 之前的原始字符串（``"8 "``、``"true"``），
    而内存里是规范化后的（``"8"``、``"1"``）。拿原值跟内存比，每一轮都判成「有差异」，于是每 30s
    白重载一遍所有模块——不报错，只是悄悄地一直在做无用功。

    ``quiet=True`` 时不为脏数据记 warning：轮询每 30s 撞见同一行脏数据，日志会被刷爆，启动那次
    已经说过了。
    """
    async with session_factory()() as session:
        rows = (await session.execute(select(ConfigOverride))).scalars().all()

    desired: dict[str, str] = {}
    for row in rows:
        if row.key not in BY_KEY:
            if not quiet:
                logger.warning("跳过库里的未知参数 %s（注册表已删？），其值不再生效", row.key)
            continue
        try:
            desired[row.key] = overrides.normalize(row.key, row.value)
        except overrides.ParamValidationError as e:
            if not quiet:
                logger.warning("跳过库里的非法参数值 %s=%r：%s", row.key, row.value, e)
    return desired


async def refresh(*, quiet: bool = True) -> tuple[list[str], list[str]]:
    """与库对一次账，返回 ``(改了的 key, 退回默认的 key)``。无差异时一个模块都不重载。

    两个方向都要走，缺一个就是半个真相：

    - 库里有、内存里没有或值不同 → :func:`overrides.apply`；
    - **内存里有、库里已经没有** → :func:`overrides.reset` 退回 ``.env`` 基线。后一条是 2026-09-20
      之前就存在的洞：老的 ``load_into_memory`` 只做加法，后台点「恢复默认」删掉库里的行，当时那个
      进程靠 ``overrides.reset`` 自己改了内存所以看着是对的，**别的进程要等到重启才退回去**。
    """
    desired = await _read_desired(quiet=quiet)
    current = overrides.active_overrides()

    to_apply = {k: v for k, v in desired.items() if current.get(k) != v}
    to_reset = [k for k in current if k not in desired]

    if to_apply:
        overrides.apply(to_apply)
    if to_reset:
        overrides.reset(to_reset)
    return sorted(to_apply), sorted(to_reset)


async def load_into_memory() -> int:
    """启动时把库里的覆盖读进内存并生效。返回成功应用的条数。

    走的是与轮询同一条 :func:`refresh`——启动只是「内存为空」那一次特例，不值得两套代码各自演化。
    """
    applied, _ = await refresh(quiet=False)
    if applied:
        logger.info("已从库载入 %d 条参数覆盖：%s", len(applied), ", ".join(applied))
    return len(applied)


async def sync_loop(interval: int | None = None) -> None:
    """每 ``interval`` 秒与库对一次账。由 API lifespan 与 worker 各起一份，**永不因报错退出**。

    库抖一下就让这个循环死掉，后果是那个进程从此再不跟进任何参数改动，而且没有任何迹象——它只是
    不再变了而已。所以这里吞掉所有异常接着睡下一轮，唯独放过 ``CancelledError``（那是关停）。
    """
    every = (
        interval if interval is not None else env_int("CONFIG_SYNC_INTERVAL_SEC", SYNC_INTERVAL_SEC)
    )
    while True:
        await asyncio.sleep(every)
        try:
            applied, reset = await refresh()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("参数覆盖对账失败，%d 秒后重试", every, exc_info=True)
            continue
        if applied or reset:
            logger.info(
                "参数覆盖已跟进：改 %s / 退回默认 %s",
                ", ".join(applied) or "—",
                ", ".join(reset) or "—",
            )


async def save(values: dict[str, str], updated_by: str) -> None:
    """把已规范化的覆盖 upsert 进库（调用方负责先经 :func:`overrides.normalize` 校验）。"""
    async with session_factory()() as session:
        for key, value in values.items():
            row = await session.get(ConfigOverride, key)
            if row is None:
                session.add(ConfigOverride(key=key, value=value, updated_by=updated_by))
            else:
                row.value = value
                row.updated_by = updated_by
        await session.commit()


async def remove(keys: list[str]) -> None:
    """删除覆盖行 = 恢复默认：该参数之后重新跟随 ``.env`` / 代码默认值。"""
    if not keys:
        return
    async with session_factory()() as session:
        await session.execute(delete(ConfigOverride).where(ConfigOverride.key.in_(keys)))
        await session.commit()
