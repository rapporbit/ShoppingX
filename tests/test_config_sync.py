"""参数覆盖的跨进程对账（阶段 1 条 8）。

**这个文件存在的理由**：2026-09-20 之前部署是 `--workers 1` 单进程，「内存是权威运行态」成立；
API 与 worker 拆开之后不成立了——后台改参数打在 API 进程，AgentLoop 在 worker 里跑。这里的用例
全部围绕一件事：**另一个进程能不能自己发现库变了**。用「清空 `_OVERRIDES` 再 refresh」来扮演那个
没收到任何通知的进程——它和真 worker 的处境是一样的，只有库这一个信源。
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

import pytest

from app.config import overrides, store
from app.config.registry import BY_KEY, PARAMS
from app.tools import item_picker

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _restore_params():
    """进出都清：env、内存覆盖、模块常量、**库行**。

    库行必须**进场时**也清一次。别的测试文件（test_admin_config 那几个密钥用例）会往
    ``config_overrides`` 里写行且不收拾，而本文件的断言是「这一轮恰好改了哪些 key」——留一行
    `OPENAI_API_KEY` 在库里，单跑绿、全量跑红，且报错指向本文件，看起来像是本文件的 bug。
    """
    snapshot = {p.key: os.environ.get(p.key) for p in PARAMS}
    await store.remove([p.key for p in PARAMS])
    yield
    await store.remove([p.key for p in PARAMS])
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    overrides._OVERRIDES.clear()
    item_picker._load_params()


def _as_another_process() -> None:
    """扮演「刚起来、或者根本没收到通知」的另一个进程：内存里什么覆盖都没有。"""
    overrides._OVERRIDES.clear()
    for p in PARAMS:
        os.environ.pop(p.key, None)
    item_picker._load_params()


async def test_another_process_picks_up_a_change() -> None:
    """条 8 的本职：A 进程改参落库，B 进程对一次账就拿到新值，模块常量真的跟着变。"""
    await store.save({"PICK_DISPLAY_CAP": "7"}, updated_by="admin")
    _as_another_process()
    assert item_picker.PICK_DISPLAY_CAP == BY_KEY["PICK_DISPLAY_CAP"].default

    applied, reset = await store.refresh()

    assert applied == ["PICK_DISPLAY_CAP"]
    assert reset == []
    assert item_picker.PICK_DISPLAY_CAP == 7


async def test_deleted_row_falls_back_to_baseline() -> None:
    """库里的行被删（后台点了恢复默认）→ 别的进程也要退回去。

    这是老 ``load_into_memory`` 的洞：它只做加法。点恢复默认的那个进程自己调了 ``overrides.reset``
    所以看着是对的，其余进程会一直抱着旧覆盖值跑到下次重启。
    """
    await store.save({"PICK_DISPLAY_CAP": "7"}, updated_by="admin")
    await store.refresh()
    assert item_picker.PICK_DISPLAY_CAP == 7

    await store.remove(["PICK_DISPLAY_CAP"])
    applied, reset = await store.refresh()

    assert applied == []
    assert reset == ["PICK_DISPLAY_CAP"]
    assert item_picker.PICK_DISPLAY_CAP == BY_KEY["PICK_DISPLAY_CAP"].default


async def test_no_diff_reloads_nothing() -> None:
    """值没变就一个模块都不许重载——否则每 30s 白重载一遍全部模块，不报错，只是一直做无用功。"""
    await store.save({"PICK_DISPLAY_CAP": "7"}, updated_by="admin")
    await store.refresh()

    with mock.patch.object(overrides, "_reload") as reload_spy:
        applied, reset = await store.refresh()

    assert (applied, reset) == ([], [])
    reload_spy.assert_not_called()


async def test_unnormalized_db_value_is_not_a_permanent_diff() -> None:
    """库里存着 ``"7 "`` 这种没规范化的老值，也只能触发一次变更，不能每轮都判成「有差异」。"""
    await store.save({"PICK_DISPLAY_CAP": "7 "}, updated_by="legacy")

    first, _ = await store.refresh()
    second, _ = await store.refresh()

    assert first == ["PICK_DISPLAY_CAP"]
    assert second == []  # 比对两边都规范化过，第二轮认得出它们是同一个值
    assert item_picker.PICK_DISPLAY_CAP == 7


async def test_dirty_row_is_skipped_not_fatal() -> None:
    """注册表里已经没有的 key 留在库里 → 跳过，不能让对账（进而让起服）炸掉。"""
    await store.save({"PICK_DISPLAY_CAP": "7"}, updated_by="admin")
    with mock.patch.dict(store.BY_KEY, {}, clear=True):
        applied, reset = await store.refresh()

    assert (applied, reset) == ([], [])


async def test_sync_loop_survives_a_db_error() -> None:
    """库抖一下不能让循环死掉——死了这个进程从此不跟进任何改动，且没有任何迹象。"""
    calls = 0
    survived = asyncio.Event()

    async def flaky() -> tuple[list[str], list[str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("库连不上")
        if calls >= 3:  # 第一轮抛了，后面两轮照常 → 循环活着
            survived.set()
        return [], []

    with mock.patch.object(store, "refresh", flaky):
        task = asyncio.create_task(store.sync_loop(interval=0))
        await asyncio.wait_for(survived.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls >= 3
