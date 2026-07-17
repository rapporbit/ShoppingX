"""请求指纹去重（refdocs 16-5 §3.2 第三层）——用户手抖连点两下，不该跑两遍 Agent。

**为什么光靠 ``active_tasks`` 不够。** ``active_tasks`` 按 ``thread_id`` 索引，只能挡住「同一会话里
重复提交」。不管 thread_id 的调用方（脚本、裸 API、压测器）每次都拿一个新 thread_id，同一条 query
连发十遍就真跑十遍——``active_tasks`` 完全看不见。指纹去重守的就是这个缺口：key 里刻意**不含
thread_id**。

**只对「不自带 thread_id」的调用方生效**（判定在 ``server.create_task`` 里）。本项目前端是
connect-first：它自己生成 thread_id、先连 WS、再 POST。把它并进别人的 thread_id 会让它那条 WS
一个事件都收不到、界面永远转圈；而且它的 thread_id 存在 localStorage、刷新不换，refdocs 设想的
「刷新 → 换 thread_id → 重复提交」在本前端根本不会发生。详见 ``server.create_task`` 的注释。

**窗口为什么这么短（默认 5 秒）。** 这层要区分的是「手抖 / 刷新」和「用户真的想再问一次」。
5 秒足够覆盖前者（双击、刷新、网络重试都在这个量级），又不会误伤后者——没有哪个用户会精确地在
5 秒内故意重发完全相同的一句话，就算真发了，代价也不过是拿到上一次的 thread_id 继续看结果。

**refdocs 的第二层（Checkpoint 防重跑）本项目不做。** 它依赖 LangGraph checkpointer 把中断的图
状态存进 Redis、重启后从断点恢复。本项目是「一请求一进程内 async task、跑到完成或取消」，
没有跨进程恢复需求（CLAUDE.md §2.2 已把它划在范围外，挂 checkpointer 只会平添依赖与不消费的
持久化语义）。所以幂等只有第一层（active_tasks）和第三层（本模块）。

**进程内、不共享。** 多副本部署时每个副本各有一份指纹表，跨副本的重复提交挡不住。要挡得把表挪
进 Redis——但那要为一个「省一次重复请求」的功能引入一次网络往返，在当前单副本 demo 里不划算。
诚实标注在这，将来真多副本了改这一个文件即可。
"""

from __future__ import annotations

import hashlib
import time

from app.utils.env import env_float

# 同一 (user_id, query) 在这个窗口内再次提交 → 判为重复。
DEDUP_WINDOW_SEC = env_float("TASK_DEDUP_WINDOW_SEC", 5.0)

# fingerprint → (提交时刻, 当时的 thread_id)。记 thread_id 是为了让重复提交能被**引导回原任务**
# （返回它的 thread_id，前端连上去继续看事件流），而不是干巴巴回一个「重复请求」错误。
_recent: dict[str, tuple[float, str]] = {}


def _fingerprint(user_id: str | None, query: str) -> str:
    """(user_id, query) 的稳定指纹。**不含 thread_id**——本层要挡的正是换 thread_id 的重复提交。"""
    raw = f"{user_id or ''}:{query.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _purge(now: float) -> None:
    """清掉过期指纹。表很小（活跃用户数量级），每次提交顺手扫一遍比起后台定时器简单得多。"""
    expired = [fp for fp, (ts, _) in _recent.items() if now - ts > DEDUP_WINDOW_SEC]
    for fp in expired:
        _recent.pop(fp, None)


def check_duplicate(user_id: str | None, query: str) -> str | None:
    """窗口内重复提交则返回**上一次的 thread_id**（不登记），否则返回 ``None``。

    返回原 thread_id 而不是布尔值，是为了让调用方能把用户引导回那个正在跑的任务——
    重复提交的正确处理是「你要的东西已经在做了，看这里」，不是「请求被拒绝」。

    **查询与登记刻意分成两步**（本函数只查，:func:`remember` 才登记）：被 429 / 被判
    ``already_running`` 的请求**不该**留下指纹，否则用户退避重试会被当成「重复提交」再拒一次，
    陷入死循环。**只有真正启动的任务才配拥有指纹。**

    代价是调用方必须自己保证 ``check_duplicate`` → ``remember`` 之间**不出现 ``await``**——中间一旦
    让出事件循环，两个并发请求会双双查到「不重复」然后各跑一遍，去重就漏在自己的窗口里。
    ``server.create_task`` 目前满足这个约束（与 ``try_reserve`` 同处一个无 await 区间）。
    """
    _purge(time.monotonic())
    hit = _recent.get(_fingerprint(user_id, query))
    return hit[1] if hit is not None else None


def remember(user_id: str | None, query: str, thread_id: str) -> None:
    """登记一次提交（调用方确认要真跑这个任务之后再调）。理由见 :func:`check_duplicate`。

    **这里也要 purge**：``check_duplicate`` 只在「调用方不自带 thread_id」时才被调到（见
    ``server.create_task``），而 ``remember`` 每个启动的任务都调。只在前者里清理的话，走前端那条
    路（永远自带 thread_id）的部署里 ``_purge`` 一次都跑不到，``_recent`` 会随任务数无界增长。
    """
    now = time.monotonic()
    _purge(now)
    _recent[_fingerprint(user_id, query)] = (now, thread_id)


def reset() -> None:
    """清空指纹表（测试用）。"""
    _recent.clear()
