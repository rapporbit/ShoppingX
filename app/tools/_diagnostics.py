"""工具 → harness 的结构化诊断侧信道（thread 作用域）。

**为什么存在**：harness 的补搜闸 / 阶段机需要工具的执行诊断（picker 的 must_have_hits /
oncat / offcat / picks 件数）。旧通路是「字段挤进模型可见 JSON 头部 + middleware 正则抠」——
拿模型可见字符串当信号总线，与截断 Hook 的砍尾位置强耦合，每加一个字段都要「改序列化位置 +
写正则 + 防截断」三连。本模块把两个受众解开：

  - **给模型看的**照旧走工具 Output 的 ``__str__``（那是动机层，要不要带、带多少按 token 算）；
  - **给 harness 看的**走这里：工具返回前 :func:`report_diagnostics` 登记结构化 dict，
    middleware 在 ``await handler()`` 返回后 :func:`consume_diagnostics` 取走。文本怎么截断、
    格式怎么改，信号都不受影响。

**作用域按 thread_id 而非 session_dir**：fork 子 Agent 继承父 session_dir（产物同目录），但
有自己的 thread_id 与自己的 middleware——诊断是「这个 loop 的这次调用」的私有信号，按
session_dir 聚合会让并行 fork 互相消费对方的诊断。用「按 thread_id 聚合的模块级 dict」而非
裸 ContextVar，理由同 ``app.api.context._RETRIEVAL_MODE``：工具在独立 context 里执行，
ContextVar 的 set 不回传 middleware 所在的 context。

**消费即删除**：middleware 每次工具调用后立即 consume，稳态下表为空；``run_agent`` 收尾再调
:func:`reset_diagnostics` 兜底清理（防 middleware 异常路径漏消费导致无界增长）。
同一工具在同一 thread 内并行调用时按 FIFO 配对——目前唯一的生产者 item_picker 不会并行
调用自身；若将来有工具会，需在 payload 里带调用标识自行区分。

所有函数同步、零 IO、异常静默降级（无 thread 作用域的单测里调用是 no-op）。
"""

from __future__ import annotations

import logging
from collections import deque

from app.api.context import get_thread_id

logger = logging.getLogger("shoppingx.diagnostics")

# 每 (thread, tool) 最多积压的未消费诊断条数。稳态是 0～1（消费即删除）；上限只为兜住
# 「middleware 异常路径持续漏消费」的病态，防模块级 dict 无界增长。
_MAX_PENDING = 8

# thread_id -> tool_name -> 未消费的诊断 payload 队列（FIFO）
_CHANNEL: dict[str, dict[str, deque[dict[str, object]]]] = {}


def report_diagnostics(tool: str, payload: dict[str, object]) -> None:
    """工具在返回前登记本次调用的诊断字段。无 thread 作用域时静默 no-op。"""
    tid = get_thread_id()
    if tid is None:
        return
    queue = _CHANNEL.setdefault(tid, {}).setdefault(tool, deque(maxlen=_MAX_PENDING))
    queue.append(dict(payload))


def consume_diagnostics(tool: str) -> dict[str, object] | None:
    """取走当前 thread 上 ``tool`` 最早一条未消费诊断；没有则返回 None。"""
    tid = get_thread_id()
    if tid is None:
        return None
    queue = _CHANNEL.get(tid, {}).get(tool)
    if not queue:
        return None
    return queue.popleft()


def reset_diagnostics(thread_id: str | None = None) -> None:
    """清空指定 thread（缺省取当前 context）的全部未消费诊断。run_agent 收尾兜底用。"""
    tid = thread_id if thread_id is not None else get_thread_id()
    if tid is None:
        return
    _CHANNEL.pop(tid, None)
