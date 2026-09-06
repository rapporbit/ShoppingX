"""按轮聚合 LLM token 用量：携带上下文大小 + 缓存命中率（测量驱动 L3 决策）。

**为什么要测这个**：主模型是 1M 窗口，所以「溢出窗口」不会发生——真正要盯的是两件事：
  1. **上下文越长模型越笨**（注意力稀释 / lost-in-the-middle），即便不溢出也会掉召回质量；
  2. **成本**：每轮都重发压缩后的历史，input 计费随轮数累积。

四个指标：
  - ``model_calls``：本轮真实的模型调用次数。
  - ``carried_input_tokens``：各次调用发出的 input token 之和（= 压缩后真实携带量）。
  - ``peak_input_tokens``：单次调用的最大 input（≈ 本轮上下文峰值，最贴近「会不会变笨」）。
  - ``cache_read_tokens`` / ``cache_hit_rate``：验证 cache-breakpoint 是否真命中（应偏高）。

这些数发给 Langfuse（score）+ 日志，作为「是否 / 何时上摘要压缩」的**判据闸门**。

**数据源要看清楚**：``Msg.usage`` 是**每条消息**一份，而一次 reply（一整轮，含十几次模型调用）
在这里只落成**一条** assistant 消息——直接数消息就会得到「model_calls 恒为 1」这种废指标，
而且 carried / cache_read 只反映最后一次调用。所以真实口径要从**记账树**取
（:func:`app.agent.token_budget.tree_snapshot`，那里每次调用都入过一次账），消息侧只用来补
``peak``（树只累加、不留单次极值）。传 ``tree`` 就走这条真实口径。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UsageSummary:
    """一轮对话的 token 用量聚合。"""

    model_calls: int  # 本轮模型调用次数（= AIMessage 带 usage 的条数）
    carried_input_tokens: int  # 各次 input 之和（压缩后真实携带量）
    peak_input_tokens: int  # 单次最大 input（上下文峰值）
    output_tokens: int  # 各次 output 之和
    cache_read_tokens: int  # 命中缓存的 input token 之和
    cache_hit_rate: float  # cache_read / carried_input，0~1（无调用时 0）

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_usage(
    messages: Sequence[object],
    tree: dict[str, float | int] | None = None,
) -> UsageSummary:
    """聚合一轮的 token 用量。

    Args:
        messages: 本轮的 ``list[Msg]``，用来取 ``peak``（单次最大 input）。
        tree: 记账树快照（``tree_snapshot()``）。**给了就以它为准**——每次模型调用都在那里入过
            账，才是真实的次数与总量；不给则退回只数消息（次数会偏小，见模块 docstring）。

    字段口径与迁移前**逐字一致**，这样两条链路的用量表能直接对照——否则「迁移后 token 涨了」
    这种结论根本没法判是真涨了还是换了口径。任一字段缺失按 0，绝不抛：观测是附属品，
    不能反噬主链路。
    """
    calls = 0
    carried = 0
    peak = 0
    output = 0
    cache_read = 0
    for msg in messages:
        if getattr(msg, "role", None) != "assistant":
            continue
        usage = getattr(msg, "usage", None)
        if not usage:
            continue
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        calls += 1
        carried += inp
        peak = max(peak, inp)
        output += int(getattr(usage, "output_tokens", 0) or 0)
        cache_read += int(getattr(usage, "cache_input_tokens", 0) or 0)
    if tree:
        # 树是权威：次数与总量都以它为准，消息侧只留 peak（树只累加，不记单次极值）。
        calls = int(tree.get("model_calls", 0) or 0)
        carried = int(tree.get("input_tokens", 0) or 0)
        output = int(tree.get("output_tokens", 0) or 0)
        cache_read = int(tree.get("cache_read_tokens", 0) or 0)
    rate = round(cache_read / carried, 4) if carried else 0.0
    return UsageSummary(
        model_calls=calls,
        carried_input_tokens=carried,
        peak_input_tokens=peak,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_hit_rate=rate,
    )
