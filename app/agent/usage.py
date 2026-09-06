"""按轮聚合 LLM token 用量：携带上下文大小 + 缓存命中率（测量驱动 L3 决策）。

**为什么要测这个**：主模型是 1M 窗口，所以「溢出窗口」不会发生——真正要盯的是两件事：
  1. **上下文越长模型越笨**（注意力稀释 / lost-in-the-middle），即便不溢出也会掉召回质量；
  2. **成本**：每轮都重发压缩后的历史，input 计费随轮数累积。

本模块从「一轮跑完的 messages」里把每次模型调用的 ``usage_metadata`` 聚合出来：
  - ``carried_input_tokens``：本轮各次模型调用发出的 input token 之和（= 压缩后真实携带量）。
  - ``peak_input_tokens``：单次调用的最大 input（≈ 本轮上下文峰值，最贴近「会不会变笨」）。
  - ``cache_read_tokens`` / ``cache_hit_rate``：验证 M6 的 cache-breakpoint 是否真命中（应偏高）。

这些数发给 Langfuse（score）+ 日志，作为「是否 / 何时上 L3 摘要」的**判据闸门**，而不是预先建 L3。

用量来自 ``Msg.usage``：AgentScope 在 ``OpenAIChatModel`` 里已把响应的
``prompt_tokens_details.cached_tokens`` 映射进 ``cache_input_tokens``，不用自己解析。供应商不报
缓存时该项为 0——命中率如实反映「网关到底报没报缓存」，不夸大。
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


def summarize_usage(messages: Sequence[object]) -> UsageSummary:
    """从一轮 ``list[Msg]`` 聚合 token 用量。

    字段口径与迁移前（LangChain 版读 ``usage_metadata``）**逐字一致**，这样两条链路的用量表
    能直接对照——否则「迁移后 token 涨了」这种结论根本没法判是真涨了还是换了口径。

    只认带 ``usage`` 的 assistant 消息（一次模型调用一条）。任一字段缺失按 0，绝不抛——
    观测是附属品，不能反噬主链路。
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
    rate = round(cache_read / carried, 4) if carried else 0.0
    return UsageSummary(
        model_calls=calls,
        carried_input_tokens=carried,
        peak_input_tokens=peak,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_hit_rate=rate,
    )
