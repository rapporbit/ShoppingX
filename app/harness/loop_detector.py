"""循环检测原语（Feedback / Computational）。

模型可能盯着同一个工具反复刷。滑动窗口里同名工具达到阈值即判定「在打转」，由
``post_tool_call`` 的 ``result_nudges`` Hook 在该工具结果尾部追加提示，推它换思路或收尾。

**不硬停**——硬停会丢掉它本可做的补救，交模型自己收敛更稳。

与 Silent Drift 互补：LoopDetector 抓「重复做同一件事」，漂移检测抓「做不同的事但在偏离」。
"""

from __future__ import annotations

from collections import deque


class LoopDetector:
    """滑动窗口循环检测：最近 ``window`` 次调用里同名工具达到 ``threshold`` 次即判定打转。

    **有进展的调用不计入阈值**（``progressed=True``）：检索类工具每次换词重试若真带回了新候选
    （登记表新增 item_id > 0），那是产出性重试、不是打转——相机 bad case 实测 4 次 item_search
    里第 2、4 次捞到了真机身，旧口径照样弹「仍无进展」，文案是假的还会诱导模型提前收尾。
    纯空转（重复召回同一批 / 被闸拦下 / memo 回放）照常计数。
    """

    def __init__(self, window: int = 6, threshold: int = 4) -> None:
        self.window = window
        self.threshold = threshold
        self._recent: deque[tuple[str, bool]] = deque(maxlen=window)

    def record(self, tool_name: str, *, progressed: bool = False) -> bool:
        """记录一次工具调用，返回 True 表示触发了循环（达到阈值）。"""
        self._recent.append((tool_name, progressed))
        return sum(1 for n, p in self._recent if n == tool_name and not p) >= self.threshold

    def nudge_message(self, tool_name: str) -> str:
        """触发循环时给模型的提示语（让它换思路或收尾，而非硬停）。"""
        return (
            f"你已在短时间内重复调用 {tool_name} {self.threshold} 次仍无进展，"
            f"请检查参数是否合理、换个思路，或调 shopping_summary 收尾。"
        )

    def reset(self) -> None:
        """清空窗口（如开启新一轮任务）。"""
        self._recent.clear()
