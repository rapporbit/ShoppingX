"""工具结果安全的确定性测试（不依赖真实 LLM）：超长结果截断 + 同工具刷屏的循环检测。

（原 test_fork_safety.py。A4 删子 Agent 后，fork 深度上限与派发容错两组测试随之删除。）
"""

from app.harness.loop_detector import LoopDetector
from app.harness.truncation import MAX_TOOL_RESULT_TOKENS, truncate_tool_result
from app.utils.tokens import count_tokens


# ---------- 结果截断 ----------
def test_truncate_short_passthrough() -> None:
    assert truncate_tool_result("短结果") == "短结果"


def test_truncate_long_capped_with_hint() -> None:
    # 造一段明显超过 MAX_TOOL_RESULT_TOKENS 的真实文本（单字符串会被 BPE 过度合并，不可控）。
    long_text = "商品名 价格 平台 评分 描述 " * 4000
    assert count_tokens(long_text) > MAX_TOOL_RESULT_TOKENS
    out = truncate_tool_result(long_text)
    assert count_tokens(out) <= MAX_TOOL_RESULT_TOKENS + 50  # 按 token 预算截到上限内
    assert len(out) < len(long_text)
    assert "已截断" in out


# ---------- 循环检测 ----------
def test_loop_detector_triggers_at_threshold() -> None:
    det = LoopDetector(window=6, threshold=4)
    triggered = [det.record("item_search") for _ in range(4)]
    assert triggered[:3] == [False, False, False]
    assert triggered[3] is True
    assert "item_search" in det.nudge_message("item_search")


def test_loop_detector_mixed_calls_no_false_trigger() -> None:
    det = LoopDetector(window=6, threshold=4)
    # 交替调用不同工具，不应触发。
    assert not any(det.record(name) for name in ["a", "b", "a", "b", "a", "b"])


def test_loop_detector_progressed_calls_dont_count() -> None:
    """产出性重试不算打转：相机 bad case 里 4 次检索有 2 次带回新机身，不该弹「仍无进展」。"""
    det = LoopDetector(window=6, threshold=4)
    # 2 次有进展 + 2 次空转：非进展计数只有 2，不触发。
    assert det.record("item_search", progressed=True) is False
    assert det.record("item_search") is False
    assert det.record("item_search", progressed=True) is False
    assert det.record("item_search") is False
    # 再来 2 次空转（窗口 6 内非进展达到 4）→ 触发。
    assert det.record("item_search") is False
    assert det.record("item_search") is True
