"""M2 验收：fork 安全四层的确定性测试（不依赖真实 LLM）。

覆盖 ROADMAP M2 验收点：
- 递归 demands 深度超限被拦（① 深度上限）
- 超长结果被截断（③ 结果截断）
- 同工具刷屏触发循环检测（④ 循环检测）
- 子任务异常转字符串、不抛崩溃（task_dispatch 容错）
"""

import pytest

from app.agent.dispatch_tool import _run_worker
from app.agent.fork_guard import (
    MAX_FORK_DEPTH,
    ForkLimitExceeded,
    current_fork_depth,
    enter_fork,
)
from app.harness.loop_detector import LoopDetector
from app.harness.truncation import MAX_TOOL_RESULT_TOKENS, truncate_tool_result
from app.utils.tokens import count_tokens


# ---------- ① 深度上限 ----------
def test_enter_fork_increments_and_restores() -> None:
    assert current_fork_depth() == 0
    with enter_fork() as d1:
        assert d1 == 1
        assert current_fork_depth() == 1
    # 离开作用域还原。
    assert current_fork_depth() == 0


def test_enter_fork_raises_beyond_limit() -> None:
    with enter_fork():  # 到达 MAX_FORK_DEPTH=1（只允许一层 fork）
        assert current_fork_depth() == MAX_FORK_DEPTH
        with pytest.raises(ForkLimitExceeded):
            with enter_fork():
                pass


async def test_dispatch_rejected_at_max_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """处于最大深度时再派发，应返回「拒绝」字符串而非抛异常或调用 LLM。"""
    import app.agent.agents as agents_mod

    async def boom(_kind: str):  # type: ignore[no-untyped-def]
        raise AssertionError("不应构建 worker —— 深度拦截未生效")

    monkeypatch.setattr(agents_mod, "build_worker_agent", boom)
    with enter_fork():  # 深度已达上限（MAX_FORK_DEPTH=1）
        result = await _run_worker("随便什么递归需求", "search")
    assert "[task_dispatch 拒绝]" in result
    assert current_fork_depth() == 0


# ---------- ③ 结果截断 ----------
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


# ---------- ④ 循环检测 ----------
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


# ---------- dispatch_tool 容错 ----------
async def test_dispatch_sub_agent_error_becomes_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker 构建/执行抛异常时，应被兜底转成字符串，不向主 loop 抛。"""
    import app.agent.agents as agents_mod

    async def boom(_kind: str):  # type: ignore[no-untyped-def]
        raise RuntimeError("模拟 worker 故障")

    monkeypatch.setattr(agents_mod, "build_worker_agent", boom)
    result = await _run_worker("demands", "search")
    assert "[task_dispatch 错误]" in result
    assert "RuntimeError" in result
