"""M2 验收：fork 安全四层的确定性测试（不依赖真实 LLM）。

覆盖 ROADMAP M2 验收点：
- 递归 demands 深度超限被拦（① 深度上限）
- 超长结果被截断（③ 结果截断）
- 同工具刷屏触发循环检测（④ 循环检测）
- 子任务异常转字符串、不抛崩溃（dispatch_tool 容错）
"""

import pytest

from app.agent.dispatch_tool import _ensure_platform_coverage, _run_sub_agent
from app.agent.fork_guard import (
    MAX_FORK_DEPTH,
    ForkLimitExceeded,
    current_fork_depth,
    enter_fork,
)
from app.agent.platform_scope import platform_scope
from app.harness.loop_detector import LoopDetector
from app.harness.truncation import MAX_TOOL_RESULT_TOKENS, truncate_tool_result
from app.utils.clean import PLATFORMS
from app.utils.tokens import count_tokens


# ---------- 平台覆盖（机制兜：parallel_dispatch 按「启用平台」补齐 + 丢弃未启用的）----------
def test_platform_coverage_fills_missing() -> None:
    # 模型只列了启用集合里的 1 个 → 机制补齐到全部启用平台（补齐的用模板克隆）。
    demands = ["在 amazon 上检索：品类=露营厨具，预算≈$70，关键词=cookware"]
    with platform_scope(["amazon", "shein", "walmart"]):
        out = _ensure_platform_coverage(demands)
    covered = {p for p in PLATFORMS if any(p in d.lower() for d in out)}
    assert covered == {"amazon", "shein", "walmart"}
    assert out[0] == demands[0]  # 模型原本写的那条原样保留在前


def test_platform_coverage_drops_disabled_platforms() -> None:
    """未启用的平台**丢弃**：用户没勾它，派出去必然空军（这就是「不 fork 注定空军的平台」）。"""
    demands = [f"在 {p} 上检索：品类=鞋，预算≈$50" for p in PLATFORMS]
    with platform_scope(["amazon"]):
        out = _ensure_platform_coverage(demands)
    assert out == ["在 amazon 上检索：品类=鞋，预算≈$50"]


def test_platform_coverage_falls_back_when_all_dropped() -> None:
    """模型列的平台全没启用 → 不把整批派空：克隆一条「只搜启用平台」的 demand 兜底。"""
    with platform_scope(["amazon"]):
        out = _ensure_platform_coverage(["在 shopee 上检索：品类=鞋，预算≈$50"])
    assert len(out) == 1 and "amazon" in out[0].lower()


def test_platform_coverage_noop_when_already_full() -> None:
    demands = [f"在 {p} 上检索：品类=鞋，预算≈$50" for p in PLATFORMS]
    with platform_scope(list(PLATFORMS)):
        assert _ensure_platform_coverage(demands) == demands


def test_platform_coverage_skips_non_platform_dispatch() -> None:
    # 不是平台检索（没有任何平台名）→ 原样放行，不强塞 5 平台。
    demands = ["爬取这 3 个商品的详情页做对比", "汇总用户评论情感"]
    assert _ensure_platform_coverage(demands) == demands


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


async def test_dispatch_rejected_at_max_depth() -> None:
    """处于最大深度时再 dispatch，应返回「拒绝」字符串而非抛异常或调用 LLM。"""
    with enter_fork():  # 深度已达上限（MAX_FORK_DEPTH=1）
        # tools_provider 故意会爆炸：若深度拦截生效，根本不会触达它。
        def boom() -> list:
            raise AssertionError("不应构建子 Agent —— 深度拦截未生效")

        result = await _run_sub_agent("随便什么递归需求", boom)
    assert "[dispatch_tool 拒绝]" in result
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
async def test_dispatch_sub_agent_error_becomes_string() -> None:
    """子 Agent 构建/执行抛异常时，应被兜底转成字符串，不向主 loop 抛。"""

    def boom() -> list:
        raise RuntimeError("模拟子 Agent 故障")

    result = await _run_sub_agent("demands", boom)
    assert "[dispatch_tool 错误]" in result
    assert "RuntimeError" in result
