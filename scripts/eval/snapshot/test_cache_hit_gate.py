"""快照评测（真 LLM）：前缀缓存命中率没掉下来（D5 的 CI 断言）。

**为什么断言的是命中率而不是「cached tokens > 0」**：后者永远绿。真正会出事的是「某处改动把
system 段或历史前缀打断了」——2026-07 的注入块塌方就是这样，命中率从 67.5% 掉到 31.6%，功能全
正常、没有任何测试变红，只有账单变贵、延迟变长。一个恒为真的断言守不住这种退化。

**为什么是三遍中位而不是单遍点值**：A0-3 基线实测同一条 query、同一工具序列，两遍的命中率是
0.600 和 0.852——网关侧噪声就有 ±0.13。拿单遍点值当门禁，绿红全看运气。三遍取中位把噪声压下去，
门槛再留一个噪声幅度的余量，只抓真退化（前缀断裂那种会掉到 0.3 以下，离门槛很远）。

跑法（真 LLM，Qdrant 隧道见 conftest 模块头；约 3 × 25s，成本约 $0.006）：
    uv run pytest scripts/eval/snapshot/test_cache_hit_gate.py -q -s
"""

import statistics
from typing import Any

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.asyncio(loop_scope="session")]

#: 与 A0-3 基线同一条 query（docs/plans/baseline-artifacts/latency_a0_3_head.json）。
QUERY = "想买个通勤双肩包，预算 300 以内，要能装 15 寸笔记本"

#: 跑几遍取中位。三遍是噪声与成本的折中——两遍的「中位」等于平均、抗不住一次离群。
RUNS = 3

#: A0-3 实测三遍命中率中位。门槛不直接用它，见下。
BASELINE_MEDIAN = 0.674

#: 容差：按实测噪声幅度（0.600 vs 0.852 → ±0.13）留一个身位。
TOLERANCE = 0.12


async def test_prefix_cache_hit_rate_not_below_baseline(snap_run: Any, needs_qdrant: None) -> None:
    rates: list[float] = []
    for _ in range(RUNS):
        result = await snap_run(QUERY)
        rate = float(getattr(result, "cache_hit_rate", 0.0) or 0.0)
        rates.append(rate)

    median = statistics.median(rates)
    floor = BASELINE_MEDIAN - TOLERANCE
    shown = [round(r, 3) for r in rates]
    print(f"\n缓存命中率 {RUNS} 遍：{shown} 中位 {median:.3f}（门槛 {floor:.3f}）")

    # 全 0 通常不是「缓存没命中」而是「这批调用压根没报 cache_read」——供应商换了字段名、
    # 或本机走了不带缓存的端点。当成失败比当成退化更有用：它说明这条门禁此刻什么都没在守。
    assert any(r > 0 for r in rates), "三遍都没报出 cache_read，先确认端点与 usage 字段"
    assert median >= floor, (
        f"缓存命中率中位 {median:.3f} 低于门槛 {floor:.3f}（A0-3 基线 {BASELINE_MEDIAN}）"
    )
