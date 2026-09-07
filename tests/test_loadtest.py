"""压测脚本的统计层（`scripts/loadtest.py`）。

只测「数是怎么算出来的」这一层——发请求那部分要真后端，归双进程冒烟管。之所以值得测：压测
脚本给出的数会被直接抄进报告和文档，**算错了不会有任何人发现**，它不崩、不报错，只是给出一个
看着合理的假数字。分位数的边界（空样本 / 单样本 / q=1.0）和「失败请求不进延迟统计」这两件事
最容易写错，所以逐条钉住。
"""

from scripts.loadtest import Attempt, StageReport, percentile, render_table


def _report(concurrency: int, oks: list[float], fails: list[str], wall: float) -> StageReport:
    attempts = [Attempt(True, s) for s in oks] + [Attempt(False, 1.0, r) for r in fails]
    return StageReport(concurrency=concurrency, attempts=attempts, wall=wall)


def test_percentile_is_nearest_rank_not_interpolated() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    # 插值实现会给 3.5（P75）与 3.85（P95）这种「谁也没经历过」的数；最近秩一定落在真实样本上。
    assert percentile(values, 0.5) == 2.0
    assert percentile(values, 0.75) == 3.0
    assert percentile(values, 0.95) == 4.0


def test_percentile_edges_do_not_blow_up() -> None:
    assert percentile([], 0.95) == 0.0  # 一档全失败时会走到这里，不能抛
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0], 0.0) == 1.0  # rank 被夹到 1，不会变成索引 -1


def test_failed_attempts_are_excluded_from_latency() -> None:
    # 失败的那次耗时 1.0s（超时前就断了），若混进统计会把 P95 拉低到看着更快。
    report = _report(5, [10.0, 20.0, 30.0], ["超时未收到终态"], wall=30.0)
    assert report.latency(0.95) == 30.0
    assert report.success_rate == 0.75


def test_throughput_counts_only_successes() -> None:
    report = _report(5, [1.0, 1.0], ["429 背压", "429 背压"], wall=4.0)
    assert report.throughput == 0.5  # 2 个成功 / 4s，而不是 4/4
    assert report.failures() == {"429 背压": 2}


def test_empty_stage_reports_zero_instead_of_dividing_by_zero() -> None:
    empty = StageReport(concurrency=2)
    assert empty.success_rate == 0.0
    assert empty.throughput == 0.0
    assert empty.latency(0.5) == 0.0


def test_render_table_has_one_row_per_stage_and_shows_failure_reasons() -> None:
    table = render_table(
        [
            _report(2, [1.0, 2.0], [], wall=2.0),
            _report(10, [3.0], ["429 背压"], wall=4.0),
        ]
    )
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 3  # 表头 + 两档
    assert "429 背压×1" in table
    assert "—" in rows[1]  # 没有失败的那档留破折号，不留空
