"""提示词 A/B 的判读口径：按版本聚合 Rubric 结果，给「达标扩桶」的机制位（批 4 / 18-3）。

**四个指标一起看，缺一不可**（手册 §10 的判据）：

1. ``p0_fail_rate`` —— 业务红线失败率。**一票否决项**：新版本只要把 P0 做差，其余再好也不放量。
2. ``p2_avg`` —— 质量分均值（1-5）。要的是「不退化」，不是「必须更好」：很多提示词改动的收益在
   延迟与成本上，质量持平就该放行。
3. ``avg_rounds`` —— 平均模型轮数。提示词一改，最先变的往往是它（多绕一圈 / 少绕一圈）。
4. ``avg_tokens`` —— 平均 token。与轮数一起构成代价面：质量持平但 token 涨三成，那不是改进。

**不做自动放量**（刻意的）：本模块只回答「达标没有、下一档该是多少」，改 ``.env`` 的手仍是人。
自动放量要成立，前提是样本量足够到能扛住 judge 本身的抖动——本仓 judge 实测存在 0↔100 对翻的
单样本抖动（见记忆 rubric-judge-calibration-pitfalls），15~20 条种子集远够不着那个前提。机制位
先摆好，判断权留给人。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.utils.env import env_float, env_int, env_str

#: 记录里没有版本号时的归属（整轮缓存命中的轮次没跑过任何提示词，聚合时单列一档）。
UNKNOWN_VERSION = "(未记版本)"


@dataclass(frozen=True)
class VariantStats:
    """一个版本在一批评测记录上的表现。"""

    version: str
    n: int  # 参与统计的条数（跑失败 / 超时的不算）
    errored: int  # 评测失败条数（基建问题，不进指标但要看得见）
    p0_fail_rate: float
    p2_avg: float
    avg_rounds: float
    avg_tokens: float
    avg_total: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n": self.n,
            "errored": self.errored,
            "p0_fail_rate": round(self.p0_fail_rate, 4),
            "p2_avg": round(self.p2_avg, 3),
            "avg_rounds": round(self.avg_rounds, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_total": round(self.avg_total, 2),
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, VariantStats]:
    """把 ``rubric_report.json`` 的记录按 ``prompt_version`` 聚合。

    可以喂多份报告的记录拼起来（对照组一份、实验组一份）——判读的单位是版本，不是文件。
    ``ok=False`` 的条目只计入 ``errored``：那是超时 / 调用炸了，把它当成「P0 失败」会让基建抖动
    伪装成提示词退化。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        version = str(rec.get("prompt_version") or "") or UNKNOWN_VERSION
        groups.setdefault(version, []).append(rec)

    out: dict[str, VariantStats] = {}
    for version, rows in groups.items():
        done = [r for r in rows if r.get("ok")]
        errored = len(rows) - len(done)
        results = [r["result"] for r in done]
        fails = [1.0 if not r["overall_pass"] else 0.0 for r in results]
        tokens = [float((r.get("tokens") or {}).get("total") or 0) for r in done]
        rounds = [float(r.get("model_calls") or 0) for r in done]
        out[version] = VariantStats(
            version=version,
            n=len(done),
            errored=errored,
            p0_fail_rate=_mean(fails),
            p2_avg=_mean([float(r.get("p2_avg") or 0.0) for r in results]),
            # 轮数 / token 缺失的记录（旧报告、或缓存命中）按 0 参与均值会把均值拉低成假象，
            # 故只对**报了数**的那些取均值。
            avg_rounds=_mean([v for v in rounds if v > 0]),
            avg_tokens=_mean([v for v in tokens if v > 0]),
            avg_total=_mean([float(r.get("total") or 0.0) for r in results]),
        )
    return out


@dataclass(frozen=True)
class RolloutPolicy:
    """「达标扩桶」的判据与档位阶梯，全部配置驱动（``.env``）。"""

    min_samples: int  # 两组各自至少多少条才作数
    max_p0_regression: float  # P0 失败率允许比对照组高多少（0 = 一点都不许）
    min_p2_delta: float  # P2 均分至少要比对照组高多少（0 = 持平即可）
    max_token_ratio: float  # token 均值最多是对照组的几倍
    steps: tuple[int, ...]  # 放量阶梯（百分比）


def policy_from_env() -> RolloutPolicy:
    """从 ``.env`` 读放量策略。默认值刻意保守：P0 零退让、质量持平、token 不许涨过两成。"""
    raw_steps = env_str("PROMPT_AB_ROLLOUT_STEPS", "10,30,50,100")
    steps: list[int] = []
    for chunk in raw_steps.split(","):
        chunk = chunk.strip()
        if chunk.isdigit() and 0 < int(chunk) <= 100:
            steps.append(int(chunk))
    return RolloutPolicy(
        min_samples=env_int("PROMPT_AB_MIN_SAMPLES", 15),
        max_p0_regression=env_float("PROMPT_AB_MAX_P0_REGRESSION", 0.0),
        min_p2_delta=env_float("PROMPT_AB_MIN_P2_DELTA", 0.0),
        max_token_ratio=env_float("PROMPT_AB_MAX_TOKEN_RATIO", 1.2),
        steps=tuple(sorted(set(steps))) or (10, 30, 50, 100),
    )


@dataclass(frozen=True)
class RolloutDecision:
    """一个候选版本的放量结论。``meets=False`` 时 ``suggested_pct`` 恒等于当前比例（原地不动）。"""

    version: str
    meets: bool
    reasons: list[str]  # 逐条判据的结论（达标与否都写，便于贴进报告）
    current_pct: int
    suggested_pct: int


def rollout_decision(
    control: VariantStats,
    candidate: VariantStats,
    current_pct: int,
    policy: RolloutPolicy | None = None,
) -> RolloutDecision:
    """候选版本该不该扩桶、扩到多少。**只算不改**——真正的放量是人去改 ``.env``。"""
    pol = policy or policy_from_env()
    reasons: list[str] = []
    ok = True

    if control.n < pol.min_samples or candidate.n < pol.min_samples:
        ok = False
        reasons.append(
            f"样本不足：对照 {control.n} 条 / 候选 {candidate.n} 条，"
            f"各需 ≥{pol.min_samples}（改 PROMPT_AB_MIN_SAMPLES）"
        )
    p0_delta = candidate.p0_fail_rate - control.p0_fail_rate
    p0_ok = p0_delta <= pol.max_p0_regression + 1e-9
    ok = ok and p0_ok
    reasons.append(
        f"{'✅' if p0_ok else '❌'} P0 失败率 {candidate.p0_fail_rate:.1%} vs 对照 "
        f"{control.p0_fail_rate:.1%}（Δ{p0_delta:+.1%}，上限 {pol.max_p0_regression:+.1%}）"
    )
    p2_delta = candidate.p2_avg - control.p2_avg
    p2_ok = p2_delta >= pol.min_p2_delta - 1e-9
    ok = ok and p2_ok
    reasons.append(
        f"{'✅' if p2_ok else '❌'} P2 均分 {candidate.p2_avg:.2f} vs 对照 "
        f"{control.p2_avg:.2f}（Δ{p2_delta:+.2f}，下限 {pol.min_p2_delta:+.2f}）"
    )
    ratio = candidate.avg_tokens / control.avg_tokens if control.avg_tokens else 0.0
    token_ok = ratio <= pol.max_token_ratio + 1e-9
    ok = ok and token_ok
    reasons.append(
        f"{'✅' if token_ok else '❌'} token 均值 {candidate.avg_tokens:.0f} vs 对照 "
        f"{control.avg_tokens:.0f}（{ratio:.2f}×，上限 {pol.max_token_ratio:.2f}×）"
    )
    # 轮数只报不判：它与 token 高度相关，两处都设闸等于同一件事罚两遍。
    reasons.append(
        f"ℹ️ 轮数均值 {candidate.avg_rounds:.2f} vs 对照 {control.avg_rounds:.2f}"
        f"（Δ{candidate.avg_rounds - control.avg_rounds:+.2f}，只报不判）"
    )

    nxt = next((s for s in pol.steps if s > current_pct), current_pct)
    return RolloutDecision(
        version=candidate.version,
        meets=ok,
        reasons=reasons,
        current_pct=current_pct,
        suggested_pct=nxt if ok else current_pct,
    )
