"""提示词版本化 A/B（批 4 / 18-3）：叠加层不漂、分桶稳定、放量只扩不洗牌、判读口径。

**这些用例真正在问的是：「这套 A/B 会不会自己骗自己」。** 一个测不出东西的实验比没有实验更糟——
它会带着一份看起来正经的报告去改线上提示词。故盯四件事：版本文件与正文**结构上**不可能漂
（1.0.0 逐字等于主文件、覆盖不存在的键当场炸）；桶号跨进程 / 跨重启稳定；改放量比例**不重排**
已分组的人（否则前后两段数据不可比）；聚合把「评测跑挂了」与「P0 破了」分得清楚。
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from app.agent import ab
from app.agent.prompts import (
    BASE_VERSION,
    _load_base_prompts,
    _load_prompts,
    available_versions,
    get_system_prompt,
)
from app.eval.ab_report import (
    RolloutPolicy,
    VariantStats,
    aggregate,
    rollout_decision,
)


@pytest.fixture(autouse=True)
def _clean_ab_env(monkeypatch: Any) -> None:
    """每个用例从「没开实验」起步——否则本机 .env 里配了变体就会让断言随环境飘。"""
    for key in ("PROMPT_VERSION", "PROMPT_AB_VARIANTS", "PROMPT_AB_SALT"):
        monkeypatch.delenv(key, raising=False)
    _load_prompts.cache_clear()


# ────────────────────────── 单一事实源：版本文件是叠加层 ──────────────────────────


def test_base_version_is_byte_identical_to_prompts_yml() -> None:
    """1.0.0 = prompt/prompts.yml 原样。这条断言是「不漂」的守门人。

    版本文件若哪天被改成「正文副本」，主文件一改它就落后一个版本，而线上某个桶会**静默**跑在
    半旧提示词上——没有报错、报告照出。故把「零覆盖」钉死成用例。
    """
    assert _load_prompts(BASE_VERSION) == _load_base_prompts()
    assert BASE_VERSION in available_versions()


def test_override_of_unknown_key_raises(tmp_path: Any, monkeypatch: Any) -> None:
    """覆盖一个基线里没有的键（八成是拼错）必须当场炸，不能静默忽略。"""
    import app.agent.prompts as prompts_mod

    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "9.9.9.yml").write_text(
        "base: prompts.yml\noverrides:\n  sytem_prompt: 拼错了\n", encoding="utf-8"
    )
    monkeypatch.setattr(prompts_mod, "_VERSIONS_DIR", versions_dir)
    _load_prompts.cache_clear()
    with pytest.raises(KeyError, match="不存在的键"):
        _load_prompts("9.9.9")
    _load_prompts.cache_clear()


def test_overlay_merges_one_level_and_keeps_rest(tmp_path: Any, monkeypatch: Any) -> None:
    """只写改动的子键：``sub_agents.search`` 换掉，``sub_agents.trade`` 与其余键原样继承。"""
    import app.agent.prompts as prompts_mod

    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "1.1.0.yml").write_text(
        "base: prompts.yml\noverrides:\n  sub_agents:\n    search: 新的检索员提示词\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompts_mod, "_VERSIONS_DIR", versions_dir)
    _load_prompts.cache_clear()
    base = _load_base_prompts()
    merged = _load_prompts("1.1.0")
    assert merged["sub_agents"]["search"] == "新的检索员提示词"
    assert merged["sub_agents"]["trade"] == base["sub_agents"]["trade"]
    assert merged["system_prompt"] == base["system_prompt"]
    _load_prompts.cache_clear()


def test_version_chain_cycle_is_rejected(tmp_path: Any, monkeypatch: Any) -> None:
    """A→B→A 的 base 链要报错，不能转到栈溢出。"""
    import app.agent.prompts as prompts_mod

    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "2.0.0.yml").write_text("base: 2.1.0\noverrides: {}\n", encoding="utf-8")
    (versions_dir / "2.1.0.yml").write_text("base: 2.0.0\noverrides: {}\n", encoding="utf-8")
    monkeypatch.setattr(prompts_mod, "_VERSIONS_DIR", versions_dir)
    _load_prompts.cache_clear()
    with pytest.raises(ValueError, match="成环"):
        _load_prompts("2.0.0")
    _load_prompts.cache_clear()


# ────────────────────────────────── 分桶 ──────────────────────────────────


def test_bucket_is_stable_across_processes() -> None:
    """桶号必须跨进程一致——用内置 hash() 会因 PYTHONHASHSEED 每进程换一套分组。"""
    same = ab.bucket_of("user-42")
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from app.agent.ab import bucket_of; print(bucket_of('user-42'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(out.stdout.strip()) == same


def test_salt_changes_grouping(monkeypatch: Any) -> None:
    """换盐 = 重新分桶（这是它唯一的用途，也是它唯一该被小心使用的理由）。"""
    before = [ab.bucket_of(f"u{i}") for i in range(30)]
    monkeypatch.setenv("PROMPT_AB_SALT", "round-2")
    after = [ab.bucket_of(f"u{i}") for i in range(30)]
    assert before != after


def test_anonymous_goes_to_control(monkeypatch: Any) -> None:
    """匿名 / 未登录不参与实验：恒走默认版本、桶号是哨兵值（口径见 ab.py 模块 docstring）。"""
    monkeypatch.setenv("PROMPT_AB_VARIANTS", f"{BASE_VERSION}:100")
    for uid in (None, ""):
        got = ab.assign(uid)
        assert got.bucket == ab.ANON_BUCKET
        assert got.in_experiment is False
        assert got.version == ab.default_version()


def test_scaling_up_never_reshuffles_assigned_users(monkeypatch: Any) -> None:
    """**扩桶只搬对照组的人**：10% → 30% 时，原实验组一个不动，新进来的全来自对照组。

    这是「达标扩桶」能与前一段数据接上账的前提。若按「哈希 % 变体数」分配，改一次比例所有人
    重排，扩桶前后就是两个互不相干的实验。
    """
    users = [f"u{i}" for i in range(400)]
    monkeypatch.setenv("PROMPT_AB_VARIANTS", f"{BASE_VERSION}:10")
    at10 = {u for u in users if ab.assign(u).in_experiment}
    monkeypatch.setenv("PROMPT_AB_VARIANTS", f"{BASE_VERSION}:30")
    at30 = {u for u in users if ab.assign(u).in_experiment}
    assert at10 <= at30  # 只增不减、无人被搬出去
    assert len(at30) > len(at10)


def test_bad_variant_config_falls_back_not_crashes(monkeypatch: Any) -> None:
    """配了不存在的版本 / 非法百分比：跳过并告警，绝不把服务打挂，也不静默当成 0%。"""
    monkeypatch.setenv("PROMPT_AB_VARIANTS", "9.9.9:50,1.0.0:abc")
    assert ab.variant_weights() == []
    assert ab.active_version() == BASE_VERSION
    assert get_system_prompt() == get_system_prompt(BASE_VERSION)


def test_variant_weights_cap_at_100(monkeypatch: Any) -> None:
    """累计超过 100% 的条目被忽略，不会让某个桶落进两个变体。"""
    monkeypatch.setenv("PROMPT_AB_VARIANTS", f"{BASE_VERSION}:80,{BASE_VERSION}:40")
    assert ab.variant_weights() == [(BASE_VERSION, 80)]


# ──────────────────────────── 判读口径与「达标扩桶」 ────────────────────────────


def _rec(version: str, *, ok: bool = True, passed: bool = True, p2: float = 4.0,
         calls: int = 5, tokens: int = 10000) -> dict[str, Any]:
    if not ok:
        return {"id": "x", "ok": False, "prompt_version": version, "error": "EvalTimeout: ..."}
    return {
        "id": "x",
        "ok": True,
        "prompt_version": version,
        "model_calls": calls,
        "tokens": {"total": tokens},
        "result": {"overall_pass": passed, "total": 70.0, "p2_avg": p2},
    }


def test_aggregate_separates_infra_failures_from_p0_breaks() -> None:
    """跑挂的条目只进 errored，不许伪装成 P0 退化——否则一次网络抖动就能否掉一个好版本。"""
    stats = aggregate(
        [_rec("1.0.0"), _rec("1.0.0", passed=False), _rec("1.0.0", ok=False)]
    )["1.0.0"]
    assert (stats.n, stats.errored) == (2, 1)
    assert stats.p0_fail_rate == 0.5


def test_aggregate_ignores_missing_cost_fields() -> None:
    """没记 token / 轮数的记录（旧报告、缓存命中）不按 0 参与均值，否则均值被拉出假象。"""
    rec = _rec("1.0.0")
    rec.pop("tokens")
    rec.pop("model_calls")
    stats = aggregate([rec, _rec("1.0.0", calls=7, tokens=20000)])["1.0.0"]
    assert stats.avg_tokens == 20000
    assert stats.avg_rounds == 7


def test_records_without_version_are_bucketed_separately() -> None:
    from app.eval.ab_report import UNKNOWN_VERSION

    stats = aggregate([_rec(""), _rec("1.0.0")])
    assert set(stats) == {UNKNOWN_VERSION, "1.0.0"}


_POLICY = RolloutPolicy(
    min_samples=2, max_p0_regression=0.0, min_p2_delta=0.0, max_token_ratio=1.2,
    steps=(10, 30, 50, 100),
)


def _stats(version: str, p0: float, p2: float, tokens: float, rounds: float = 5.0,
           n: int = 10) -> VariantStats:
    return VariantStats(
        version=version, n=n, errored=0, p0_fail_rate=p0, p2_avg=p2,
        avg_rounds=rounds, avg_tokens=tokens, avg_total=70.0,
    )


def test_rollout_gate_requires_all_three() -> None:
    """P0 一票否决、P2 不许退、token 不许涨过 1.2×——任一不满足就原地不动。"""
    control = _stats("1.0.0", 0.2, 4.0, 10000)
    good = _stats("1.1.0", 0.1, 4.2, 10500)
    assert rollout_decision(control, good, 10, _POLICY).meets is True

    for bad in (
        _stats("1.1.0", 0.3, 4.2, 10000),  # P0 变差
        _stats("1.1.0", 0.2, 3.9, 10000),  # P2 退了
        _stats("1.1.0", 0.1, 4.5, 13000),  # token 涨 30%
    ):
        d = rollout_decision(control, bad, 10, _POLICY)
        assert d.meets is False
        assert d.suggested_pct == d.current_pct  # 未达标 = 原地不动


def test_rollout_needs_enough_samples() -> None:
    """样本不够就不给结论——judge 单样本会 0↔100 对翻，小样本上的「达标」是噪声。"""
    policy = RolloutPolicy(2, 0.0, 0.0, 1.2, (10, 30))
    d = rollout_decision(_stats("1.0.0", 0.0, 4.0, 1e4, n=1), _stats("1.1.0", 0.0, 4.5, 1e4, n=1),
                         10, policy)
    assert d.meets is False
    assert any("样本不足" in r for r in d.reasons)


def test_rollout_climbs_one_step_and_stops_at_top() -> None:
    """达标只往上走**一档**（阶梯配置驱动），到顶就停在 100%，不会算出 >100 的比例。"""
    control = _stats("1.0.0", 0.2, 4.0, 10000)
    cand = _stats("1.1.0", 0.1, 4.2, 10000)
    assert rollout_decision(control, cand, 10, _POLICY).suggested_pct == 30
    assert rollout_decision(control, cand, 30, _POLICY).suggested_pct == 50
    assert rollout_decision(control, cand, 100, _POLICY).suggested_pct == 100


# ─────────────────────── 端到端：桶号真的换掉了提示词 ───────────────────────


def test_user_in_variant_bucket_gets_variant_prompt(tmp_path: Any, monkeypatch: Any) -> None:
    """全链路那一小步：ContextVar 里的 user_id → 桶 → 版本 → ``get_system_prompt`` 的实际字节。

    这是整套机制唯一「真正生效」的接缝：前面几条都对、这里没接上，实验就是空跑。
    """
    import app.agent.prompts as prompts_mod
    from app.utils.thread_ctx import thread_scope

    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "1.0.0.yml").write_text("base: prompts.yml\noverrides: {}\n", encoding="utf-8")
    (versions_dir / "1.1.0.yml").write_text(
        "base: prompts.yml\noverrides:\n  system_prompt: 变体提示词\n", encoding="utf-8"
    )
    monkeypatch.setattr(prompts_mod, "_VERSIONS_DIR", versions_dir)
    _load_prompts.cache_clear()
    monkeypatch.setenv("PROMPT_AB_VARIANTS", "1.1.0:100")  # 全量给候选，桶号无关地生效

    # 用 thread_scope 而不是裸 set_thread_context：后者只 set 不 reset，会把 thread_id /
    # session_dir 泄漏给同进程的后续用例（实测打挂了 test_token_budget 与 test_refine_turn）。
    try:
        with thread_scope("t-ab", tmp_path, user_id="alice"):
            assert ab.assign().version == "1.1.0"
            assert get_system_prompt() == "变体提示词"
    finally:
        _load_prompts.cache_clear()
