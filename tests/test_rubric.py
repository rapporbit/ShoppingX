"""M11 验收：Rubric 评测的确定性部分（不依赖真实 LLM）。

judge 调用（生成细则 / 打分）是 LLM 行为，不进单测；这里只锁住**纯函数地基**——一旦地基错了，
judge 打得再准，聚合出来的分也是错的：
- ``aggregate``：P0 一票否决（任一 fail → total=0）、P1 每违规扣分、P2 均分归一。
- ``extract_tool_calls`` / ``render_trajectory``：从 messages 还原父 loop 工具轨迹，供 judge 评 P1。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.eval.rubric import (
    P1_PENALTY,
    CriterionScore,
    Rubric,
    RubricCriterion,
    aggregate,
    extract_tool_calls,
    render_trajectory,
)


def _score(
    tier: str, *, passed: bool | None = None, score: int | None = None, dim: str = "d"
) -> CriterionScore:
    return CriterionScore(id=f"{tier}-x", tier=tier, dimension=dim, passed=passed, score=score)


# ---------- aggregate：P0 闸 ----------
def test_p0_failure_forces_overall_fail_and_zero() -> None:
    scores = [
        _score("P0", passed=False, dim="超预算"),
        _score("P2", score=5),  # 即便 P2 满分
    ]
    res = aggregate("q", Rubric(), scores)
    assert res.overall_pass is False
    assert res.total == 0.0
    assert res.is_high_score is False
    assert res.p0_failures == ["超预算"]


def test_all_pass_full_quality_is_hundred() -> None:
    scores = [_score("P0", passed=True), _score("P2", score=5), _score("P2", score=5)]
    res = aggregate("q", Rubric(), scores)
    assert res.overall_pass is True
    assert res.total == 100.0
    assert res.p2_avg == 5.0
    assert res.is_high_score is True


# ---------- aggregate：P1 扣分 + P2 归一 ----------
def test_p1_violation_penalizes_from_p2_baseline() -> None:
    # P2 均分 4/5 → 80 分基线；一条 P1 违规扣 P1_PENALTY。
    scores = [
        _score("P0", passed=True),
        _score("P1", passed=False, dim="没收尾"),
        _score("P2", score=4),
        _score("P2", score=4),
    ]
    res = aggregate("q", Rubric(), scores)
    assert res.p2_avg == 4.0
    assert res.total == 80.0 - P1_PENALTY
    assert res.p1_violations == ["没收尾"]
    assert res.overall_pass is True  # P1 只扣分，不否决


def test_penalty_floors_at_zero() -> None:
    scores = [_score("P0", passed=True)] + [_score("P1", passed=False) for _ in range(20)]
    res = aggregate("q", Rubric(), scores)
    assert res.total == 0.0  # 扣到负数被夹到 0
    assert res.overall_pass is True  # 仍非红线失败


def test_no_p2_defaults_to_full_baseline() -> None:
    # 闲聊兜底类可能没有 P2 维度，质量基线给满，仅受 P1 影响。
    scores = [_score("P0", passed=True), _score("P1", passed=True)]
    res = aggregate("q", Rubric(), scores)
    assert res.total == 100.0


# ---------- 轨迹抽取 ----------
def test_extract_tool_calls_in_order() -> None:
    messages = [
        HumanMessage(content="买个旅行包"),
        AIMessage(
            content="",
            tool_calls=[{"name": "planner", "args": {"intent": "买包"}, "id": "1"}],
        ),
        ToolMessage(content="{...}", tool_call_id="1"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "item_search", "args": {"query": "旅行包", "platform": "all"}, "id": "2"}
            ],
        ),
        AIMessage(content="给你清单"),  # 终结回复，无 tool_calls
    ]
    calls = extract_tool_calls(messages)
    assert [c["name"] for c in calls] == ["planner", "item_search"]
    rendered = render_trajectory(calls)
    assert "1. planner(" in rendered
    assert "2. item_search(" in rendered


def test_render_empty_trajectory() -> None:
    assert "未调用任何工具" in render_trajectory([])


def test_trajectory_drops_id_args() -> None:
    # 轨迹脱敏：item_ids 这类含 id 的入参不渲染，避免 judge 误判「泄露内部 id」。
    calls = [{"name": "item_picker", "args": {"item_ids": ["shp_001", "amz_02"], "top_k": 3}}]
    rendered = render_trajectory(calls)
    assert "shp_001" not in rendered
    assert "item_ids" not in rendered
    assert "top_k=3" in rendered  # 非 id 参数保留


# ---------- rubric 缓存（回归对照固定尺子）----------
async def test_generate_rubric_uses_cache_without_judge(tmp_path: Path, monkeypatch: Any) -> None:
    """缓存命中时直接复用，不触发 judge 调用——这是回归对照「尺子刻度固定」的地基。"""
    from app.eval import rubric as R

    # 预置一条缓存文件（key 与 _rubric_cache_key 一致）。
    query, constraints, intent = "买个旅行包", {"budget_usd": 50}, "shopping"
    key = R._rubric_cache_key(query, constraints, intent)
    cached = Rubric(criteria=[RubricCriterion(tier="P0", dimension="预算", criterion="超50即fail")])
    (tmp_path / f"{key}.json").write_text(cached.model_dump_json(), encoding="utf-8")

    # judge 一旦被调用就炸——命中缓存就不该走到这里。
    def _boom() -> None:
        raise AssertionError("命中缓存却仍调用了 judge")

    monkeypatch.setattr("app.agent.llm.get_judge_llm", _boom)

    got = await R.generate_rubric(query, constraints, intent, use_cache=True, cache_dir=tmp_path)
    assert [c.dimension for c in got.criteria] == ["预算"]


def test_cache_key_changes_with_prompt(monkeypatch: Any) -> None:
    """生成 prompt 一改，缓存 key 即变（旧缓存自动失效，尺子定义变了就重建）。"""
    from app.eval import rubric as R

    k1 = R._rubric_cache_key("q", {}, "shopping")
    monkeypatch.setattr(R, "_GEN_PROMPT", R._GEN_PROMPT + "（改了纪律）")
    k2 = R._rubric_cache_key("q", {}, "shopping")
    assert k1 != k2


# ---------- Rubric 分数注入 Langfuse Score（refdocs 16-3 §2.4）----------
class _FakeLangfuseClient:
    """记下 create_score 的调用参数；不发网络。"""

    def __init__(self) -> None:
        self.scores: list[dict[str, Any]] = []

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)


def _result_with_failures() -> Any:
    """一条 P0 破 + P1 违规的评测结论，用来验证 comment 里能看出「为什么扣分」。"""
    from app.eval.rubric import RubricResult

    return RubricResult(
        query="买个旅行包",
        overall_pass=False,
        total=0.0,
        is_high_score=False,
        p0_failures=["预算红线"],
        p1_violations=["未调 shipping_calc"],
        p2_avg=3.2,
        scores=[
            CriterionScore(
                id="P0-1",
                tier="P0",
                dimension="预算红线",
                passed=False,
                rationale="主体商品 $420 超出 $300 预算",
            ),
            CriterionScore(
                id="P1-2",
                tier="P1",
                dimension="未调 shipping_calc",
                passed=False,
                rationale="用户要到手价却没算运费",
            ),
            CriterionScore(
                id="P1-1",
                tier="P1",
                dimension="有收尾清单",
                passed=True,
                rationale="通过项的依据是噪声",
            ),
        ],
    )


def test_rubric_comment_surfaces_failed_dimensions_and_rationale() -> None:
    """comment 是「5 分钟定位 badcase」SOP 的 Step 2——要一眼看出哪个维度破了、judge 依据是什么。"""
    from app.agent.tracing import _rubric_comment

    comment = _rubric_comment(_result_with_failures())
    assert "P0 破: 预算红线" in comment
    assert "P1 违规: 未调 shipping_calc" in comment
    assert "超出 $300 预算" in comment  # 带上判定依据，而不是只报个计数
    assert "通过项的依据是噪声" not in comment  # 只摘失败项，别把 comment 占满


def test_record_rubric_scores_noop_without_trace_id(monkeypatch: Any) -> None:
    """未启用观测（trace_id 为 None）时零副作用——evaluate 里那行调用不该拖累纯评测链路。"""
    from app.agent import tracing as T

    fake = _FakeLangfuseClient()
    monkeypatch.setattr(T, "_get_client", lambda: fake)
    T.record_rubric_scores(None, _result_with_failures())
    assert fake.scores == []


def test_record_rubric_scores_emits_three_scores(monkeypatch: Any) -> None:
    from app.agent import tracing as T

    fake = _FakeLangfuseClient()
    monkeypatch.setattr(T, "_get_client", lambda: fake)
    T.record_rubric_scores("trace-abc", _result_with_failures())

    by_name = {s["name"]: s for s in fake.scores}
    assert set(by_name) == {"rubric_total", "rubric_pass", "rubric_p2_avg"}
    assert all(s["trace_id"] == "trace-abc" for s in fake.scores)
    # total 归一到 0-1：Langfuse UI 里按「Score < 0.65」筛低分 trace 就是这个刻度。
    assert by_name["rubric_total"]["value"] == 0.0
    assert by_name["rubric_total"]["data_type"] == "NUMERIC"
    assert "预算红线" in by_name["rubric_total"]["comment"]
    # BOOLEAN 的 value 走 0/1。
    assert by_name["rubric_pass"]["value"] == 0.0
    assert by_name["rubric_pass"]["data_type"] == "BOOLEAN"
    assert by_name["rubric_p2_avg"]["value"] == 3.2


def test_record_rubric_scores_swallows_client_error(monkeypatch: Any) -> None:
    """观测层故障绝不能把评测结论带走——异常吞掉，evaluate 照常返回。"""
    from app.agent import tracing as T

    class _Boom:
        def create_score(self, **_: Any) -> None:
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(T, "_get_client", lambda: _Boom())
    T.record_rubric_scores("trace-abc", _result_with_failures())  # 不抛即通过


async def test_evaluate_passes_trace_id_from_run_result(monkeypatch: Any) -> None:
    """锁住数据流：run_agent 返回的 trace_id → evaluate → record_rubric_scores。

    这条线断掉的表现是「trace 有、score 没有」且全程零报错——最难查的那类故障，值得一个测试盯着。
    """
    from app.eval import rubric as R

    seen: dict[str, Any] = {}
    monkeypatch.setattr(R, "record_rubric_scores", lambda tid, res: seen.update(tid=tid, res=res))

    async def _fake_gen(*_: Any, **__: Any) -> Rubric:
        return Rubric(criteria=[RubricCriterion(tier="P2", dimension="覆盖度", criterion="1-5")])

    async def _fake_score(*_: Any, **__: Any) -> list[CriterionScore]:
        return [CriterionScore(id="P2-1", tier="P2", dimension="覆盖度", score=4)]

    monkeypatch.setattr(R, "generate_rubric", _fake_gen)
    monkeypatch.setattr(R, "score_against_rubric", _fake_score)

    run_result = {"trace_id": "trace-xyz", "final_text": "清单", "items": [], "messages": []}
    result = await R.evaluate("买个旅行包", run_result)

    assert seen["tid"] == "trace-xyz"
    assert seen["res"] is result
