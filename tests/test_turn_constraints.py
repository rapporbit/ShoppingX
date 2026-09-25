"""本轮约束 P_t（无状态）：构造规则 + planner 的上文渲染 / 预算出处 / 收货国第 2 层 + 原话落盘。

P_t 不再跨轮合并（2026-09-25），跨轮只存用户原话。这里钉住三件确定性的事：
① ``TurnConstraints.build`` 的词表形态；② planner 怎么用前几轮原话（上文、预算币种、收货国）；
③ orchestrator 怎么读回原话（坏数据按空）。「LLM 能否从原话里把旧约束重新抽出来」不在这里测。
"""

from agentscope.state import AgentState

from app.agent.orchestrator import PRIOR_QUERIES_KEY, _prior_queries
from app.api.context import set_prior_queries
from app.api.run_state import reset_run_state
from app.memory.turn_constraints import TurnConstraints
from app.tools.planner import (
    _render_prior_context,
    budget_source,
    resolve_budget_currency,
    resolve_dest_country_layered,
)
from app.utils.thread_ctx import thread_scope


def test_build_normalizes_and_dedups_across_buckets() -> None:
    pt = TurnConstraints.build(
        category="背包",
        budget_usd=42.0,
        exclude=["Plastic", "plastic", " "],
        avoid=["plastic", "flashy"],
        prefer=["flashy", "canvas"],
    )
    assert pt.exclude_terms == ["plastic"]
    assert pt.avoid_terms == ["flashy"]  # 已进硬淘汰的不再重复进软桶
    assert pt.prefer_terms == ["canvas"]  # 同理，按「硬 > 软 > 加分」只留一处
    assert pt.dislike_terms() == ["plastic"] and pt.like_terms() == ["canvas"]


def test_empty_state() -> None:
    assert TurnConstraints().is_empty()
    assert not TurnConstraints.build(budget_usd=10.0).is_empty()


def test_render_prior_context_first_turn_is_empty() -> None:
    assert _render_prior_context([]) == ""


def test_render_prior_context_lists_queries_in_order() -> None:
    text = _render_prior_context(["想买背包，不要塑料", "预算 80 美元"])
    assert text.index("1. 想买背包，不要塑料") < text.index("2. 预算 80 美元")
    assert text.rstrip().endswith("【本轮用户原话】")


def test_budget_source_prefers_newest_digit_match() -> None:
    utts = ["预算 80 美元", "不要皮革的", "预算改成 80 块"]
    assert budget_source(80, utts) == "预算改成 80 块"
    assert budget_source(80, utts[:2]) == "预算 80 美元"


def test_budget_source_currency_follows_the_source_turn() -> None:
    """旧版坑：前几轮说的 80 美元，按本轮原话（没提币种）解析会落默认 CNY，缩水 7 倍。"""
    src = budget_source(80, ["预算 80 美元", "换一个颜色"])
    assert src == "预算 80 美元"  # 「换一个」里的「一」不能抢先
    assert resolve_budget_currency(src)[0] == "USD"


def test_budget_source_rejects_made_up_amount() -> None:
    assert budget_source(500, ["预算 80 美元", "不要皮革的"]) is None


def test_prior_queries_from_state_tolerates_bad_shapes() -> None:
    assert _prior_queries(None) == []
    st = AgentState()
    st.middle_context[PRIOR_QUERIES_KEY] = ["a", "", 3, "b"]
    assert _prior_queries(st) == ["a", "b"]
    st.middle_context[PRIOR_QUERIES_KEY] = "not a list"
    assert _prior_queries(st) == []


async def test_dest_country_layer2_reads_prior_queries(tmp_path) -> None:
    with thread_scope("t-prior", tmp_path):
        set_prior_queries(["买个背包，寄到日本", "再便宜点"])
        assert await resolve_dest_country_layered("换个颜色") == ("JP", False)
        # 本轮明示压过前几轮
        assert (await resolve_dest_country_layered("改寄到英国"))[0] == "GB"
    reset_run_state(tmp_path)
