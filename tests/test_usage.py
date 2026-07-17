"""usage.summarize_usage：从一轮 messages 聚合 token 用量（确定性，无 LLM）。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.usage import summarize_usage


def _ai(input_tokens: int, output_tokens: int, cache_read: int = 0) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
    )


def test_aggregates_across_model_calls() -> None:
    msgs = [
        HumanMessage(content="q"),
        _ai(1000, 200, cache_read=800),
        ToolMessage(content="r", tool_call_id="c1", name="item_search"),
        _ai(3000, 150, cache_read=2400),  # peak 在这条
    ]
    u = summarize_usage(msgs)
    assert u.model_calls == 2
    assert u.carried_input_tokens == 4000
    assert u.peak_input_tokens == 3000
    assert u.output_tokens == 350
    assert u.cache_read_tokens == 3200
    assert u.cache_hit_rate == round(3200 / 4000, 4)  # 0.8


def test_no_usage_metadata_is_zero() -> None:
    """没有任何带 usage 的 AIMessage 时全 0、命中率 0、不抛。"""
    msgs = [HumanMessage(content="q"), AIMessage(content="hi")]
    u = summarize_usage(msgs)
    assert u.model_calls == 0
    assert u.carried_input_tokens == 0
    assert u.cache_hit_rate == 0.0


def test_missing_cache_read_counts_zero() -> None:
    """供应商没报 cached_tokens（缺 input_token_details）时命中率为 0，如实反映。"""
    ai = AIMessage(
        content="",
        usage_metadata={"input_tokens": 500, "output_tokens": 50, "total_tokens": 550},
    )
    u = summarize_usage([ai])
    assert u.carried_input_tokens == 500
    assert u.cache_read_tokens == 0
    assert u.cache_hit_rate == 0.0


def test_as_dict_roundtrip() -> None:
    u = summarize_usage([_ai(100, 10, cache_read=50)])
    d = u.as_dict()
    assert d["carried_input_tokens"] == 100
    assert d["cache_hit_rate"] == 0.5
