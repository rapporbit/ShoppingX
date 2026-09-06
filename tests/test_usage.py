"""usage.summarize_usage：从一轮 messages 聚合 token 用量（确定性，无 LLM）。"""

from agentscope.message import Msg, TextBlock
from agentscope.message._base import Usage

from app.agent.usage import summarize_usage


def _ai(input_tokens: int, output_tokens: int, cache_read: int = 0) -> Msg:
    """一次模型调用 = 一条带 usage 的 assistant 消息。"""
    msg = Msg(name="shoppingx", role="assistant", content=[TextBlock(type="text", text="")])
    msg.usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )
    return msg


def _user(text: str) -> Msg:
    return Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])


def test_aggregates_across_model_calls() -> None:
    msgs = [
        _user("q"),
        _ai(1000, 200, cache_read=800),
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
    """没有任何带 usage 的 assistant 消息时全 0、命中率 0、不抛。"""
    msgs = [
        _user("q"),
        Msg(name="shoppingx", role="assistant", content=[TextBlock(type="text", text="hi")]),
    ]
    u = summarize_usage(msgs)
    assert u.model_calls == 0
    assert u.carried_input_tokens == 0
    assert u.cache_hit_rate == 0.0


def test_missing_cache_read_counts_zero() -> None:
    """供应商没报 cached_tokens 时命中率为 0，如实反映（不夸大缓存效果）。"""
    u = summarize_usage([_ai(500, 50)])
    assert u.carried_input_tokens == 500
    assert u.cache_read_tokens == 0
    assert u.cache_hit_rate == 0.0


def test_as_dict_roundtrip() -> None:
    u = summarize_usage([_ai(100, 10, cache_read=50)])
    d = u.as_dict()
    assert d["carried_input_tokens"] == 100
    assert d["cache_hit_rate"] == 0.5


# --- AgentScope 侧（批 0 迁移期与上面的 LangChain 版并存）-----------------------


def _as_ai(input_tokens: int, output_tokens: int, cache_read: int = 0):  # type: ignore[no-untyped-def]
    from agentscope.message import Msg, TextBlock
    from agentscope.message._base import Usage

    return Msg(
        name="assistant",
        role="assistant",
        content=[TextBlock(type="text", text="")],
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
    )


def test_aggregation_matches_pre_migration_shape() -> None:
    """口径与 LangChain 版**逐字段一致**——否则迁移前后的用量表没法对照。"""
    from agentscope.message import Msg, TextBlock

    from app.agent.usage import summarize_usage

    msgs = [
        Msg(name="user", role="user", content=[TextBlock(type="text", text="q")]),
        _as_ai(1000, 200, cache_read=800),
        _as_ai(3000, 150, cache_read=2400),
    ]
    u = summarize_usage(msgs)
    assert u.model_calls == 2
    assert u.carried_input_tokens == 4000
    assert u.peak_input_tokens == 3000
    assert u.output_tokens == 350
    assert u.cache_read_tokens == 3200
    assert u.cache_hit_rate == round(3200 / 4000, 4)


def test_msgs_without_usage_are_zero() -> None:
    """没有 usage 的消息（工具结果 / 用户输入）全 0、不抛。"""
    from agentscope.message import Msg, TextBlock

    from app.agent.usage import summarize_usage

    msgs = [
        Msg(name="user", role="user", content=[TextBlock(type="text", text="q")]),
        Msg(name="assistant", role="assistant", content=[TextBlock(type="text", text="hi")]),
    ]
    u = summarize_usage(msgs)
    assert u.model_calls == 0
    assert u.cache_hit_rate == 0.0
