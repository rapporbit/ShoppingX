"""``app.agent.invoke``：一次性 LLM 调用的三件事（消息归一 / 流式取尾 / 结构化解壳）。

这一层薄，但每条都对应一个**静默失效**：取错 chunk 拿到残缺前缀、结构化多包一层壳导致
业务字段全默认值——都不报错，只是产出恒定为空。所以断言一律落在业务字段上，不看形状。
"""

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from app.agent.invoke import _unwrap_structured, call_structured, call_text, to_msgs


class _Plan(BaseModel):
    category: str = ""
    keywords: list[str] = Field(default_factory=list)


class _Wrapper(BaseModel):
    """字段名恰好撞上供应商的壳键名——这种 schema 不该被下钻。"""

    parameters: dict = Field(default_factory=dict)


class _FakeModel:
    model = "fake"

    def __init__(self, content: Any = None, text: str = "") -> None:
        self._content = content
        self._text = text

    async def generate_structured_output(self, _messages: Any, _schema: Any, **_kw: Any) -> Any:
        return SimpleNamespace(content=self._content, usage=None)

    async def __call__(self, _messages: Any, **_kw: Any) -> Any:
        pieces = self._text

        async def _stream() -> Any:
            # 累积语义：每个 chunk 是当前完整前缀，最后一个才是完整回答。
            for i in range(1, len(pieces) + 1):
                yield SimpleNamespace(
                    content=[{"type": "text", "text": pieces[:i]}],
                    usage=None,
                    is_last=i == len(pieces),
                )

        return _stream()


# ---------- 消息归一 ----------
def test_to_msgs_accepts_three_shapes() -> None:
    assert [m.role for m in to_msgs("hi")] == ["user"]
    assert [m.role for m in to_msgs([("system", "s"), ("user", "u")])] == ["system", "user"]
    msgs = to_msgs([("system", "s")])
    assert to_msgs(msgs) is not msgs and to_msgs(msgs)[0].role == "system"  # Msg 原样透传


# ---------- 结构化解壳（L7 实测：DashScope 上的 deepseek-v4-flash） ----------
def test_unwraps_parameters_shell() -> None:
    """forced 策略下多包的 ``parameters`` 壳要剥掉，否则字段全落默认值且不报错。"""
    got = _unwrap_structured({"parameters": {"category": "背包", "keywords": ["bag"]}}, _Plan)
    assert got["category"] == "背包"


def test_unwraps_output_shell() -> None:
    """回落 auto 策略时壳键名变成 ``output``——所以判据不能写死键名。"""
    got = _unwrap_structured({"output": {"category": "沙发"}}, _Plan)
    assert got["category"] == "沙发"


def test_keeps_well_formed_content() -> None:
    payload = {"category": "杯子", "keywords": ["mug"]}
    assert _unwrap_structured(payload, _Plan) is payload


def test_does_not_unwrap_when_key_is_a_schema_field() -> None:
    """单键 + 值是 dict，但键名正是 schema 自己的字段 → 那是真结果，不许下钻。"""
    payload = {"parameters": {"a": 1}}
    assert _unwrap_structured(payload, _Wrapper) is payload


def test_unwrap_survives_garbage() -> None:
    """壳里不是 dict / 多键 / 空 —— 一律原样返回，交给 model_validate 去报错。"""
    assert _unwrap_structured({"output": "not a dict"}, _Plan) == {"output": "not a dict"}
    assert _unwrap_structured({"a": {}, "b": {}}, _Plan) == {"a": {}, "b": {}}
    assert _unwrap_structured({}, _Plan) == {}


async def test_call_structured_unwraps_end_to_end() -> None:
    plan = await call_structured(
        _FakeModel(content={"parameters": {"category": "背包"}}), "q", _Plan
    )
    assert plan.category == "背包"  # 空壳（全默认值）就是这条测试要挡的假绿


async def test_call_structured_prefers_auto_then_falls_back_to_ladder() -> None:
    """auto 优先：forced 在 DashScope/deepseek 上只回存根（合法 JSON、全默认值 → 假绿），
    显式 auto 才拿得到整张表；auto 没产出（模型没调工具）时再回落框架默认梯。"""
    from agentscope.exception import StructuredOutputError

    calls: list[Any] = []

    class _Model(_FakeModel):
        def __init__(self, fail_auto: bool) -> None:
            super().__init__()
            self._fail_auto = fail_auto

        async def generate_structured_output(self, _m: Any, _s: Any, **kw: Any) -> Any:
            calls.append(kw.get("tool_choice"))
            if kw.get("tool_choice") is not None:
                if self._fail_auto:
                    raise StructuredOutputError("no tool call")
                return SimpleNamespace(content={"category": "auto 给的"}, usage=None)
            return SimpleNamespace(content={"category": "梯子给的"}, usage=None)

    plan = await call_structured(_Model(fail_auto=False), "q", _Plan)
    assert plan.category == "auto 给的"
    assert len(calls) == 1 and getattr(calls[0], "mode", None) == "auto"

    calls.clear()
    plan = await call_structured(_Model(fail_auto=True), "q", _Plan)
    assert plan.category == "梯子给的"
    assert [getattr(c, "mode", None) for c in calls] == ["auto", None]


# ---------- 流式取尾 ----------
async def test_call_text_takes_last_chunk() -> None:
    """chunk 是累积快照，取最后一个；取第一个只会拿到一个字。"""
    assert await call_text(_FakeModel(text="到手价"), "q") == "到手价"


async def test_call_text_handles_non_streaming_model() -> None:
    class _Blocking:
        model = "fake"

        async def __call__(self, _messages: Any, **_kw: Any) -> Any:
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}], usage=None)

    assert await call_text(_Blocking(), "q") == "ok"
