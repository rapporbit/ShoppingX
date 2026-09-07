"""``app.agent.invoke``：一次性 LLM 调用的三件事（消息归一 / 流式取尾 / 结构化解壳）。

这一层薄，但每条都对应一个**静默失效**：取错 chunk 拿到残缺前缀、结构化多包一层壳导致
业务字段全默认值——都不报错，只是产出恒定为空。所以断言一律落在业务字段上，不看形状。
"""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator

from app.agent.invoke import (
    EmptyStructuredOutput,
    _unwrap_structured,
    call_structured,
    call_text,
    to_msgs,
)
from app.tools._args import drop_none_values


class _Plan(BaseModel):
    # 与线上 schema 同形：挂了「显式 null 归一为缺席」的 before validator（badcase cdee1d6d）。
    _null_is_absent = model_validator(mode="before")(staticmethod(drop_none_values))

    category: str = ""
    keywords: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)


class _Wrapper(BaseModel):
    """字段名恰好撞上供应商的壳键名——这种 schema 不该被下钻。"""

    parameters: dict = Field(default_factory=dict)


class _FakeModel:
    model = "fake"

    def __init__(
        self, content: Any = None, text: str = "", contents: Sequence[Any] | None = None
    ) -> None:
        # contents = 每次结构化调用依次吐一份（用尽后重复最后一份），用来演「第一次空、
        # 第二次满」这种**跨采样**的行为；calls 记调用次数，断言「真的重采样了」。
        self._contents = list(contents) if contents is not None else [content]
        self._text = text
        self.calls = 0

    async def generate_structured_output(self, _messages: Any, _schema: Any, **_kw: Any) -> Any:
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return SimpleNamespace(content=content, usage=None)

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


# ---------- 空表闸（required_any） ----------
async def test_resamples_once_when_first_sample_is_a_stub() -> None:
    """第一次只回 ``{"tasks": [...]}`` 这种存根 → 重采样，第二次的满表才是结果。

    存根形态照实测抄（DashScope / deepseek-v4-flash 被 forced tool_choice 时的原样返回）：
    它校验得过，业务字段全落默认值，不加这道闸就是一张「看起来合法」的空表。
    """
    model = _FakeModel(contents=[{"tasks": ["recommend"]}, {"category": "背包", "tasks": []}])
    plan = await call_structured(model, "q", _Plan, required_any=("category", "keywords"))
    assert plan.category == "背包"
    assert model.calls == 2  # 真的重采样了，不是把第一次的空表放过去


async def test_raises_when_both_samples_are_stubs() -> None:
    """两次都空 → 抛明确异常（带 schema 名与判据字段），由调用方按各自语义降级。"""
    model = _FakeModel(contents=[{"tasks": ["recommend"]}])
    with pytest.raises(EmptyStructuredOutput) as exc:
        await call_structured(model, "q", _Plan, required_any=("category", "keywords"))
    assert model.calls == 2  # 只重采样一次，不无限重试
    assert exc.value.schema_name == "_Plan"
    assert exc.value.missing == ("category", "keywords")


async def test_without_required_any_behaviour_unchanged() -> None:
    """不传 required_any = 老行为一字不变：存根照收、只调一次模型。

    空结果本就合法的调用点（偏好解析 / 记忆管家「本轮没有可提升的偏好」）靠的就是这条——
    给它们上闸只会把合法的空结果误判成失败。
    """
    model = _FakeModel(contents=[{"tasks": ["recommend"]}])
    plan = await call_structured(model, "q", _Plan)
    assert plan.category == "" and model.calls == 1


async def test_explicit_null_counts_as_present() -> None:
    """模型显式吐 ``category: null`` = 它在认真回答「这项没有」，不是存根，不该重采样。

    这条不能用 ``model_fields_set`` 单独判：``drop_none_values`` 在校验前就把显式 null 的键
    丢了，只看 fields_set 会把「明确说没有」误判成「压根没提」。
    """
    model = _FakeModel(contents=[{"category": None, "keywords": None}])
    plan = await call_structured(model, "q", _Plan, required_any=("category", "keywords"))
    assert plan.category == "" and plan.keywords == [] and model.calls == 1


async def test_after_validator_assignment_does_not_fake_presence() -> None:
    """schema 的 after 校验器自己赋过的字段**不算**模型吐过 —— 否则这道闸自己就是静默失效。

    照 ``PlanOutput`` 的真实形态写：它有两个 ``mode="after"`` 校验器会给 exclude_keywords /
    bundle_slots 赋值，于是一张 ``{"tasks": [...]}`` 的存根，``model_fields_set`` 里凭空多出
    这两个名字。第一版判据用 fields_set，闸门因此恒不触发（实测）。
    """

    class _SelfAssigning(BaseModel):
        tasks: list[str] = Field(default_factory=list)
        keywords: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _normalize(self) -> "_SelfAssigning":
            self.keywords = [k.strip() for k in self.keywords]
            return self

    model = _FakeModel(contents=[{"tasks": ["recommend"]}])
    with pytest.raises(EmptyStructuredOutput):
        await call_structured(model, "q", _SelfAssigning, required_any=("keywords",))


async def test_required_any_typo_fails_loudly() -> None:
    """判据字段名不在 schema 上 → 当场炸。字段改名后这道闸会静默失效，那比炸难查得多。"""
    with pytest.raises(ValueError, match="required_any"):
        await call_structured(_FakeModel(content={}), "q", _Plan, required_any=("categorie",))


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
