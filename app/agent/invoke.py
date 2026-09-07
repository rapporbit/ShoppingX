"""对 AgentScope 模型做**一次性调用**的薄助手（批 0 / L6 起）。

不是所有 LLM 调用都跑在 Agent loop 里：评测判官、离线标注投票、记忆管家、偏好解析……这些是
「给一段 prompt、要一段回答」的单次调用。LangChain 侧有 ``llm.ainvoke(prompt)`` 这种一行写法，
AgentScope 侧则要自己处理三件事：

1. 入参得包成 ``list[Msg]``（模型层不吃裸字符串）；
2. 本仓的模型一律 ``stream=True``（主 loop 要流式），于是 ``__call__`` 返回的是异步生成器——
   基类会**累积**增量并在最后吐一个 ``is_last=True`` 的完整 ``ChatResponse``，所以取结果 =
   迭代到最后一个；半路 break 拿到的是残缺前缀（且会把网关闸门的 slot 留在生成器里，见
   ``gateway._stream_holding_slot``）；
3. 文本在 ``response.content`` 的 text block 里，不是 ``.text`` 属性。

三件事各处重写一遍迟早漂，收在这里。结构化输出（原 ``with_structured_output``）走同模块的
:func:`call_structured`，同样别各写各的。
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, TypeVar

from agentscope.exception import StructuredOutputError
from agentscope.message import Msg, TextBlock
from agentscope.tool import ToolChoice
from pydantic import BaseModel

from app.agent.token_budget import charge_usage

logger = logging.getLogger("shoppingx.invoke")

T = TypeVar("T", bound=BaseModel)

# 入参可以是一段纯 prompt、``[("system", ...), ("user", ...)]`` 这种 LangChain 老写法，
# 或已经造好的 ``Msg``。老写法保留是为了让迁移时调用点只换函数名、不重排 prompt 拼装。
Prompt = str | Sequence[tuple[str, str]] | Sequence[Msg]


def to_msgs(prompt: Prompt) -> list[Msg]:
    """把三种入参形态归一成 ``list[Msg]``。"""
    if isinstance(prompt, str):
        return [Msg(name="user", role="user", content=[TextBlock(type="text", text=prompt)])]
    msgs: list[Msg] = []
    for item in prompt:
        if isinstance(item, Msg):
            msgs.append(item)
            continue
        role, text = item
        if role == "system":
            kind: Literal["user", "assistant", "system"] = "system"
        elif role == "assistant":
            kind = "assistant"
        else:  # 老写法里只可能是这三种，未知角色按 user 送（宁可多说一句也不要丢内容）
            kind = "user"
        msgs.append(Msg(name=kind, role=kind, content=[TextBlock(type="text", text=text)]))
    return msgs


def _text_of(response: Any) -> str:
    """从 ``ChatResponse`` 取纯文本：拼所有 text block，丢 thinking / tool_use / 多媒体块。"""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype != "text":
            continue
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


async def call_text(model: Any, prompt: Prompt) -> str:
    """给模型一段 prompt，拿回完整文本。流式 / 非流式两种模型都吃。

    **超时由调用方包**（``async with asyncio.timeout(...)``）：本函数只管把一次调用跑完整，
    「等多久算挂了」是各场景自己的事——离线批量标注等得起 150 秒，线上一次判分等不起。
    挂起的请求会占死并发槽让 ``gather`` 永不返回，批量脚本务必包上（M21 实测教训）。
    """
    result = await model(to_msgs(prompt))
    if not hasattr(result, "__aiter__"):
        charge_usage(getattr(model, "model", ""), getattr(result, "usage", None))
        return _text_of(result)
    last: Any = None
    async for chunk in result:
        last = chunk
    if last is None:
        return ""
    charge_usage(getattr(model, "model", ""), getattr(last, "usage", None))
    return _text_of(last)


def _unwrap_structured(content: Any, schema: type[BaseModel]) -> Any:
    """剥掉供应商给工具入参多包的那层壳。

    实测（批 0 / L7，DashScope 兼容端点）：``deepseek-v4-flash`` 回来的 tool_call 入参是
    ``{"parameters": {真正的字段…}}``，回落 auto 策略时又变成 ``{"output": {…}}``；
    同一条请求换 ``qwen3.5-flash`` 则字段直接在顶层。框架不管这层（它把 tool_call 的
    input 原样 json.loads 就返回），而**多包一层不会报错**——业务 schema 的字段全带默认值，
    ``model_validate({"parameters": …})`` 照样通过，只是**每个字段都是默认值**。症状是
    planner 恒定解析出空意图、judge 恒定 0 条细则，全程静默。

    所以判据不写死键名（实测已见两个不同的键，写死必漏），改判**字段交集**：顶层键与
    schema 字段有交集就是真结果；否则单键且值为 dict 就下钻（最多两层，防病态嵌套）。
    """
    fields = set(schema.model_fields)
    fields |= {f.alias for f in schema.model_fields.values() if f.alias}
    for _ in range(2):
        if not isinstance(content, dict) or not content or set(content) & fields:
            break
        if len(content) != 1:
            break
        inner = next(iter(content.values()))
        if not isinstance(inner, dict):
            break
        content = inner
    return content


class EmptyStructuredOutput(RuntimeError):
    """连续两次采样都拿回一张「校验得过的空表」——调用方据此走各自的降级。

    单独立一个异常类（而不是复用 ``ValueError``）是为了让调用点能把它**和真的解析失败区分开**：
    解析失败是「这次请求坏了」，空表是「模型在这条 prompt 上退化了」，两者该记的日志和该给用户
    的话都不一样。带上 schema 名与判据字段，是因为线上只看得到一行日志。
    """

    def __init__(self, schema_name: str, missing: Sequence[str]) -> None:
        self.schema_name = schema_name
        self.missing = tuple(missing)
        super().__init__(
            f"{schema_name} 连续两次结构化输出都是空表："
            f"{'/'.join(self.missing)} 一个都没出现（模型只回了存根）"
        )


def _present_fields(instance: BaseModel, raw: Any) -> set[str]:
    """这次模型**真正吐出来过**的字段名 —— 判据是**剥壳后原始 dict 的键**。

    为什么判「键出现过」而不是「值不等于默认值」：用户确实没给预算时 ``budget_amount=None``
    是合法结果，按值比较会把它误判成空表，于是每条无预算的 query 都白白多采样一次。同理，
    模型显式吐 ``"category": null`` 是它在认真回答「这项没有」（本仓多个 schema 挂了
    ``drop_none_values`` 把显式 null 归一为缺席），不是退化成存根，该算出现过。

    为什么**不用** ``model_fields_set``：它记的不只是输入里的键，还包括 ``mode="after"``
    校验器自己赋过的字段。实测 ``PlanOutput`` 的 ``_resolve_exclude_strength`` /
    ``_clean_bundle_slots`` 会在校验期给 ``exclude_keywords`` / ``bundle_slots`` 赋值，于是
    对一张 ``{"tasks": [...]}`` 的存根，``model_fields_set`` 里凭空多出这两个名字——闸门恒不
    触发，本身就成了一处静默失效。只有 ``raw`` 不是 dict（供应商吐了别的形态）时才退回它。
    """
    if not isinstance(raw, dict):
        return set(instance.model_fields_set)
    # 模型按 alias 吐键时归一回字段名，否则 required_any 写字段名会匹配不上。
    alias_to_name = {f.alias: n for n, f in type(instance).model_fields.items() if f.alias}
    return {alias_to_name.get(k, k) for k in raw if isinstance(k, str)}


async def call_structured(
    model: Any, prompt: Prompt, schema: type[T], *, required_any: Sequence[str] = ()
) -> T:
    """给模型一段 prompt，拿回一个**已验证**的 ``schema`` 实例。

    取代 LangChain 的 ``llm.with_structured_output(S, method="function_calling").ainvoke(...)``。
    AgentScope 的 ``generate_structured_output`` 自带策略梯（forced → auto → no_think → none），
    所以记忆 structured-output-method-must-be-pinned 里「默认 method 随模型浮动、qwen 系走
    json_object 打挂 planner」的坑在这条路上结构性不存在，**不用也没法再钉 method**（L0/S2 实测）。

    三处踩过的坑钉在这里：① 结果在 ``StructuredResponse.content``（dict），**不是**
    ``.metadata``——读错字段配上「全字段有默认值」的 schema，``model_validate({})`` 会给出假绿；
    ② 部分供应商把入参多包一层壳，见 :func:`_unwrap_structured`；③ 用量在 ``.usage`` 上，
    顺手入账（LangChain 侧要挂 ``UsageMetadataCallbackHandler`` 才有账）。

    **第四个坑（2026-09-08 挖出，比前三个都大）：forced tool_choice 拿到的是存根。** DashScope 上的
    ``deepseek-v4-flash`` 被 ``tool_choice`` 强制指定函数时，只回 ``{"tasks": ["recommend"]}`` 这种
    一两个字段的入参（思考开关无关、stream 无关、绕开框架直连 OpenAI 客户端同样复现）；换成
    ``auto`` 就回完整参数（q21 五次采样全部拆出三个槽 + 预算 300）。框架的策略梯把 forced 排第一，
    而存根是合法 JSON、全默认值的 schema 照样校验通过——于是永远轮不到 auto，线上多数轮次拿着
    一张空计划在跑（15 次采样 13 次只有 1~4 个字段），planner 的预算 / 品类 / 槽位全靠下游规则
    兜底。这是批 0 迁移漏掉的第 6 个静默失效。

    所以这里**auto 优先**：先显式 ``tool_choice=auto`` 走一次（框架里显式 tool_choice 绕过策略梯），
    模型没调工具（``StructuredOutputError``）或供应商拒绝时，再回落框架默认梯（forced → …）。
    auto 路径下模型可能先写一段推理文本再调工具，多花的是几十个 token，换回来的是整张表。

    失败照抛（``StructuredOutputError`` / 校验错），由调用方决定降级——各处的降级语义不一样
    （planner 回退规则解析、curator 整轮跳过），收在这里只会把它们抹平。

    ``required_any``:**空表闸**。剥完壳、校验完了，结果里这几个字段一个都没出现过 → 这次采样
    是张存根（实测：DashScope 上的 ``deepseek-v4-flash`` 被 forced tool_choice 时只回
    ``{"tasks":["recommend"]}``；auto 策略也偶发回 0 个字段），**重采样一次**；第二次仍空则抛
    :class:`EmptyStructuredOutput`。为什么非得有这道闸：本仓的输出 schema 几乎全字段带默认值，
    ``model_validate(存根)`` 校验照过，调用方拿到一张「看起来合法」的空表，全程零报错——planner
    恒定解析出空意图、judge 恒定 0 条打分，症状全是静默跑空。默认不传 = 老行为一字不变
    （空结果本就合法的调用点——偏好解析、记忆管家——不该被逼着重采样）。
    """
    unknown = [name for name in required_any if name not in schema.model_fields]
    if unknown:
        # 字段改名后这道闸会静默失效（永远判不出空表），比当场炸难查得多。
        raise ValueError(f"required_any 里的字段不在 {schema.__name__} 上：{unknown}")
    msgs = to_msgs(prompt)
    for attempt in range(2):
        response = await _generate_auto_first(model, msgs, schema)
        charge_usage(getattr(model, "model", ""), getattr(response, "usage", None))
        raw = _unwrap_structured(response.content or {}, schema)
        out = schema.model_validate(raw)
        if not required_any or _present_fields(out, raw) & set(required_any):
            return out
        logger.warning(
            "%s 第 %d 次结构化输出是空表（%s 一个都没出现），重采样",
            schema.__name__,
            attempt + 1,
            "/".join(required_any),
        )
    raise EmptyStructuredOutput(schema.__name__, required_any)


async def _generate_auto_first(model: Any, msgs: list[Msg], schema: type[BaseModel]) -> Any:
    """一次结构化采样：显式 ``tool_choice=auto`` 优先，模型没调工具 / 供应商拒绝时回落框架梯。"""
    fallback_on: tuple[type[Exception], ...] = (StructuredOutputError, *_fallback_exceptions(model))
    try:
        return await model.generate_structured_output(
            msgs, schema, tool_choice=ToolChoice(mode="auto")
        )
    except fallback_on as exc:
        logger.info("结构化输出 auto 路径未产出（%s），回落框架策略梯", type(exc).__name__)
        return await model.generate_structured_output(msgs, schema)


def _fallback_exceptions(model: Any) -> tuple[type[Exception], ...]:
    """供应商侧「拒绝这种 tool_choice」的异常类（框架按模型类给），拿不到就只认框架自己的。"""
    getter = getattr(model, "_get_structured_output_fallback_exceptions", None)
    try:
        return tuple(getter()) if callable(getter) else ()
    except Exception:  # noqa: BLE001 —— 假模型 / 老版本没有这个钩子
        return ()
