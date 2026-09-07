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


async def call_structured(model: Any, prompt: Prompt, schema: type[T]) -> T:
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
    """
    messages = to_msgs(prompt)
    fallback_on: tuple[type[Exception], ...] = (StructuredOutputError, *_fallback_exceptions(model))
    try:
        response = await model.generate_structured_output(
            messages, schema, tool_choice=ToolChoice(mode="auto")
        )
    except fallback_on as exc:
        logger.info("结构化输出 auto 路径未产出（%s），回落框架策略梯", type(exc).__name__)
        response = await model.generate_structured_output(messages, schema)
    charge_usage(getattr(model, "model", ""), getattr(response, "usage", None))
    return schema.model_validate(_unwrap_structured(response.content or {}, schema))


def _fallback_exceptions(model: Any) -> tuple[type[Exception], ...]:
    """供应商侧「拒绝这种 tool_choice」的异常类（框架按模型类给），拿不到就只认框架自己的。"""
    getter = getattr(model, "_get_structured_output_fallback_exceptions", None)
    try:
        return tuple(getter()) if callable(getter) else ()
    except Exception:  # noqa: BLE001 —— 假模型 / 老版本没有这个钩子
        return ()
