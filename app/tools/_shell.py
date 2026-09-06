"""工具的**声明外壳**：把一个 async 函数变成带 name / description / 入参 schema 的工具对象。

为什么不直接让 12 个工具裸奔成普通函数、只留 ``FunctionTool`` 一副壳？因为工具的**入参强转**
必须有个 pydantic schema 落脚（``StrListArg``
那类 ``BeforeValidator`` 容错就活在这一步——模型爱把 list 编码成 JSON 字符串），而工具函数自己
只声明注解、不生成 schema。这层就干三件事：

1. 从签名 + 注解生成两份 schema：``args_schema``（全参数，直接调用用）与 ``tool_call_schema``
   （**剔除注入参数**，这份才是给模型看的）。
2. 记住 ``response_format``——只有 ``shopping_summary`` 用 ``content_and_artifact``（返回
   ``(给人看的文案, 结构化产物)``），两条通道的语义在这里保住。
3. 提供 ``ainvoke``：走 schema 校验后调实现。测试与离线脚本用它直接调工具，不必绕 Agent。

**注入参数**（``Annotated[T, InjectedToolArg]``）是「程序传、模型看不见」的入口：``item_picker``
的 candidates、``shopping_summary`` 的 picks 都是——模型侧的合法路径是登记表，注入口只留给
直接调用与单测。它们必须从 ``tool_call_schema`` 里消失，否则模型会以为自己该把候选抄一遍进来。

文件后半是 :func:`to_function_tool`——把声明好的工具包成 AgentScope 运行时真正调用的
``FunctionTool``。声明与运行时外壳挨在一起，是为了让「新增一个工具要做什么」只用看一个文件。
"""

import inspect
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolMiddlewareBase
from agentscope.tool._response import ToolChunk, ToolResultState
from pydantic import BaseModel, create_model


class InjectedToolArg:
    """标记「这个参数由程序注入，不给模型看」。用法：``Annotated[T, InjectedToolArg]``。

    只作哨兵用，不实例化（与 LangChain 同名标记的用法一致，迁移时工具签名一个字没改）。
    """


@dataclass
class ToolResult:
    """``ainvoke`` 收到 tool_call 形式入参时的返回体（两条通道并存）。

    ``content`` 是给模型 / 用户看的文本，``artifact`` 是结构化产物。只有
    ``response_format="content_and_artifact"`` 的工具会用到 artifact 这条通道。
    """

    content: Any
    artifact: Any = None
    name: str = ""
    tool_call_id: str = ""


def _is_injected(annotation: Any) -> bool:
    """注解是否带 ``InjectedToolArg`` 标记。"""
    if get_origin(annotation) is not Annotated:
        return False
    return any(meta is InjectedToolArg for meta in get_args(annotation)[1:])


def _build_schemas(func: Callable[..., Any]) -> tuple[type[BaseModel], type[BaseModel]]:
    """从函数签名生成 (args_schema, tool_call_schema)。

    注解保留 ``Annotated`` 元数据（``include_extras=True``）——``StrListArg`` 的
    ``BeforeValidator`` 就在里面，丢了它模型传来的 JSON 字符串列表会直接校验失败。
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    full: dict[str, Any] = {}
    visible: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        annotation = hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        full[name] = (annotation, default)
        if not _is_injected(annotation):
            visible[name] = (annotation, default)
    title = f"{func.__name__}_args"
    args_schema = create_model(title, **full)
    if len(visible) == len(full):
        return args_schema, args_schema
    return args_schema, create_model(title, **visible)


class ToolShell:
    """一个工具：实现函数 + 元数据 + 入参 schema。``@tool`` 装饰后得到的就是它。"""

    def __init__(self, func: Callable[..., Coroutine[Any, Any, Any]], response_format: str) -> None:
        if not inspect.iscoroutinefunction(func):
            # 本仓工具一律 async（全链路 async/await）。同步实现会在 loop 里阻塞住整条链，
            # 与其等它在压测时才现形，不如声明期就炸。
            raise TypeError(f"工具 {func.__name__} 必须是 async 函数")
        self.func = func
        self.name = func.__name__
        self.description = inspect.getdoc(func) or ""
        self.response_format = response_format
        self.args_schema, self.tool_call_schema = _build_schemas(func)

    @property
    def coroutine(self) -> Callable[..., Coroutine[Any, Any, Any]]:
        """原始 async 实现（``_as_tools`` 包 ``FunctionTool`` 时取的就是它）。"""
        return self.func

    async def __call__(self, **kwargs: Any) -> Any:
        """直接按关键字调实现，**不过 schema**（调用方自己保证类型）。"""
        return await self.func(**kwargs)

    async def ainvoke(self, payload: dict[str, Any]) -> Any:
        """按入参字典调用工具，入参先过 ``args_schema`` 校验强转。

        两种入参形态：
        - **普通 args 字典**：返回实现函数的返回值；``content_and_artifact`` 工具只返回文案那半
          （要拿结构化产物就用下面那种形态）。
        - **tool_call 字典**（含 ``type="tool_call"``）：返回 :class:`ToolResult`，文案在
          ``content``、结构化产物在 ``artifact``。
        """
        is_call = payload.get("type") == "tool_call"
        args = dict(payload.get("args") or {}) if is_call else dict(payload)
        validated = self.args_schema.model_validate(args)
        out = await self.func(**{n: getattr(validated, n) for n in self.args_schema.model_fields})
        if self.response_format == "content_and_artifact":
            content, artifact = out
        else:
            content, artifact = out, None
        if is_call:
            return ToolResult(
                content=content,
                artifact=artifact,
                name=self.name,
                tool_call_id=str(payload.get("id") or ""),
            )
        return content


def tool(
    func: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    *,
    response_format: str = "content",
) -> Any:
    """把 async 函数声明成工具。

    ``@tool`` 与 ``@tool(response_format="content_and_artifact")`` 两种写法都行。

    docstring 就是给模型看的描述（一句话摘要 + 何时调用 + 参数说明），别写成给人看的实现笔记。
    """

    def wrap(f: Callable[..., Coroutine[Any, Any, Any]]) -> ToolShell:
        return ToolShell(f, response_format)

    return wrap if func is None else wrap(func)


# ============================================================================
# 运行时外壳：ToolShell → AgentScope 的 FunctionTool
# ============================================================================


def _unwrap(out: Any) -> Any:
    """``content_and_artifact`` 工具返回 ``(给模型看的文本, 结构化 artifact)``——**取 artifact**。

    它是超集（``shopping_summary`` 的 artifact 里就含那段文案本身），而且下游按 schema 解析的
    那些环节——收尾取商品卡、harness 的 schema 断言、前端 tool_end payload——全都吃结构化那一份。

    反过来取文案那半的代价实测过一次：``_to_text`` 把整个元组 ``json.dumps(default=str)``，
    模型收到 ``["文案", "summary='…' items=[…]"]`` 这种半 repr 的东西，清单照样写得出来，但
    ``run_agent`` 再也解析不出 items——商品卡不出货、``result.json`` 不落盘、行为历史不记，
    全是静默的。
    """
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], BaseModel):
        return out[1]
    return out


def _to_text(out: Any) -> str:
    """实现函数的返回值 → 给模型看的文本。``*Output`` 一律 JSON，保持结构化契约。"""
    if isinstance(out, BaseModel):
        return out.model_dump_json()
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(out)


def to_function_tool(
    shell: ToolShell,
    *,
    is_read_only: bool = False,
    is_concurrency_safe: bool = True,
    middlewares: list[ToolMiddlewareBase] | None = None,
) -> FunctionTool:
    """把一个 :class:`ToolShell` 包成 AgentScope 的 ``FunctionTool``（loop 里真正被调的那副壳）。

    工厂式而非每个工具文件手写一份 wrapper：手写就是 12 份重复签名 + 12 处可能与实现漂移的
    参数默认值。走工厂则名字 / 描述 / 入参 schema **单一事实源**，且三件事一次做对：

    - **入参强转不丢**：过 ``tool_call_schema.model_validate`` 再调实现，``StrListArg`` 那类
      ``BeforeValidator`` 容错照常生效——直接把 kwargs 丢给实现函数就把这层丢了。
    - **错误语义一次到位**：实现抛出的异常统一转 ``state=ERROR`` + ``[error] ...`` 文本，
      harness 的 ``result_nudges`` 照读；不再有「Error invoking tool with kwargs」那种外层报错。
    - **形态钉死**：工具 return **单个** ToolChunk（不 yield 增量片段——本仓工具都是一次性产出
      完整 JSON，而框架对 yield 的多个 chunk 是增量拼接语义，会把结果拼坏。L0 的 S0 spike 实测）。

    Args:
        shell: ``@tool`` 声明出来的工具对象。
        is_read_only: 只读标记。**这是权限边界的依据**——SearchAgent 靠它与 ``PermissionEngine``
            做结构性拦截，不是靠提示词劝退，所以不能凭感觉标。
        is_concurrency_safe: 能否被同轮并发调用（``task_dispatch`` 要 True）。
        middlewares: 挂在这只工具上的中间件（harness 的工具适配器走这里）。**它持有 per-loop
            的状态**，所以带中间件的工具实例不能跨 loop 复用——见 ``tool_registry._make_tools``。
    """
    impl = shell.coroutine
    # tool_call_schema 而不是 args_schema：前者已剔除注入型参数，就是模型该看见的那份。
    schema = shell.tool_call_schema

    async def _call(**kwargs: Any) -> ToolChunk:
        try:
            # 先过 pydantic：StrListArg 这类 BeforeValidator 容错就活在这一步
            validated = schema.model_validate(kwargs)
            args = {name: getattr(validated, name) for name in type(validated).model_fields}
            out = _unwrap(await impl(**args))
        except Exception as exc:  # noqa: BLE001 - 工具内部错误不外抛，见模块 docstring
            return ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {type(exc).__name__}: {exc}")],
                state=ToolResultState.ERROR,
                metadata={"tool": shell.name},
            )
        return ToolChunk(
            content=[TextBlock(type="text", text=_to_text(out))],
            state=ToolResultState.SUCCESS,
            metadata={"schema": type(out).__name__, "tool": shell.name},
        )

    return FunctionTool(
        _call,
        name=shell.name,
        description=shell.description,
        input_schema=schema,
        is_read_only=is_read_only,
        is_concurrency_safe=is_concurrency_safe,
        middlewares=middlewares,
    )
