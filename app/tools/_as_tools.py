"""AgentScope 侧的工具外壳：把现有工具实现包成 ``FunctionTool``（批 0 / L2）。

**双包装并存**：迁移期每个工具同时挂两副壳——旧的 LangChain ``@tool``（1042 个存量测试与
线上链路都挂在它上面）与新的 ``FunctionTool``（AgentScope 运行时用）。两副壳指向**同一个
async 实现函数**，所以不存在「新旧行为漂移」这种最难查的迁移 bug。L8 摘 LangChain 时删的是
旧壳与本模块对 ``StructuredTool`` 的依赖，实现函数本身一行不动。

**为什么用工厂而不是每个文件手写一份 wrapper**（对执行手册模板的一处刻意偏离）：
手册模板要求每个工具文件里再写一遍完整签名的 ``async def xxx(...) -> ToolChunk``。12 个工具
就是 12 份重复签名 + 12 处可能与实现漂移的参数默认值，而且为了让 ``FunctionTool`` 的注解反射
生效还得逐个删掉 ``from __future__ import annotations``。工厂路线把这些一次做对：

- **名字 / 描述 / 入参 schema 单一事实源**：全部取自已有的 tool 对象（描述就是给模型看的
  docstring），新旧两副壳看到的**永远是同一份**，不可能漂。
- **入参强转不丢**：走 ``args_schema.model_validate`` 再调实现，``StrListArg`` 那类
  ``BeforeValidator`` 容错（模型把 list 编码成 JSON 字符串）在新壳上照常生效——这是必须走
  校验的硬理由，直接把 kwargs 丢给实现函数就把这层容错丢了。
- **错误语义一次到位**：实现抛出的异常统一转 ``state=ERROR`` + ``[error] ...`` 文本，
  harness 的 ``result_nudges`` 照读；不再有「Error invoking tool with kwargs」那种外层报错。

L0 的 S0 spike 钉死的形态在这里落地：**工具 return 单个 ToolChunk**（不 yield 增量片段——
本仓工具都是一次性产出完整 JSON，yield 的增量拼接语义会把结果拼坏）。
"""

import json
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolMiddlewareBase
from agentscope.tool._response import ToolChunk, ToolResultState
from langchain_core.tools import BaseTool
from pydantic import BaseModel


def _unwrap(out: Any) -> Any:
    """LangChain 的 ``content_and_artifact`` 返回 ``(给模型看的文本, 结构化 artifact)``。

    那边有两条通道（``ToolMessage.content`` 给模型、``.artifact`` 给程序），AgentScope 只有一条，
    所以**取 artifact**：它是超集（``shopping_summary`` 的 artifact 里就含那段文案本身），
    而且下游按 schema 解析的那些环节——收尾取商品卡、harness 的 schema 断言、前端 tool_end
    payload——全都吃结构化那一份。

    反过来取 content 的代价实测过一次：``_to_text`` 把整个元组 ``json.dumps(default=str)``，
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


def as_function_tool(
    lc_tool: BaseTool,
    *,
    is_read_only: bool = False,
    is_concurrency_safe: bool = True,
    middlewares: list[ToolMiddlewareBase] | None = None,
) -> FunctionTool:
    """把一个现有的 LangChain 工具对象包成 ``FunctionTool``，共用它的实现与元数据。

    Args:
        lc_tool: 现有的 ``@tool`` 对象（``.coroutine`` 是原始 async 实现）。
        is_read_only: 只读标记。**这是权限边界的依据**——批 1 的 SearchAgent 靠它与
            ``PermissionEngine`` 做结构性拦截，不是靠提示词劝退，所以不能凭感觉标。
        is_concurrency_safe: 能否被同轮并发调用（``task_dispatch`` 要 True）。
        middlewares: 挂在这只工具上的中间件（harness 的工具适配器走这里）。**它持有 per-loop
            的状态**，所以带中间件的工具实例不能跨 loop 复用——见 ``tool_registry._make_as_tools``。
    """
    impl = getattr(lc_tool, "coroutine", None)
    if impl is None:  # pragma: no cover - 本仓工具全是 async
        raise TypeError(f"工具 {lc_tool.name} 没有 async 实现，无法包成 FunctionTool")
    # tool_call_schema 而不是 args_schema：前者已剔除注入型参数，就是模型该看见的那份。
    schema = lc_tool.tool_call_schema
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        # 兜住「有朝一日某个工具改成 dict schema」的情况——那时入参强转会静默失效，必须炸出来。
        raise TypeError(f"工具 {lc_tool.name} 的入参 schema 不是 pydantic 模型：{schema!r}")

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
                metadata={"tool": lc_tool.name},
            )
        return ToolChunk(
            content=[TextBlock(type="text", text=_to_text(out))],
            state=ToolResultState.SUCCESS,
            metadata={"schema": type(out).__name__, "tool": lc_tool.name},
        )

    return FunctionTool(
        _call,
        name=lc_tool.name,
        description=lc_tool.description,
        input_schema=schema,
        is_read_only=is_read_only,
        is_concurrency_safe=is_concurrency_safe,
        middlewares=middlewares,
    )
