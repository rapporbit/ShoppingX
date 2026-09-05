"""L2 工具新壳（FunctionTool）：入参强转 / 错误语义 / 只读标记 / 与旧壳同源。

新壳与旧 ``@tool`` 壳指向同一个实现函数，所以这里**不重测业务逻辑**（那是 test_tools.py 的
81 个用例的事），只测外壳该负责的四件事：参数怎么进去、异常怎么出来、标记对不对、
两副壳的元数据是不是同一份。
"""

import json

import pytest
from agentscope.message import ToolCallBlock
from agentscope.state import AgentState
from agentscope.tool import Toolkit, ToolResponse
from agentscope.tool._response import ToolResultState
from langchain_core.tools import tool
from pydantic import BaseModel

from app.tools._args import StrListArg
from app.tools._as_tools import as_function_tool


class _Out(BaseModel):
    got: list[str]
    n: int


@tool
async def _probe(names: StrListArg, n: int = 1) -> _Out:
    """探针工具。

    参数：
      - names：名字列表。
      - n：次数。
    """
    return _Out(got=names, n=n)


@tool
async def _boomer(x: str) -> _Out:
    """总是炸的探针工具。

    参数：
      - x：随便。
    """
    raise RuntimeError(f"炸了：{x}")


@tool(response_format="content_and_artifact")
async def _artifact_probe(x: str) -> tuple[str, _Out]:
    """content_and_artifact 型探针（形态同 shopping_summary）。

    参数：
      - x：随便。
    """
    return f"给模型看的文案：{x}", _Out(got=[x], n=7)


async def _run(ft, **kwargs) -> ToolResponse:  # type: ignore[no-untyped-def]
    """走 Toolkit.call_tool——Agent 的真实路径，不是直接 await 工具函数。"""
    toolkit = Toolkit()
    await toolkit.add_tool(ft)
    call = ToolCallBlock(
        type="tool_call",
        id="c1",
        name=ft.name,
        input=json.dumps(kwargs),
    )
    final = None
    async for item in toolkit.call_tool(call, AgentState()):
        if isinstance(item, ToolResponse):
            final = item
    assert final is not None
    return final


def _text(resp: ToolResponse) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


@pytest.mark.asyncio
async def test_success_returns_json_and_schema_metadata() -> None:
    resp = await _run(as_function_tool(_probe, is_read_only=True), names=["a", "b"], n=3)
    assert resp.state == ToolResultState.SUCCESS
    assert json.loads(_text(resp)) == {"got": ["a", "b"], "n": 3}
    # metadata 里的 schema 名是 harness schema_assertion 与前端 tool_end 的依据
    assert resp.metadata["schema"] == "_Out"


@pytest.mark.asyncio
async def test_stringified_list_is_coerced() -> None:
    """模型把 list 编码成 JSON 字符串时照样能跑——新壳必须走 pydantic 校验才有这层容错。

    这是「工厂而不是手写 wrapper」的硬理由：直接把 kwargs 丢给实现函数就把 StrListArg 的
    BeforeValidator 绕过去了，换个模型（qwen3.5-flash）就连挂 4 次。
    """
    resp = await _run(as_function_tool(_probe), names='["塑料","plastic"]')
    assert resp.state == ToolResultState.SUCCESS
    assert json.loads(_text(resp))["got"] == ["塑料", "plastic"]


@pytest.mark.asyncio
async def test_implementation_error_becomes_error_chunk_not_raise() -> None:
    """工具内部报错不外抛：转 state=ERROR + [error] 文本，让 harness 的 result_nudges 读得到。"""
    resp = await _run(as_function_tool(_boomer), x="ok")
    assert resp.state == ToolResultState.ERROR
    assert _text(resp).startswith("[error] RuntimeError: 炸了：ok")


@pytest.mark.asyncio
async def test_invalid_args_also_become_error_chunk() -> None:
    """入参校验失败同样走 ERROR 通路，而不是抛出「Error invoking tool with kwargs」那种外层报错。"""
    resp = await _run(as_function_tool(_probe), names=123)
    assert resp.state == ToolResultState.ERROR
    assert "[error] ValidationError" in _text(resp)


# --- 真实 12 个工具：标记与同源 -------------------------------------------------


def test_as_tools_cover_all_business_tools_with_same_metadata() -> None:
    """新旧两副壳一一对应，元数据取自同一处——描述就是给模型看的 docstring，漂了就是两套行为。"""
    from app.agent.tool_registry import _BUSINESS_TOOLS, AS_TOOLS, AS_TOOLS_BY_NAME

    # 12 个业务工具 + L3 加进来的派发入口 task_dispatch（它没有 LangChain 旧壳，是原生新工具）
    assert len(_BUSINESS_TOOLS) == 12
    assert len(AS_TOOLS) == 13
    assert "task_dispatch" in AS_TOOLS_BY_NAME
    for lc in _BUSINESS_TOOLS:
        ft = AS_TOOLS_BY_NAME[lc.name]
        assert ft.description == lc.description


def test_read_only_flags_are_exactly_the_read_side() -> None:
    """只读标记是批 1 权限边界的依据，不是注释——写工具误标 True 会让 SearchAgent 拿到下单能力。"""
    from app.agent.tool_registry import AS_TOOLS

    read_only = {t.name for t in AS_TOOLS if t.is_read_only}
    assert read_only == {
        "planner",
        "image_understand",
        "item_search",
        "price_compare",
        "shipping_calc",
        "category_insight",
        "item_picker",
        "web_search",
    }
    # 会挂起等用户、删长期偏好、决定 loop 收尾的四个，一个都不许标只读
    write_side = {"ask_user", "forget_preference", "shopping_summary", "chat_fallback"}
    assert write_side & read_only == set()


@pytest.mark.asyncio
async def test_build_toolkit_roles_produce_schemas() -> None:
    from app.agent.tool_registry import build_toolkit

    for role in ("main", "search", "trade"):
        toolkit = await build_toolkit(role)
        schemas = await toolkit.get_tool_schemas()
        # 批 0 三个角色都发全集：12 业务工具 + task_dispatch（读写切分是批 1 的事）
        assert len(schemas) == 13
        assert all(s["function"]["description"] for s in schemas)

    with pytest.raises(ValueError):
        await build_toolkit("nope")


async def test_content_and_artifact_tool_yields_structured_json() -> None:
    """``content_and_artifact`` 型工具（shopping_summary）新壳必须吐**结构化那一份**。

    取错通道不会报错，只会让 run_agent 解析不出 items——商品卡不出货、result.json 不落盘，
    全程静默。L3 的真实 LLM 验收就是这么发现的，这里钉死。
    """
    resp = await _run(as_function_tool(_artifact_probe), x="买包")
    assert resp.state == ToolResultState.SUCCESS
    # 取的是 artifact 那一份（可解析回 schema），不是把元组 json.dumps 出来的半 repr
    assert json.loads(_text(resp)) == {"got": ["买包"], "n": 7}
    assert resp.metadata["schema"] == "_Out"
