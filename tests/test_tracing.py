"""Langfuse 观测：fork 子 loop 的**双记**回归。

坑的形状（实测于一条 5 平台 fork 的真 trace）：子 Agent 的每次 LLM 调用、每个 item_search span
都**成对出现**，trace 里 token 虚高 65%——照着它做成本分析全歪。根因不在 Langfuse，在
LangGraph：它的 ``ensure_config`` 对 ``callbacks`` 是**合并**（``_merge_callbacks``），不是
LangChain 原生 ``ensure_config`` 的**覆盖**。工具体内 ``var_child_runnable_config`` 里躺着父 loop
的 handler（LangChain 跑工具时写入的「本次 tool run 的子 callback manager」），子 Agent 的
``ainvoke`` 于是同时带着父 + 子两个 handler。

修法是「父 handler 已在场就别再挂」。它的**前提是 LangGraph 的合并语义**——所以第一个测试直接把
这条语义钉死：哪天 LangGraph 改成覆盖，子 Agent 会**彻底丢观测**（不是双记，是没有），届时这个
测试先红，提醒我们把 handler 加回去。
"""

from typing import Any, TypedDict

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from app.agent import tracing as T


class _Counter(AsyncCallbackHandler):
    """数 on_chat_model_start 触发次数——「一次 LLM 调用被记几遍」的直接度量。"""

    def __init__(self) -> None:
        self.calls = 0

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1


class _FakeClient:
    def create_trace_id(self) -> str:
        return "trace-fixed"


class _SubState(TypedDict):
    x: str


def _sub_graph() -> Any:
    """一个最小的「子 Agent」：LangGraph 图，节点里打一次 LLM。"""
    llm = FakeListChatModel(responses=["ok"] * 4)

    async def node(_: _SubState) -> _SubState:
        await llm.ainvoke("hi")
        return {"x": "done"}

    builder: StateGraph[_SubState, None, _SubState, _SubState] = StateGraph(_SubState)
    # 包成 RunnableLambda 而非直接传 async 函数：功能等价，但 add_node 的重载认得出它的类型。
    builder.add_node("n", RunnableLambda(node))
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile()


async def test_langgraph_merges_ambient_callbacks_into_subgraph() -> None:
    """钉死修法的前提：子图**继承**父的 handler（合并语义），故子 loop 不必自己再挂一个。

    父 handler 计到 1 = 子图的 LLM 调用确实被父记到了。若哪天这里变成 0，说明 LangGraph 改成了
    覆盖语义 → apply_tracing 的 root=False 分支必须把 handler 加回去，否则 fork 全无观测。
    """
    parent, child = _Counter(), _Counter()
    graph = _sub_graph()

    @tool
    async def dispatch(x: str) -> str:
        """复刻 dispatch_tool：工具体内新建 config，invoke 一个子图。"""
        cfg: RunnableConfig = {"callbacks": [child], "recursion_limit": 10}
        await graph.ainvoke({"x": x}, config=cfg)
        return "done"

    await dispatch.ainvoke({"x": "go"}, config={"callbacks": [parent]})

    assert parent.calls == 1  # 父 handler 自动继承到了子图内部 —— 修法成立的前提
    assert child.calls == 1  # 子 handler 也记了一遍 —— 两者相加即「双记」


@pytest.mark.parametrize("as_manager", [False, True])
def test_detects_inherited_langfuse_handler(as_manager: bool) -> None:
    """检测既要认 list 形态的 callbacks，也要认 CallbackManager 形态（真实工具上下文里是后者）。"""
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler()
    callbacks: Any = (
        AsyncCallbackManager(handlers=[], inheritable_handlers=[handler])
        if as_manager
        else [handler]
    )
    token = var_child_runnable_config.set({"callbacks": callbacks})
    try:
        assert T._langfuse_handler_inherited() is True
    finally:
        var_child_runnable_config.reset(token)


def test_ignores_non_langfuse_handlers() -> None:
    """上下文里只有别人的回调时不算「已继承」——子 loop 仍需自己挂，否则真的没 trace。"""
    token = var_child_runnable_config.set({"callbacks": [_Counter()]})
    try:
        assert T._langfuse_handler_inherited() is False
    finally:
        var_child_runnable_config.reset(token)


def _handler_count(config: RunnableConfig) -> int:
    callbacks = config.get("callbacks") or []
    return len(callbacks) if isinstance(callbacks, list) else 0


def test_fork_does_not_attach_second_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心回归：父 handler 已在上下文里时，fork 不再追加 handler（否则每次调用记两遍）。"""
    monkeypatch.setattr(T, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(T, "_langfuse_handler_inherited", lambda: True)

    config: RunnableConfig = {"recursion_limit": 10}
    T.apply_tracing(config)  # root=False，即 dispatch_tool 的调用形态

    assert _handler_count(config) == 0


def test_fork_still_attaches_when_no_parent_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """examples / 测试直调子 Agent（没有父 run 上下文）时照旧自己挂——不能因为修双记把这条路撇了。"""
    monkeypatch.setattr(T, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(T, "_langfuse_handler_inherited", lambda: False)
    monkeypatch.setattr(T, "_make_handler", lambda _client, _tid: _Counter())

    config: RunnableConfig = {"recursion_limit": 10}
    T.apply_tracing(config)

    assert _handler_count(config) == 1


def test_root_always_attaches(monkeypatch: pytest.MonkeyPatch) -> None:
    """主 loop 是 trace 的源头，任何情况下都要挂 handler + 写下本轮 trace_id。"""
    monkeypatch.setattr(T, "_get_client", lambda: _FakeClient())
    # 即便「已继承」也不影响 root——主 loop 就是那个源头 handler 的出处。
    monkeypatch.setattr(T, "_langfuse_handler_inherited", lambda: True)
    monkeypatch.setattr(T, "_make_handler", lambda _client, _tid: _Counter())

    config: RunnableConfig = {"recursion_limit": 10}
    T.apply_tracing(config, session_id="thread-1", root=True)

    assert _handler_count(config) == 1
    assert T.current_trace_id() == "trace-fixed"
