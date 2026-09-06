"""``app/agent/runtime.py``：``run_agent`` 的运行时选择点（批 0 / L6）。

这层薄归薄，但它决定了 L8 的验收能不能做——评测脚本必须跟着 ``.env`` 的 ``AGENT_RUNTIME``
一起切到新链路，否则「跑 Rubric 对照迁移前基线」跑的还是老链路，绿得毫无意义。
"""

from app.agent import runtime


def test_default_is_langchain(monkeypatch) -> None:
    """迁移期默认仍是老链路——默认值改了就是悄悄换掉了线上主链路。"""
    monkeypatch.delenv("AGENT_RUNTIME", raising=False)
    assert runtime.agent_runtime() == runtime.RUNTIME_LANGCHAIN
    assert runtime.resolve_run_agent().__module__ == "app.agent.main_agent"


def test_agentscope_selected(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RUNTIME", "agentscope")
    assert runtime.resolve_run_agent().__module__ == "app.agent.orchestrator"


def test_value_is_normalized(monkeypatch) -> None:
    """.env 里手写的值常带空格 / 大小写不一，别让 ``AgentScope `` 静默落回 LangChain。"""
    monkeypatch.setenv("AGENT_RUNTIME", "  AgentScope ")
    assert runtime.agent_runtime() == runtime.RUNTIME_AGENTSCOPE
    assert runtime.resolve_run_agent().__module__ == "app.agent.orchestrator"


def test_unknown_value_falls_back(monkeypatch) -> None:
    """认不出的值退老链路（保守），不抛——启动期抛异常等于整个服务起不来。"""
    monkeypatch.setenv("AGENT_RUNTIME", "llamaindex")
    assert runtime.resolve_run_agent().__module__ == "app.agent.main_agent"


def test_both_implementations_share_signature() -> None:
    """两个 run_agent 的**调用面**必须一致——调用方（server / 三个评测脚本）不做分支。

    只比参数名与默认值，不比注解对象：``main_agent`` 带 ``from __future__ import annotations``、
    ``orchestrator`` 没带，注解一个是字符串一个是类型对象，比了永远不等，与调用面无关。
    """
    import inspect

    from app.agent.main_agent import run_agent as lc
    from app.agent.orchestrator import run_agent as as_

    def face(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert face(lc) == face(as_)
