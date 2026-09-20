"""阶段 2 第 3 条：fallback 目标的能力门 + 切换可见性。

门的形态与计划原文不同（yaml 五列全绿 → 只判工具调用的否决门），理由写在
:mod:`app.agent.capabilities` 的 docstring 里。这里守两件最容易回退的事：

1. **查不到 ≠ 不支持**。litellm 的表查不到时静默返回 False，把 None 和 False 合并处理，
   就会把本仓自己的模型全判死（``dashscope/qwen3.8-flash`` 就不在表里）。
2. **Router 路径上的切换必须看得见**。父类的 ``_report_fallback_once`` 靠 ``role="fallback"``
   触发，而 litellm 内部的 fallback 角色始终是 main——不补那一下，切换在前端与日志里都是隐形的。
"""

import logging

import pytest

from app.agent import capabilities
from app.agent.capabilities import degraded_against, gate_fallback_refs, supports_tool_calls


class TestGate:
    def test_表里标不支持工具调用的目标被拒(self, caplog: pytest.LogCaptureFixture) -> None:
        """deepseek-reasoner 在 litellm 表里标 supports_function_calling=False。

        真去查表而不是 mock：这条断言同时在守「上游情报变了我们要知道」。它红了不代表代码坏，
        代表 deepseek 那边（或 litellm 的表）变了，正是我们要被告知的事。
        """
        assert supports_tool_calls("deepseek/deepseek-reasoner") is False
        with caplog.at_level(logging.ERROR):
            assert gate_fallback_refs(["deepseek/deepseek-reasoner"]) == []
        assert "不支持工具调用" in caplog.text

    def test_表里全绿的目标放行(self) -> None:
        assert supports_tool_calls("deepseek/deepseek-chat") is True
        assert supports_tool_calls("dashscope/deepseek-v4-flash") is True
        chain = ["deepseek/deepseek-chat"]
        assert gate_fallback_refs(chain) == chain

    def test_表里查不到的目标放行而不是判死(self, caplog: pytest.LogCaptureFixture) -> None:
        """本仓自己的模型名多半不在 litellm 表里；当成 False 会把主链直接清空。"""
        assert supports_tool_calls("dashscope/qwen3.8-flash") is None
        with caplog.at_level(logging.WARNING):
            assert gate_fallback_refs(["dashscope/qwen3.8-flash"]) == ["dashscope/qwen3.8-flash"]
        assert "能力未知" in caplog.text

    def test_链里只剔除被拒的那条(self) -> None:
        kept = gate_fallback_refs(["deepseek/deepseek-reasoner", "deepseek/deepseek-chat"])
        assert kept == ["deepseek/deepseek-chat"]

    def test_整条链全被拒时抬成error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert gate_fallback_refs(["deepseek/deepseek-reasoner"]) == []
        assert "等于没有备用出口" in caplog.text

    def test_情报源炸了不拦住装配(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """能力查询是附属品：它自己出问题时必须放行，不能把模型装配一起带走。"""

        def boom() -> None:
            raise RuntimeError("litellm 挂了")

        monkeypatch.setattr(capabilities, "configure_litellm", boom)
        assert supports_tool_calls("deepseek/deepseek-reasoner") is None
        assert gate_fallback_refs(["deepseek/deepseek-reasoner"]) == ["deepseek/deepseek-reasoner"]


class TestDegraded:
    def test_主家有备家没有的才算降级(self) -> None:
        """dashscope/deepseek-v4-flash 支持 reasoning，deepseek/deepseek-chat 不支持。"""
        out = degraded_against("dashscope/deepseek-v4-flash", "deepseek/deepseek-chat")
        assert "reasoning" in out

    def test_有一边查不到就不报假降级(self) -> None:
        assert degraded_against("dashscope/qwen3.8-flash", "deepseek/deepseek-chat") == []
        assert degraded_against("deepseek/deepseek-chat", "dashscope/qwen3.8-flash") == []

    def test_同一个出口不算降级(self) -> None:
        assert degraded_against("deepseek/deepseek-chat", "deepseek/deepseek-chat") == []
