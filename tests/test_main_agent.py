"""M9 验收：主 AgentLoop 组装的确定性测试（不依赖真实 LLM）。

覆盖 ROADMAP M9 的工程接缝（合龙点）：
- ``run_agent`` 入口注入偏好 + 会话级 P_t → 跑主 loop（这里换成假 agent）→ 收尾跑记忆管家
  （curator，这里 monkeypatch 成假的，避免打真实 LLM）+ 上报。
- ``_extract_summary`` 从 shopping_summary 的 artifact 可靠取回结构化清单。
- Harness 控制面（安全四层③④）：pre_tool_call 各硬闸 + post_tool_call 截断 / 循环检测提示。
- ``build_agent_middleware`` 每次新建（LoopDetector 有状态，不可跨 Agent 复用）。

真实 LLM 全链路（planner→fork item_search→…→shopping_summary 并正常终止）见
``examples/09_main_agent.py``——那条是功能验收，不进单测（不依赖外部模型）。
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agent.main_agent as main_agent
import app.memory.injector as injector
from app.harness.agent_middleware import HarnessAgentMiddleware, build_agent_middleware
from app.harness.budgets import (
    MAX_TERMINAL_NUDGE_RETRIES,
    SUB_ITEM_SEARCH_CAP,
    fork_budget_scope,
)
from app.harness.setup import setup_harness
from app.harness.state import GuardState
from app.harness.truncation import truncate_tool_result
from app.memory.history import read_turns
from app.memory.session_state import SessionConstraint, SessionPrefState, save_pt
from app.memory.store import PreferenceStore, get_store
from app.tools.shopping_summary import ShoppingSummaryOutput, SummaryItem


def _store() -> PreferenceStore:
    """本地后端 + 确定性本地编码器，整段离线可复现（与 test_memory 同套路）。"""
    return get_store()


def _summary_msg() -> ToolMessage:
    """造一条 shopping_summary 的 ToolMessage（content 可读文案 + artifact 结构化清单）。

    summary 已不产偏好（偏好沉淀剥离给会话结束后的 curator）——artifact 只含清单文案 + 商品卡。
    """
    out = ShoppingSummaryOutput(
        summary="为你精选 1 件：帆布旅行包。",
        items=[SummaryItem(item_id="A1", platform="amazon", title="canvas bag")],
    )
    return ToolMessage(
        content=out.summary, name="shopping_summary", tool_call_id="c1", artifact=out
    )


async def _noop_curate(*_args: Any, **_kwargs: Any) -> None:
    """假 curator：什么都不做——run_agent 测试用它顶掉真实记忆判定（否则会打真实 LLM）。"""
    return None


# ---------- _extract_summary ----------
def test_extract_summary_from_artifact() -> None:
    messages = [
        HumanMessage(content="买个旅行包"),
        _summary_msg(),
        AIMessage(content="已为你整理好清单。"),
    ]
    got = main_agent._extract_summary(messages)
    assert got is not None
    # 从 artifact 可靠取回结构化清单（商品卡 items）。
    assert got.items[0].item_id == "A1"
    assert got.summary == "为你精选 1 件：帆布旅行包。"


def test_extract_summary_none_without_summary() -> None:
    # 闲聊兜底走 chat_fallback，没有 shopping_summary → 返回 None，不写回 / 不落产物。
    messages = [HumanMessage(content="你好"), AIMessage(content="你好，我能帮你找商品。")]
    assert main_agent._extract_summary(messages) is None


# ---------- run_agent 全链路接缝（假 agent + 假 curator，无真实 LLM）----------
async def test_run_agent_injects_persists_and_reports(tmp_path: Path, monkeypatch: Any) -> None:
    store = _store()
    monkeypatch.setattr(injector, "get_store", lambda: store)

    # 会话已累积一条 P_t 约束——验证它不进 system prompt，而是拼进当轮 human message 尾部。
    session_dir = main_agent.ensure_session_dir("t-1")
    save_pt(
        session_dir,
        SessionPrefState(
            constraints=[
                SessionConstraint(
                    id="c1",
                    category="material",
                    content="不要塑料",
                    polarity="dislike",
                    keywords=["塑料", "plastic"],
                )
            ]
        ),
    )

    captured: dict[str, Any] = {}

    def _fake_build(system_prompt: str, **_kw: Any) -> Any:
        # 断言入口把长期偏好注入了 system prompt；会话级 P_t 不该出现在这里（见下方 payload 断言）。
        captured["system_prompt"] = system_prompt

        class _FakeAgent:
            async def ainvoke(self, payload: Any, config: Any) -> dict[str, Any]:
                captured["payload"] = payload
                return {
                    "messages": [
                        HumanMessage(content="买个旅行包"),
                        _summary_msg(),
                        AIMessage(content="已为你整理好清单。"),
                    ]
                }

        return _FakeAgent()

    monkeypatch.setattr(main_agent, "_build_main_agent", _fake_build)

    # 假 curator：验证 run_agent→curator 接线（收到正确 query/final_text/prev_pt），并模拟它判定
    # 「喜欢小众品牌」为一贯取向、经**单一落库口**落库——从而 learned_preferences plumbing 被打通。
    async def _fake_curate(
        user_id: str,
        query: str,
        final_text: str,
        prev_pt: Any,
        session_domains: Any = None,
    ) -> None:
        captured["curate"] = (user_id, query, final_text, prev_pt.turn)
        # run_agent 必须把「清理前的快照」传进来：curator 跑在收尾 finally **之后**，那时
        # 品类域已从模块级 dict 里清掉，curator 自己去 ContextVar 读只会读到空——
        # 域兜底会静默失效（本次 code review 抓到的真 bug，别再回来）。
        captured["session_domains"] = session_domains
        await injector.persist_new_preferences(
            user_id,
            [
                SimpleNamespace(
                    slug="niche",
                    domain="",
                    content="喜欢小众品牌",
                    category="brand",
                    polarity="like",
                )
            ],
        )

    monkeypatch.setattr(main_agent, "curate_turn", _fake_curate)

    results: list[str] = []

    async def _fake_task_result(
        text: str, items: Any = None, elapsed_ms: Any = None, tokens: Any = None
    ) -> None:
        results.append(text)
        captured["items"] = items
        captured["elapsed_ms"] = elapsed_ms
        captured["tokens"] = tokens

    monkeypatch.setattr(main_agent.monitor, "report_task_result", _fake_task_result)

    # 观测未启用时 current_trace_id() 为 None；这里假装本轮挂上了 trace，验证它被带进返回值。
    monkeypatch.setattr(main_agent, "current_trace_id", lambda: "trace-abc")

    out = await main_agent.run_agent("买个旅行包", thread_id="t-1", user_id="user-9")

    # 出口带 trace_id：评测靠它把 Rubric 分数挂回这条 trace。**显式返回，不让下游读 ContextVar**
    # ——run_agent 在 API 层跑在 create_task 里，子 task 的 set 不回传父上下文，外面永远读到 None。
    assert out["trace_id"] == "trace-abc"
    # 出口：最终文案返回 + task_result 上报（带商品卡 items）。
    assert out["final_text"] == "已为你整理好清单。"
    assert results == ["已为你整理好清单。"]
    assert captured["items"] and captured["items"][0]["item_id"] == "A1"
    # task_result 带本轮耗时（毫秒，非负整数），供前端在该轮右下角显示「用时」。
    assert isinstance(captured["elapsed_ms"], int) and captured["elapsed_ms"] >= 0
    # 同一耗时也落进 messages 表末轮的 assistant（实时与回看同一口径；累加写，取 -1）。
    last_turn = (await read_turns("t-1"))[-1]
    assert last_turn["elapsed_ms"] == captured["elapsed_ms"]
    # token 用量同口径：task_result 下发的 tokens 与落库的一致（实时与回看一致）。本测用假模型、
    # 不驱动全树记账，故 snap=None → tokens 为 None、库里不写该键；真链路则为 dict。
    assert captured["tokens"] == last_turn.get("tokens")
    # 产物落盘：summary.md（取 shopping_summary 结构化 summary 字段）/ result.json 可供下载接口取。
    session_dir = main_agent.ensure_session_dir("t-1")
    assert (session_dir / "summary.md").read_text(encoding="utf-8") == "为你精选 1 件：帆布旅行包。"
    assert (session_dir / "result.json").exists()
    # 收尾跑 curator：收到本轮 query/final_text（prev_pt 首轮 turn=0）。
    assert captured["curate"] == ("user-9", "买个旅行包", "已为你整理好清单。", 0)
    # 快照必须真的传到了（不是 None）——否则 curator 读 ContextVar 会读到已被清空的状态。
    assert captured["session_domains"] is not None
    # 出口写回（现由 curator 经单一落库口完成）：新偏好落库 + 汇总进 learned_preferences。
    assert out["learned_preferences"] == ["喜欢小众品牌"]
    persisted = await store.read("user-9")
    assert any(p.content == "喜欢小众品牌" for p in persisted)
    # 入口注入位确实被渲染进 system prompt：长期偏好块在，但会话级 P_t 不该出现在这里
    # （挪去了当轮 human message 尾部，避免每轮必变的 P_t 打断 prompt cache 前缀）。
    assert "<user_long_term_preferences>" in captured["system_prompt"]
    assert "<session_preferences>" not in captured["system_prompt"]
    assert "不要塑料" not in captured["system_prompt"]
    # P_t 拼在 payload 里当轮的最后一条 human message 尾部，而不是 system prompt。
    last_role, last_content = captured["payload"]["messages"][-1]
    assert last_role == "user"
    assert "<session_constraints>" in last_content
    assert "不要塑料" in last_content
    assert last_content.endswith("买个旅行包")


async def test_run_agent_anonymous_skips_store(tmp_path: Path, monkeypatch: Any) -> None:
    store = _store()
    monkeypatch.setattr(injector, "get_store", lambda: store)

    def _fake_build(system_prompt: str, **_kw: Any) -> Any:
        class _FakeAgent:
            async def ainvoke(self, payload: Any, config: Any) -> dict[str, Any]:
                return {"messages": [HumanMessage(content="q"), _summary_msg()]}

        return _FakeAgent()

    monkeypatch.setattr(main_agent, "_build_main_agent", _fake_build)
    monkeypatch.setattr(main_agent.monitor, "report_task_result", _noop)
    # curator 顶成假的（不打真实 LLM）：匿名场景它内部也会因 user_id 空跳过长期落库。
    monkeypatch.setattr(main_agent, "curate_turn", _noop_curate)

    # 匿名（user_id=None）：本轮无长期偏好落库。
    out = await main_agent.run_agent("q", thread_id="t-2", user_id=None)
    assert out["learned_preferences"] == []


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def test_run_agent_uses_summary_text_when_zero_items(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """shopping_summary 零候选（如实道歉）时，展示给用户的 final_text 用它自己的干净文案，
    不采信收尾轮模型又加的自由文字——真实复现过：模型在这轮里混入 category_insight 的品类
    聚合数据给没找到的具体商品背书（如"这两款通常都是 4.3-4.7 星"），prompt 明文禁止但没有
    机制兜，靠这里的 items==0 短路把口子堵死。
    """
    store = _store()
    monkeypatch.setattr(injector, "get_store", lambda: store)

    empty_summary = ShoppingSummaryOutput(
        summary="抱歉，未找到该商品的可靠信息，建议您换个更具体的型号重新搜索。",
        items=[],
    )
    empty_summary_msg = ToolMessage(
        content=empty_summary.summary,
        name="shopping_summary",
        tool_call_id="c1",
        artifact=empty_summary,
    )

    def _fake_build(system_prompt: str, **_kw: Any) -> Any:
        class _FakeAgent:
            async def ainvoke(self, payload: Any, config: Any) -> dict[str, Any]:
                return {
                    "messages": [
                        HumanMessage(content="比比这两款耳机"),
                        empty_summary_msg,
                        AIMessage(
                            content=(
                                "这两款通常都是 4.3-4.7 星级别的产品，属于头部梯队"
                                "（品类聚合背书，不该出现在最终回复里）。"
                            )
                        ),
                    ]
                }

        return _FakeAgent()

    monkeypatch.setattr(main_agent, "_build_main_agent", _fake_build)
    monkeypatch.setattr(main_agent.monitor, "report_task_result", _noop)
    monkeypatch.setattr(main_agent, "curate_turn", _noop_curate)

    out = await main_agent.run_agent("比比这两款耳机", thread_id="t-empty", user_id=None)

    assert out["final_text"] == empty_summary.summary
    assert "4.3-4.7" not in out["final_text"]
    last_turn = (await read_turns("t-empty"))[-1]
    assert last_turn["content"] == empty_summary.summary


# ---------- Harness 控制面：pre_tool_call 硬闸 + post_tool_call 结果守卫 ----------
# 控制面已全部迁进 Hook Pipeline（ToolGuardMiddleware 不复存在）。这些测试驱动**唯一**的适配器
# HarnessAgentMiddleware.awrap_tool_call，等价于过去驱动 TGM.wrap_tool_call——区别是现在跑的是
# 真实的 Hook 链（各闸 + 截断 + 循环检测 + 提示），而不是一个类的方法。


def _mw(**guard_kw: Any) -> HarnessAgentMiddleware:
    """建一个只带控制面的中间件；guard_kw 调 GuardState 的口径（cap / 阈值）。"""
    setup_harness()
    return HarnessAgentMiddleware(guard=GuardState(**guard_kw))


_REQ_SEQ = itertools.count()


def _req(tool_name: str) -> Any:
    """最小假 ToolCallRequest：读 request.tool_call 的 name / args / id。

    args 每次生成唯一值：本文件的用例反复调同名工具是为了测**预算 / 循环 / 阶段闸**（按工具名
    判定），若 args 也相同会先被 tool_memo 的同参数回放截胡，闸根本轮不到执行。同参数回放
    自身的行为由 tests/test_tool_memo.py 覆盖。
    """
    args = {"probe": next(_REQ_SEQ)}
    return SimpleNamespace(tool_call={"name": tool_name, "args": args, "id": "c"})


def _handler_returning(content: str, tool_name: str) -> Any:
    """造一个假 async tool handler：忽略入参、固定返回一条 ToolMessage。"""

    async def handler(_req_arg: Any) -> ToolMessage:
        return ToolMessage(content=content, name=tool_name, tool_call_id="c")

    return handler


async def _call(mw: HarnessAgentMiddleware, tool_name: str, content: str = "结果") -> Any:
    return await mw.awrap_tool_call(_req(tool_name), _handler_returning(content, tool_name))


class _FakeModelReq:
    """最小假 ModelRequest：messages / system_message / tools + 链式 override。"""

    def __init__(self, messages: list[Any], tools: list[Any] | None = None) -> None:
        self.messages = messages
        self.system_message = None
        self.tools = tools or []

    def override(self, **kw: Any) -> _FakeModelReq:
        out = _FakeModelReq(kw.get("messages", self.messages), kw.get("tools", self.tools))
        out.system_message = kw.get("system_message", self.system_message)
        return out


async def test_tool_guard_truncates_long_result() -> None:
    mw = _mw(max_tool_tokens=50)  # 上限 > 提示语(~19 token)，才放得下提示
    long_text = "x" * 500  # ~63 token，超过上限
    out = await _call(mw, "item_search", long_text)
    assert isinstance(out, ToolMessage)
    assert len(out.content) < len(long_text)
    assert "已截断" in out.content


async def test_tool_guard_short_result_unchanged() -> None:
    mw = _mw()
    out = await _call(mw, "price_compare", "短结果")
    # 未超长、未触发循环 → 内容原样（Hook 没改写就不复制）。
    assert out.content == "短结果"


async def test_tool_guard_nudges_summary_after_picker() -> None:
    # item_picker 出口定点注入收尾提示：治「口头收尾」（item_picker 后没真调 shopping_summary）。
    mw = _mw()
    out = await _call(mw, "item_picker", "精选3件")
    assert "shopping_summary" in out.content
    assert out.content != "精选3件"  # 确有追加


async def test_tool_guard_nudges_on_loop() -> None:
    mw = _mw(loop_window=6, loop_threshold=4)
    last = None
    for _ in range(4):
        last = await _call(mw, "item_search", "结果")
    assert last is not None
    # 第 4 次同名工具命中阈值 → 追加系统提示推模型换思路/收尾。
    assert "[系统提示]" in last.content
    assert "item_search" in last.content


async def test_tool_guard_passes_through_non_toolmessage() -> None:
    mw = _mw()
    sentinel = object()  # 模拟 Command（控制流），不属截断范围

    async def handler(_req_arg: Any) -> Any:
        return sentinel

    assert await mw.awrap_tool_call(_req("planner"), handler) is sentinel


async def test_tool_guard_forces_convergence_over_retrieval_budget() -> None:
    mw = _mw(retrieval_cap=3, loop_threshold=99)  # 关掉循环检测，单测预算
    last = None
    for _ in range(4):  # 第 4 次检索超过 cap=3 → 软收敛（仍执行，结果带提示）
        last = await _call(mw, "item_search")
    assert last is not None
    assert "[强制收敛]" in last.content
    # 点名收尾要走的工具，给弱模型明确下一步。
    assert "price_compare" in last.content and "shopping_summary" in last.content


async def test_tool_guard_hard_blocks_after_soft_converge() -> None:
    # 再越预算就硬挡：检索不执行、回哨兵（连真实 handler 都不调）。
    called = {"n": 0}

    async def handler(_req_arg: Any) -> ToolMessage:
        called["n"] += 1
        return ToolMessage(content="真实结果", name="item_search", tool_call_id="c")

    mw = _mw(retrieval_cap=2, loop_threshold=99)
    outs = [await mw.awrap_tool_call(_req("item_search"), handler) for _ in range(5)]
    # cap=2：前 2 次 ok、第 3 次软收敛（共 3 次真调 handler），第 4/5 次硬挡（不调 handler）。
    assert called["n"] == 3
    assert "[检索预算耗尽]" in outs[3].content and "[检索预算耗尽]" in outs[4].content
    # 软收敛只出现一次（第 3 次），其后是硬挡。
    assert sum("[强制收敛]" in str(o.content) for o in outs) == 1


async def test_tool_guard_retrieval_budget_ignores_non_retrieval_tools() -> None:
    mw = _mw(retrieval_cap=2, loop_threshold=99)
    outs = [await _call(mw, "price_compare", "ok") for _ in range(5)]
    assert all(
        "[强制收敛]" not in str(o.content) and "[检索预算耗尽]" not in str(o.content) for o in outs
    )


async def test_tool_guard_budget_ignores_fork_dispatch() -> None:
    # fork 元工具不计入检索预算（跨平台检索走 fork，计入会让 fork 密集的正常轮误触发收敛）。
    mw = _mw(retrieval_cap=2, loop_threshold=99)
    outs = [await _call(mw, "parallel_dispatch_tool", "子任务结果") for _ in range(5)]
    assert all(
        "[强制收敛]" not in str(o.content) and "[检索预算耗尽]" not in str(o.content) for o in outs
    )


async def test_retrieval_budget_is_per_instance_not_shared() -> None:
    # 检索预算是 per-instance：不同 Agent 实例（主 / 各子）各自计数，互不影响。
    with fork_budget_scope():  # 即便开了树作用域，检索也只按 per-instance 计，不跨实例累加
        a = _mw(retrieval_cap=3, loop_threshold=99)
        b = _mw(retrieval_cap=3, loop_threshold=99)
        a_outs = [await _call(a, "item_search") for _ in range(3)]
        b_outs = [await _call(b, "item_search") for _ in range(3)]
    assert all("[强制收敛]" not in str(o.content) for o in a_outs + b_outs)
    assert all("[检索预算耗尽]" not in str(o.content) for o in a_outs + b_outs)


async def test_fork_budget_blocks_second_parallel_round() -> None:
    # 树级 fork 预算：跨平台并行 fork 只放行一轮，第二轮 re-fork 硬挡回哨兵。
    with fork_budget_scope():  # 默认 max_parallel=1
        a = _mw(loop_threshold=99)
        b = _mw(loop_threshold=99)  # 不同实例也共享同一棵树的 fork 预算
        first = await _call(a, "parallel_dispatch_tool", "子任务候选")
        second = await _call(b, "parallel_dispatch_tool", "子任务候选")
    assert "子任务候选" in first.content  # 第一轮放行（真执行子任务）
    assert "[禁止再 fork]" in second.content  # 第二轮硬挡
    # 耗尽理由是「并行轮已跑过」——文案必须提到跨平台，不能被串行耗尽的文案顶替。
    assert "跨平台" in second.content


async def test_fork_budget_serial_dispatch_capped_and_blocked_after_parallel() -> None:
    # 串行 dispatch_tool：并行轮之前可用 max_serial 次；超出即硬挡。
    with fork_budget_scope(max_parallel=1, max_serial=2):
        mw = _mw(loop_threshold=99)
        o1 = await _call(mw, "dispatch_tool", "候选")  # serial 1 → ok
        o2 = await _call(mw, "dispatch_tool", "候选")  # serial 2 → ok
        o3 = await _call(mw, "dispatch_tool", "候选")  # serial 3 > max_serial → 硬挡
    assert "候选" in o1.content and "候选" in o2.content
    assert "[禁止再 fork]" in o3.content
    # 这里耗尽的是纯串行配额，从未跑过并行轮——文案不能谎称「已并行 fork 过一轮跨平台检索」。
    assert "跨平台" not in o3.content


async def test_fork_budget_serial_exhausted_without_any_parallel_round() -> None:
    # target_refs 场景：全程只用串行 dispatch_tool，撞的是「串行配额耗尽」，跟跨平台泛搜无关。
    with fork_budget_scope(max_parallel=1, max_serial=1):
        mw = _mw(loop_threshold=99)
        o1 = await _call(mw, "dispatch_tool", "调查结果")
        o2 = await _call(mw, "dispatch_tool", "调查结果")
    assert "调查结果" in o1.content
    assert "[禁止再 fork]" in o2.content
    assert "跨平台" not in o2.content
    assert "独立子任务" in o2.content


# ---------- isolated_retrieval 接入（串行 dispatch_tool 打开隔离检索作用域）----------
# 不依赖真实 LLM：monkeypatch 掉 langchain.agents.create_agent，用一个假子 Agent 在 ainvoke
# 期间直接探测 ContextVar 是否处于隔离作用域，验证 _run_sub_agent 的 isolated_retrieval 接线。
class _FakeSubAgent:
    def __init__(self, on_ainvoke: Any) -> None:
        self._on_ainvoke = on_ainvoke

    async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self._on_ainvoke()
        return {"messages": [AIMessage(content="子任务结果")]}


async def test_run_sub_agent_isolated_retrieval_active_for_serial_dispatch(
    monkeypatch: Any,
) -> None:
    from app.agent.dispatch_tool import _run_sub_agent
    from app.agent.retrieval_budget import _isolated_var

    captured: dict[str, bool] = {}
    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda **_kw: _FakeSubAgent(lambda: captured.__setitem__("isolated", _isolated_var.get())),
    )

    result = await _run_sub_agent("demands", lambda: [], isolated_retrieval=True)
    assert captured["isolated"] is True  # 串行 dispatch_tool：隔离作用域确实激活
    assert "子任务结果" in result
    assert _isolated_var.get() is False  # 离开 _run_sub_agent 后正确还原


async def test_run_sub_agent_isolated_retrieval_inactive_for_parallel_style(
    monkeypatch: Any,
) -> None:
    from app.agent.dispatch_tool import _run_sub_agent
    from app.agent.retrieval_budget import _isolated_var

    captured: dict[str, bool] = {}
    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda **_kw: _FakeSubAgent(lambda: captured.__setitem__("isolated", _isolated_var.get())),
    )

    # isolated_retrieval 不传，默认 False——parallel_dispatch_tool 内部调用走的就是这条默认值。
    result = await _run_sub_agent("demands", lambda: [])
    assert captured["isolated"] is False  # 维持全树共享语义，不隔离
    assert "子任务结果" in result


# ---------- parallel_dispatch_tool 按 demands 内容机制识别隔离场景（切回并行后新增）----------
# 定点商品调查从"串行 dispatch_tool"改成"一次 parallel_dispatch_tool"后，isolated_retrieval
# 不能再靠调用方是谁（dispatch_tool vs parallel_dispatch_tool）区分，得靠 demands 里有没有
# 平台名机制识别：有平台名＝跨平台泛搜（共享收敛信号）、没有＝点名商品定点调查（各自隔离）。


async def test_parallel_dispatch_tool_isolates_non_platform_batch(monkeypatch: Any) -> None:
    from app.agent.dispatch_tool import make_dispatch_tools
    from app.agent.retrieval_budget import _isolated_var

    captured: list[bool] = []
    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda **_kw: _FakeSubAgent(lambda: captured.append(_isolated_var.get())),
    )

    _, parallel_dispatch_tool = make_dispatch_tools(lambda: [])
    result = await parallel_dispatch_tool.ainvoke(
        {
            "demands_list": [
                "调查这个具体商品：Sony WH-1000XM5。品类=降噪耳机",
                "调查这个具体商品：Bose QC45。品类=降噪耳机",
            ]
        }
    )
    # 没有平台名 → 机制判定为定点调查批次，两个子任务各自隔离（不共享 web_search 门控信号）。
    assert captured == [True, True]
    assert result.count("子任务结果") == 2


async def test_parallel_dispatch_tool_shares_signal_for_platform_batch(monkeypatch: Any) -> None:
    from app.agent.dispatch_tool import make_dispatch_tools
    from app.agent.retrieval_budget import _isolated_var

    captured: list[bool] = []
    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda **_kw: _FakeSubAgent(lambda: captured.append(_isolated_var.get())),
    )

    from app.agent.platform_scope import platform_scope

    _, parallel_dispatch_tool = make_dispatch_tools(lambda: [])
    # 用户勾了 3 个平台（多平台比价场景）；模型只列了 1 个 → 机制补齐到这 3 个（不再是写死的 5 个：
    # 未启用的平台不派，见 platform_scope）。
    with platform_scope(["amazon", "shein", "walmart"]):
        await parallel_dispatch_tool.ainvoke(
            {"demands_list": ["在 amazon 上检索：品类=降噪耳机，预算≈$300"]}
        )
    # 有平台名 → 机制判定为跨平台泛搜批次，维持全树共享 web_search 门控信号，不隔离
    # （回归保护：这是原有行为，不该被本次改动波及）。
    assert len(captured) == 3
    assert all(v is False for v in captured)


async def test_fork_budget_unbounded_without_scope() -> None:
    # 无树作用域（单测 / 无 fork 树）则不对 fork 设限：连发多轮 parallel_dispatch 都放行。
    mw = _mw(loop_threshold=99)
    outs = [await _call(mw, "parallel_dispatch_tool", "候选") for _ in range(3)]
    assert all("[禁止再 fork]" not in str(o.content) for o in outs)


# ---------- 深度闸 / web_search 兜底门 / 全树检索预算（机制兜职责边界，不靠 prompt）----------
async def test_depth_gate_blocks_aggregation_tools_in_sub() -> None:
    # 聚合/终结工具需要跨平台全局视图：主 loop(depth 0)放行，子 Agent(depth≥1)硬挡（权限闸）。
    from app.agent.fork_guard import enter_fork

    mw = _mw(loop_threshold=99)
    out0 = await _call(mw, "shopping_summary", "收尾")
    assert "收尾" in out0.content  # depth 0：放行
    with enter_fork():  # depth → 1
        for name in ("price_compare", "shipping_calc", "item_picker", "shopping_summary"):
            out = await _call(mw, name, "X")
            assert "[子任务无权收尾]" in out.content
        # 能力同质：子的检索工具不受深度闸影响，照常放行。
        ok = await _call(mw, "item_search", "候选")
        assert "候选" in ok.content


async def test_depth_gate_blocks_platform_agnostic_context_tools_in_sub() -> None:
    # 平台无关上下文工具（planner / category_insight）也钉成 depth==0 only。
    from app.agent.fork_guard import enter_fork

    mw = _mw(loop_threshold=99)
    out0 = await _call(mw, "planner", "拆解")
    assert "拆解" in out0.content
    with enter_fork():  # depth → 1
        for name in ("planner", "category_insight"):
            out = await _call(mw, name, "X")
            assert "[子任务无需此工具]" in out.content
        ok = await _call(mw, "item_search", "候选")
        assert "候选" in ok.content


# ---------- cache 前缀稳定：工具表恒定，所有禁用统一在执行层哨兵 ----------
async def test_model_call_never_touches_tool_list() -> None:
    """控制面从不改工具表——所有禁用都是执行层哨兵，工具 schema 每轮字节稳定（保 prompt cache）。

    过去这条不变式由「TGM.wrap_model_call 不摘工具」保证；现在整条模型侧只剩 pre_think Hook
    （预算 hint + 压缩），压根不碰 request.tools。这里钉死它：无论子搜满 cap、还是主 loop fork
    过一轮，送进模型的工具表都原样。
    """
    from app.agent.fork_guard import enter_fork

    seen: list[list[str]] = []

    # 带 tool_calls：避免触发 terminal_enforcer 的重发，把变量隔离到「工具表变没变」这一件事上。
    ai = AIMessage(content="", tool_calls=[{"name": "item_search", "args": {}, "id": "t1"}])

    async def handler(req: Any) -> Any:
        seen.append([t.name for t in req.tools])
        return SimpleNamespace(result=[ai])

    tools = [SimpleNamespace(name="item_search"), SimpleNamespace(name="price_compare")]

    mw = _mw(loop_threshold=99)
    mw._guard.item_search_calls = SUB_ITEM_SEARCH_CAP  # 模拟已搜满
    with enter_fork():  # depth 1（子）
        await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="q")], tools), handler)

    with fork_budget_scope() as budget:  # depth 0（主 loop），已 fork 过一轮
        budget.charge("parallel_dispatch_tool")
        mw2 = _mw(loop_threshold=99)
        await mw2.awrap_model_call(_FakeModelReq([HumanMessage(content="q")], tools), handler)

    assert seen == [["item_search", "price_compare"]] * 2


async def test_tool_call_blocks_sub_item_search_over_cap() -> None:
    # 执行层硬闸：子搜满 cap 后真搜不动。
    from app.agent.fork_guard import enter_fork

    mw = _mw(loop_threshold=99)
    with enter_fork():  # depth 1
        outs = [await _call(mw, "item_search", "候选") for _ in range(SUB_ITEM_SEARCH_CAP + 2)]
    assert all("候选" in o.content for o in outs[:SUB_ITEM_SEARCH_CAP])  # 前 cap 次执行
    assert all("本平台检索已用满" in o.content for o in outs[SUB_ITEM_SEARCH_CAP:])  # 其后硬挡


async def test_tool_call_blocks_main_item_search_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 执行层硬闸：主 loop fork 后直接 item_search 也真拦下（堵 balloon-squeeze）。
    # 拦截以「候选池非空」为前提（结果感知解锁）：整轮 fork 失败/全空时哨兵那句
    # 「候选已汇集」是假话，直搜是仅剩的补救通路——该场景在 test_harness 的
    # TestPostforkGateRelease 覆盖，这里 patch 出有货的池子验证正常拦截语义。
    from app.harness.hooks import tool_gates

    monkeypatch.setattr(tool_gates, "candidate_count", lambda: 5)
    mw = _mw(loop_threshold=99)
    with fork_budget_scope() as budget:  # depth 0
        before = await _call(mw, "item_search", "候选")
        assert "候选" in before.content  # fork 前可搜
        budget.charge("parallel_dispatch_tool")  # 跨平台并行 fork 跑过一轮
        after = await _call(mw, "item_search", "候选")
        assert "检索阶段已结束" in after.content  # fork 后硬挡


async def test_terminal_reached_blocks_further_tools() -> None:
    # over-loop 治理（终结硬停）：主 loop 调过终结工具收尾后，后续任何工具都被拦。
    mw = _mw(loop_threshold=99)
    done = await _call(mw, "shopping_summary", "清单")
    assert "清单" in done.content  # 终结工具本身正常放行
    for name in ("item_search", "item_picker", "shopping_summary"):
        out = await _call(mw, name, "X")
        assert "本轮已收尾" in out.content


async def test_terminal_hardstop_scoped_to_main_loop() -> None:
    # 只对主 loop（depth 0）：子 Agent（depth≥1）不受终结硬停闸约束。
    from app.agent.fork_guard import enter_fork

    mw = _mw(loop_threshold=99)
    with enter_fork():  # depth 1（子）
        mw._guard.terminal_reached = True  # 即便标志为真，子 loop 也不受该闸约束
        ok = await _call(mw, "item_search", "候选")
        assert "候选" in ok.content


async def test_sub_item_search_result_shows_budget_note() -> None:
    # 预算可见：子的 item_search 结果尾部带「还能搜几次 + 找够用就停」批注。
    from app.agent.fork_guard import enter_fork

    mw = _mw(loop_threshold=99)
    with enter_fork():  # depth 1
        out = await _call(mw, "item_search", "候选JSON")
    assert "候选JSON" in out.content
    assert "检索预算" in out.content  # 预算可见批注在


async def test_websearch_gate(tmp_path: Path) -> None:
    # web_search 门控：独立知识查询放行 / 空召回兜底放行 / 有候选时拦。
    from app.agent.retrieval_budget import note_item_search, reset_tree
    from app.utils.thread_ctx import thread_scope

    mw = _mw(loop_threshold=99)
    with thread_scope("t-web", tmp_path):
        try:
            standalone = await _call(mw, "web_search", "外部")
            assert "外部" in standalone.content  # 还没 item_search → 独立知识查询，放行
            note_item_search(0)  # 召回为空
            ok = await _call(mw, "web_search", "兜底线索")
            assert "兜底线索" in ok.content  # 空召回 → 兜底放行
            note_item_search(5)  # 有命中
            blk = await _call(mw, "web_search", "X")
            assert "[web_search 未执行]" in blk.content  # 有候选 → 拦
        finally:
            reset_tree()


async def test_tree_retrieval_budget_shared_across_instances(tmp_path: Path) -> None:
    # 全树检索预算：主 + 各子（不同中间件实例）的 item_search 计进同一 session_dir 计数。
    from app.agent.retrieval_budget import reset_tree
    from app.utils.thread_ctx import thread_scope

    with thread_scope("t-tree", tmp_path):
        try:
            a = _mw(tree_retrieval_cap=3, loop_threshold=99)
            b = _mw(tree_retrieval_cap=3, loop_threshold=99)  # 模拟主 + 子
            outs = [
                await _call(a, "item_search", "候选"),  # 全树 1
                await _call(b, "item_search", "候选"),  # 2
                await _call(a, "item_search", "候选"),  # 3
                await _call(b, "item_search", "候选"),  # 4 → 软收敛
                await _call(a, "item_search", "候选"),  # 5 → 硬挡
            ]
        finally:
            reset_tree()
    assert "[强制收敛]" in outs[3].content
    assert "[检索预算耗尽]" in outs[4].content


def test_truncate_tool_result_tiny_cap_never_inverts() -> None:
    # cap 比提示语还短：不能反而把结果撑长，退化为硬截。
    out = truncate_tool_result("x" * 500, max_tokens=1)  # cap=4 < hint
    assert len(out) <= 500


# ---------- 主 loop 必须走终结工具才能收尾（P0 收尾纪律，机制兜底而非只靠 prompt）----------
# 现在是 post_reflect 的 terminal_enforcer Hook 置 retry_nudge，适配器据此当场重发一次模型。
# 必须当场重发而不能等下一轮：模型没产出 tool_call，Agent 框架据此判定该结束了，根本没有下一轮。


def _responder(*responses: Any) -> tuple[Any, list[Any]]:
    """造一个 async model handler，依次返回给定响应；返回 (handler, 收到的 request 列表)。"""
    calls: list[Any] = []

    async def handler(req: Any) -> Any:
        calls.append(req)
        idx = min(len(calls) - 1, len(responses) - 1)
        return SimpleNamespace(result=[responses[idx]])

    return handler, calls


async def test_model_call_forces_terminal_tool_when_missing_at_depth_zero() -> None:
    # depth 0、没调终结工具就想吐文字收尾 → 重发一次，第二次响应带 tool_calls 就放行。
    no_tool_calls = AIMessage(content="这是我的对比结论，没调工具")
    with_tool_calls = AIMessage(
        content="", tool_calls=[{"name": "shopping_summary", "args": {}, "id": "tc1"}]
    )
    handler, calls = _responder(no_tool_calls, with_tool_calls)
    mw = _mw(loop_threshold=99)
    out = await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="比较一下")]), handler)
    assert len(calls) == 2
    # 首答 + nudge 一并落 state（曾被丢弃 → 下一轮 prompt 缺这两条，斩断前缀缓存链），
    # 生效的回复恒为最后一条 AIMessage。
    assert out.result[-1] is with_tool_calls
    assert out.result[0] is no_tool_calls
    assert any(
        isinstance(m, HumanMessage) and "必须真的调用工具" in m.content for m in out.result
    )
    nudge_texts = [m.content for m in calls[1].messages if isinstance(m, HumanMessage)]
    assert any("必须真的调用工具" in t for t in nudge_texts)


async def test_model_call_gives_up_after_max_retries() -> None:
    # 同一次模型调用内一直不调工具 → 重发到上限后放弃，原样返回最后一次响应（不无限重试）。
    no_tool_calls = AIMessage(content="还是没调工具")
    handler, calls = _responder(no_tool_calls)
    mw = _mw(loop_threshold=99)
    out = await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="比较一下")]), handler)
    assert len(calls) == 1 + MAX_TERMINAL_NUDGE_RETRIES
    assert out.result[0] is no_tool_calls


async def test_terminal_nudge_quota_is_per_model_call_not_per_loop() -> None:
    """配额每次模型调用重置：第 3 轮想纯文字收尾被催过，第 12 轮再想蒙混同样得被催。

    回归保护：把计数器放进 GuardState 却不按调用清零，会让整个 loop 只催一次——长对话里
    P0 收尾纪律形同虚设。
    """
    no_tool_calls = AIMessage(content="还是没调工具")
    mw = _mw(loop_threshold=99)

    handler1, calls1 = _responder(no_tool_calls)
    await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="第一次")]), handler1)
    assert len(calls1) == 1 + MAX_TERMINAL_NUDGE_RETRIES

    handler2, calls2 = _responder(no_tool_calls)
    await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="很多轮之后")]), handler2)
    assert len(calls2) == 1 + MAX_TERMINAL_NUDGE_RETRIES, "后续轮次不再催 → P0 收尾纪律被削弱"


async def test_model_call_skips_nudge_when_terminal_tool_already_called() -> None:
    # 历史里已经调过终结工具：这是它之后的自然收尾文字，不该被误伤重发。
    no_tool_calls = AIMessage(content="收尾闲聊")
    handler, calls = _responder(no_tool_calls)
    history = [
        HumanMessage(content="比较一下"),
        ToolMessage(content="已收尾", name="shopping_summary", tool_call_id="c1"),
    ]
    mw = _mw(loop_threshold=99)
    out = await mw.awrap_model_call(_FakeModelReq(history), handler)
    assert len(calls) == 1
    assert out.result[0] is no_tool_calls


async def test_model_call_skips_nudge_for_sub_agent() -> None:
    # 子 Agent（depth≥1）正常收尾就是直接吐调查结果文本，不该被拖去调不属于自己权限的终结工具。
    from app.agent.fork_guard import enter_fork

    no_tool_calls = AIMessage(content="子任务调查结果")
    handler, calls = _responder(no_tool_calls)
    mw = _mw(loop_threshold=99)
    with enter_fork():  # depth → 1
        out = await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="调查")]), handler)
    assert len(calls) == 1
    assert out.result[0] is no_tool_calls


async def test_model_call_no_nudge_when_response_has_tool_calls() -> None:
    # 回归保护：响应本来就带 tool_calls 的正常路径不受影响。
    with_tool_calls = AIMessage(
        content="", tool_calls=[{"name": "item_search", "args": {}, "id": "tc1"}]
    )
    handler, calls = _responder(with_tool_calls)
    mw = _mw(loop_threshold=99)
    out = await mw.awrap_model_call(_FakeModelReq([HumanMessage(content="比较一下")]), handler)
    assert len(calls) == 1
    assert out.result[0] is with_tool_calls


async def test_sync_paths_are_disabled() -> None:
    # 全链路 async：同步路径会绕过整个控制面，必须显式失败而不是静默放行。
    mw = _mw()
    with pytest.raises(NotImplementedError, match="async"):
        mw.wrap_model_call(_FakeModelReq([]), lambda r: None)
    with pytest.raises(NotImplementedError, match="async"):
        mw.wrap_tool_call(_req("item_search"), lambda r: None)


# ---------- build_agent_middleware ----------
def test_build_agent_middleware_shape_and_freshness() -> None:
    """控制面全在 Hook 里 → 栈里只剩唯一的适配器；每次新建（GuardState 有状态，不能跨实例复用）。"""
    stack1 = build_agent_middleware()
    stack2 = build_agent_middleware()
    assert [type(m).__name__ for m in stack1] == ["HarnessAgentMiddleware"]
    assert stack1[0] is not stack2[0]
    assert stack1[0]._guard is not stack2[0]._guard
