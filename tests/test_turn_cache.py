"""整轮结果缓存的单测（批2-5）：键的四个成分、不入缓存的两类轮次、以及主链路上的命中回放。

集成部分沿用 tests/test_orchestrator.py 那套打桩（假 Agent + 顶掉 build_main_agent），判据是
**build_main_agent 被调了几次**——「命中了」只能这么证：断言返回文案相同是假绿（真跑一遍也相同）。
"""

from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import Msg, TextBlock, ToolCallBlock
from agentscope.state import AgentState

import app.agent.orchestrator as orch
from app.recall import semantic_cache as sc


# ---------- 键的四个成分 ----------
def test_key_changes_with_every_component(monkeypatch: pytest.MonkeyPatch) -> None:
    base = sc.turn_cache_key(buyer="u1", prefs_fp="p1", query="买个包", model="m1")
    assert base == sc.turn_cache_key(buyer="u1", prefs_fp="p1", query="买个包", model="m1")
    assert base != sc.turn_cache_key(buyer="u2", prefs_fp="p1", query="买个包", model="m1")
    assert base != sc.turn_cache_key(buyer="u1", prefs_fp="p2", query="买个包", model="m1")
    assert base != sc.turn_cache_key(buyer="u1", prefs_fp="p1", query="买个箱", model="m1")
    assert base != sc.turn_cache_key(buyer="u1", prefs_fp="p1", query="买个包", model="m2")
    assert base != sc.turn_cache_key(
        buyer="u1", prefs_fp="p1", query="买个包", model="m1", prompt_version="1.1.0"
    )


def test_key_changes_when_prompts_yml_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """改了提示词要自动失效整片缓存——否则调完 prompt 重跑看到的还是旧行为，会把人引到错误结论。"""
    monkeypatch.setattr(sc, "_prompts_fingerprint", lambda: "aaaa")
    a = sc.turn_cache_key(buyer="u1", prefs_fp="p", query="q", model="m")
    monkeypatch.setattr(sc, "_prompts_fingerprint", lambda: "bbbb")
    assert a != sc.turn_cache_key(buyer="u1", prefs_fp="p", query="q", model="m")


def test_preference_fingerprint_is_order_independent() -> None:
    """库里的返回顺序不保证稳定；不排序就会「同一组偏好算出两个指纹」= 缓存永远不命中且不报错。"""
    a = SimpleNamespace(dedup_key="neg:material:bag:leather", content="不要皮革", is_blocking=True)
    b = SimpleNamespace(dedup_key="pos:style:bag:minimal", content="喜欢极简", is_blocking=False)
    assert sc.preference_fingerprint([a, b]) == sc.preference_fingerprint([b, a])
    assert sc.preference_fingerprint([a]) != sc.preference_fingerprint([a, b])


# ---------- 哪些轮次不许入缓存 ----------
@pytest.mark.parametrize("tool", sorted(sc.UNCACHEABLE_TOOLS))
def test_write_and_interactive_turns_are_not_cacheable(tool: str) -> None:
    assert sc.turn_is_cacheable([tool], "清单在此") is False
    assert sc.turn_is_cacheable(["item_search", tool], "清单在此") is False


def test_read_only_turn_is_cacheable() -> None:
    assert sc.turn_is_cacheable(["planner", "item_search", "shopping_summary"], "清单") is True


def test_empty_answer_is_not_cacheable() -> None:
    """空回复存下来就是把一次失败固化成「秒回的失败」。"""
    assert sc.turn_is_cacheable(["item_search"], "   ") is False


# ---------- 状态与开关 ----------
def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURN_CACHE_ENABLED", raising=False)
    assert sc.turn_cache_enabled() is False
    assert sc.turn_cache_status() == {"enabled": False, "entries": 0}


def test_status_reports_entries_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURN_CACHE_ENABLED", "1")
    sc.reset_turn_cache()
    sc.get_turn_cache().put("k", sc.TurnCacheEntry(final_text="t", items=[]))
    assert sc.turn_cache_status() == {"enabled": True, "entries": 1}
    sc.reset_turn_cache()


def test_capacity_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = sc.TurnCache(max_entries=2, ttl=100.0)
    for i in range(3):
        cache.put(f"k{i}", sc.TurnCacheEntry(final_text=str(i), items=[]))
    assert len(cache) == 2 and cache.get("k0") is None


# ---------- 主链路：命中就不跑模型 ----------
def _fake_agent(final_text: str, *, context: list[Msg] | None = None) -> Any:
    class _FakeAgent:
        def __init__(self) -> None:
            self.state = AgentState()
            self.state.context = context if context is not None else []

        async def reply_stream(self, inputs: Any, yield_final_msg: bool = False) -> Any:
            yield Msg(
                name="shoppingx",
                role="assistant",
                content=[TextBlock(type="text", text=final_text)],
            )

    return _FakeAgent()


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Any:
    """顶掉打网络 / 打库的收尾环节，并数 build_main_agent 被调了几次。"""
    calls = {"builds": 0, "results": []}
    ctx: list[Msg] = []

    async def _build(**_kw: Any) -> Any:
        calls["builds"] += 1
        return _fake_agent("为你精选 1 件：帆布旅行包。", context=ctx), SimpleNamespace()

    async def _noop_curate(*_a: Any, **_kw: Any) -> None:
        return None

    async def _task_result(text: str, **kw: Any) -> None:
        calls["results"].append((text, kw.get("items")))

    monkeypatch.setattr(orch, "build_main_agent", _build)
    monkeypatch.setattr(orch, "curate_turn", _noop_curate)
    monkeypatch.setattr(orch.monitor, "report_task_result", _task_result)
    monkeypatch.setenv("TURN_CACHE_ENABLED", "1")
    sc.reset_turn_cache()
    yield calls, ctx
    sc.reset_turn_cache()


async def test_hit_across_threads_skips_the_model(wired: Any) -> None:
    """同一个人问同一句（另起一个 thread）→ 直接回放，一次模型调用都不发起。压测就靠它。"""
    calls, _ctx = wired
    await orch.run_agent("买个旅行包", thread_id="tc-a1")
    out = await orch.run_agent("买个旅行包", thread_id="tc-a2")

    assert calls["builds"] == 1  # 第二轮压根没建 Agent
    assert out["cached"] is True
    assert out["final_text"] == "为你精选 1 件：帆布旅行包。"
    assert len(calls["results"]) == 2  # 命中轮照样发 task_result（对前端透明）


async def test_second_turn_in_same_thread_never_consults_cache(wired: Any) -> None:
    """带上文的轮次：上文不在 key 里，命中就是串味，所以既不查也不写。"""
    calls, _ctx = wired
    await orch.run_agent("买个旅行包", thread_id="tc-b1")
    await orch.run_agent("买个旅行包", thread_id="tc-b1")  # 同 thread，已有历史
    assert calls["builds"] == 2


async def test_write_intent_turn_is_not_stored(wired: Any) -> None:
    """本轮出现过写工具 → 不入缓存。判据是 tool_call 块（表达了写意图就算），不是有没有结果。"""
    calls, ctx = wired
    ctx.append(
        Msg(
            name="shoppingx",
            role="assistant",
            content=[ToolCallBlock(type="tool_call", id="c1", name="create_order", input="{}")],
        )
    )
    await orch.run_agent("把第二个买了", thread_id="tc-c1")
    await orch.run_agent("把第二个买了", thread_id="tc-c2")
    assert calls["builds"] == 2  # 第二轮老老实实又跑了一遍


async def test_disabled_switch_runs_every_time(monkeypatch: pytest.MonkeyPatch, wired: Any) -> None:
    calls, _ctx = wired
    monkeypatch.setenv("TURN_CACHE_ENABLED", "0")
    await orch.run_agent("买个旅行包", thread_id="tc-d1")
    await orch.run_agent("买个旅行包", thread_id="tc-d2")
    assert calls["builds"] == 2


# ---------- 评测拒跑闸 ----------
def _load_rubric_guard() -> Any:
    """按路径加载评测脚本里的那道闸（脚本不是包，不能直接 import）。"""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "run_rubric.py"
    spec = importlib.util.spec_from_file_location("_run_rubric_probe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_rubric_refuses_when_turn_cache_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """开着缓存跑评测，分数是「上一次那份的复读」且全程零报错——所以拒跑，不是警告。"""
    mod = _load_rubric_guard()
    monkeypatch.setenv("TURN_CACHE_ENABLED", "1")
    with pytest.raises(SystemExit) as e:
        mod._assert_turn_cache_off()
    assert "TURN_CACHE_ENABLED" in str(e.value)


def test_run_rubric_runs_when_cache_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_rubric_guard()
    monkeypatch.delenv("TURN_CACHE_ENABLED", raising=False)
    monkeypatch.delenv(mod.HEALTH_URL_ENV, raising=False)
    mod._assert_turn_cache_off()  # 不抛即通过


def test_run_rubric_refuses_when_remote_health_says_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """后端另起一个进程时，本进程开关是关的也不作数——配了 EVAL_HEALTH_URL 就去问那一边。"""
    import httpx

    mod = _load_rubric_guard()
    monkeypatch.delenv("TURN_CACHE_ENABLED", raising=False)
    monkeypatch.setenv(mod.HEALTH_URL_ENV, "http://backend/api/health")

    def _fake_get(url: str, timeout: float = 5.0) -> Any:
        return httpx.Response(200, json={"status": "ok", "turn_cache": {"enabled": True}})

    monkeypatch.setattr(httpx, "get", _fake_get)
    with pytest.raises(SystemExit) as e:
        mod._assert_turn_cache_off()
    assert "后端整轮缓存开着" in str(e.value)


def test_run_rubric_does_not_block_when_health_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """探不到就别拦：本进程那道判据已经把住了主路，为一次网络抖动挡下整批评测不划算。"""
    import httpx

    mod = _load_rubric_guard()
    monkeypatch.delenv("TURN_CACHE_ENABLED", raising=False)
    monkeypatch.setenv(mod.HEALTH_URL_ENV, "http://backend/api/health")

    def _boom(url: str, timeout: float = 5.0) -> Any:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", _boom)
    mod._assert_turn_cache_off()
