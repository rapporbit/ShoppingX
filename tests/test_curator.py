"""记忆管家 curator（``app.memory.curator``）的确定性测试（假 LLM，无真实模型）。

刀1（P_t 单写者重构）后 curator 只剩一件事——判「一贯取向」并经单一落库口写**长期库**：
- persistent_preferences → 仅登录用户落库；本轮约束什么都不产（planner 的活）。
- 匿名用户:整个跳过（长期库不写、P_t 不归它管），连 LLM 都不调。
- LLM 调用失败:降级返回 None、长期库不写，绝不抛（它在主回复下发后跑，不该反噬主链路）。
- **不碰 P_t**:没有 session_dir 参数、不 import merge_pt/save_pt——这是签名层面的保证。
"""

from __future__ import annotations

from typing import Any

import app.memory.curator as curator
import app.memory.injector as injector
from app.memory.curator import (
    CurationResult,
    _PersistentPref,
    curate_turn,
)
from app.memory.session_state import SessionPrefState
from app.memory.store import PreferenceStore, get_store


def _store() -> PreferenceStore:
    return get_store()


class _FakeStructured:
    def __init__(self, payload: Any, calls: list[Any] | None = None) -> None:
        self._payload = payload
        self._calls = calls

    async def ainvoke(self, messages: Any) -> Any:
        if self._calls is not None:
            self._calls.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeLLM:
    def __init__(self, payload: Any, calls: list[Any] | None = None) -> None:
        self._payload = payload
        self._calls = calls

    def with_structured_output(self, _schema: Any, **kwargs: Any) -> _FakeStructured:
        self.structured_kwargs = kwargs
        return _FakeStructured(self._payload, self._calls)


def _patch_llm_and_store(
    monkeypatch: Any, payload: Any, store: PreferenceStore, calls: list[Any] | None = None
) -> None:
    monkeypatch.setattr(curator, "get_fast_llm", lambda: _FakeLLM(payload, calls))
    # curator 读长期偏好 + persist 落库都走 get_store()——两处 import 各自 patch 到同一个测试 store。
    monkeypatch.setattr(curator, "get_store", lambda: store)
    monkeypatch.setattr(injector, "get_store", lambda: store)


# ---------- 只沉淀一贯取向，本轮约束不产出 ----------
async def test_curate_persists_only_long_term(monkeypatch: Any) -> None:
    """curator 判「一贯取向」落长期库；本轮约束（蓝色 / 预算）它无字段可产、自然不落。"""
    store = _store()
    payload = CurationResult(
        persistent_preferences=[
            _PersistentPref(
                content="讨厌塑料材质",
                category="material",
                slug="plastic",
                domain="footwear",
                polarity="dislike",
                keywords=["塑料", "plastic"],
            ),
        ]
    )
    _patch_llm_and_store(monkeypatch, payload, store)

    result = await curate_turn(
        "user-1",
        "这次预算300，我一直讨厌塑料，今天想要蓝色",
        "已为你精选。",
        SessionPrefState(budget_usd=42.0),  # 只读参考：渲染给 LLM 看，不据此写任何东西
    )
    assert result is not None

    # 一贯取向（塑料）进长期库，且**只有**这一条（蓝色 / 预算不该出现在长期库里）
    entries = await store.read("user-1")
    assert [e.dedup_key for e in entries] == ["dislike:material:footwear:plastic"]


# ---------- 匿名:整个跳过，LLM 都不调 ----------
async def test_curate_anonymous_skips_entirely(monkeypatch: Any) -> None:
    store = _store()
    calls: list[Any] = []
    payload = CurationResult(
        persistent_preferences=[
            _PersistentPref(content="讨厌塑料", category="material", slug="plastic")
        ]
    )
    _patch_llm_and_store(monkeypatch, payload, store, calls)

    result = await curate_turn("", "讨厌塑料", "已为你精选。", SessionPrefState())

    assert result is None  # 匿名直接短路
    assert calls == []  # 连 LLM 都没调（P_t 不归它管、长期库不落匿名，无事可做）
    assert await store.read("") == []


# ---------- LLM 失败:降级不崩，长期库不写 ----------
async def test_curate_llm_failure_degrades(monkeypatch: Any) -> None:
    store = _store()
    _patch_llm_and_store(monkeypatch, RuntimeError("boom"), store)

    result = await curate_turn("u", "q", "a", SessionPrefState())
    assert result is None  # 降级
    assert await store.read("u") == []  # 长期库未动


# ---------- 漏填 domain → 兜底到 planner 本轮判定的域（而不是 global）----------
def test_missing_domain_falls_back_to_session_domain() -> None:
    """curator 漏判 domain（落 other）时，用 planner 本轮判的域兜底。

    ``other`` 是保守档——它谁也匹配不上，等于这条偏好几乎不会生效。而本轮 planner 已经判过
    「在买鞋」了，那就是这条偏好最可能归属的域，比 other 强得多。

    **绝不兜到 global**：那是激进档（全局生效、跨品类杀商品）。漏填的默认值绝不能是它——
    这正是改造前的坑（domain 留空即全局，于是买鞋时说的「不喜欢皮革」在买沙发时也生效）。
    """
    prefs = [
        curator._PersistentPref(content="喜欢小众设计", category="style", slug="niche"),
        curator._PersistentPref(
            content="我素食，任何皮革都不要", category="material", slug="leather", domain="global"
        ),
    ]

    curator._scope_to_session_domain(prefs, ["footwear"])

    assert prefs[0].domain == "footwear"  # 漏填（other）→ 兜到本轮域
    assert prefs[1].domain == "global"  # 已明确判为跨品类底线 → 不动


def test_domain_fallback_noop_without_session_domain() -> None:
    """本轮判不出域（闲聊轮 / planner 没跑）时保持 other——没有依据就别猜。"""
    prefs = [curator._PersistentPref(content="喜欢小众", category="style", slug="niche")]

    curator._scope_to_session_domain(prefs, [])

    assert prefs[0].domain == "other"
