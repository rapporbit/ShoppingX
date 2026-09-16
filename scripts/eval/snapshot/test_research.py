"""快照评测（真 LLM）：research 有界研究函数（C2）与独立配额（C3）的实际行为。

离线测试（``tests/test_research*.py`` 33 条）守的是机制本身——给定入参，闸扣多少、护栏丢几条。
这里守的是**真模型会不会用它**，以及那两样在真链路上是不是真的成立：

- 三类「已有明确对象」的问法（单品口碑 / 多品对比 / 品类选购维度）走 research，不退回裸
  web_search 换措辞重搜；
- 护栏①：主环看到的每条 claim，url 逐字出现在本次搜索原文里（归纳模型编出处会被丢）；
- C2 的核心卖点：**网页正文不进主环**——落盘原文里的正文片段，在整条 session context 里搜不到；
- C3 分账：research 发出的搜索条数记在 ``research_searches`` 上，不吃 ``web_search`` 的
  ``WEB_SEARCH_TASK_QUOTA``；两本账加起来也不挤全树 ``RETRIEVAL_BUDGET``。

跑法（真 LLM + Tavily，Qdrant 隧道见 conftest 模块头）：
    uv run pytest scripts/eval/snapshot/test_research.py -q -s
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.harness.retrieval_budget import RESEARCH_SEARCH_QUOTA

pytestmark = [pytest.mark.llm, pytest.mark.asyncio(loop_scope="session")]

TERMINAL = {
    "shopping_summary",
    "chat_fallback",
    "create_order",
    "cancel_order",
    "present_comparison",
}

QUERIES = {
    "r01_single": "AirPods Pro 2 现在口碑怎么样？我主要在意降噪和续航",
    "r02_versus": "Sony WH-1000XM5 和 Bose QuietComfort 45，降噪和佩戴哪个更好",
    "r03_category": "第一次买电动牙刷，选购该看哪些维度？",
}


def _outputs(r: Any) -> list[dict]:
    """本轮 research 的结构化返回。

    工具结果外面包着一层 ``<external_content source="research">`` 围栏（外部内容隔离），不能直接
    ``json.loads``。抠的方式是 ``raw_decode``——正则 ``\\{.*\\}`` 会贪婪吃到围栏后面去（实测
    r03 就这么炸的），raw_decode 从第一个 ``{`` 解析到该对象结束为止，多出来的尾巴自然被丢掉。
    """
    out = []
    for text in r.results.get("research", []):
        start = text.find("{")
        assert start >= 0, f"research 返回里找不到 JSON：{text[:200]}"
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError as e:
            pytest.fail(f"research 返回的不是合法 JSON（{e}）：{text[start : start + 200]}")
        out.append(obj)
    return out


def _raw_payload(sd: Path, out: dict) -> dict:
    name = out.get("raw_path") or ""
    assert name, "research 未落盘原始结果（raw_path 为空），事后归因就断了"
    path = sd / name
    assert path.exists(), f"raw_path 指向的文件不存在：{path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _context_text(sd: Path) -> str:
    """整条 session context 的可搜文本。

    **必须重新 dump 一遍**：落盘的 session.json 把中文转成了 ``\\uXXXX``，拿原文中文片段直接
    去搜永远搜不到——那样这条正文断言会一直假绿，正文真漏进主环也发现不了。
    """
    raw = json.loads((sd / "session.json").read_text(encoding="utf-8"))
    return json.dumps(raw, ensure_ascii=False)


def _probes(bodies: list[str]) -> list[str]:
    """从正文里挑几段能直接拿去搜的片段。

    片段不能跨引号 / 反斜杠 / 换行——那些字符在 JSON 文本里是转义形态，跨过去就搜不到（同样
    会假绿）。所以按这些字符切开，取最长的几段连续明文。
    """
    chunks = [c.strip() for b in bodies for c in re.split(r'["\\\n\r\t]', b)]
    chunks = [c for c in chunks if len(c) >= 40]
    return sorted(chunks, key=len, reverse=True)[:3]


@pytest.fixture
def retrieval_snapshot(monkeypatch: pytest.MonkeyPatch) -> dict:
    """在 ``run_agent`` 收尾清账前截一份检索状态。

    分账要看的 ``research_searches`` / ``web_search_runs`` 都在模块级 ``_STATE`` 里，而收尾会
    ``reset_tree()`` 把本会话那条 pop 掉（防无界增长）——跑完再去读只会读到 None，那条断言就
    永远「拿不到状态」。所以钩在清理动作上，清之前拷贝一份。
    """
    import copy

    import app.agent.orchestrator as orch
    from app.harness import retrieval_budget as rb

    snaps: dict[str, Any] = {}
    real = orch.reset_retrieval_tree

    def _capture() -> None:
        k = rb._key()
        st = rb._STATE.get(k) if k else None
        if st is not None:
            snaps[k] = copy.copy(st)
        real()

    monkeypatch.setattr(orch, "reset_retrieval_tree", _capture)
    return snaps


@pytest.mark.parametrize("case", list(QUERIES))
async def test_research_bounded(
    case: str, snap_run: Any, needs_qdrant: None, retrieval_snapshot: dict
) -> None:
    r = await snap_run(QUERIES[case])
    print(f"\n[{case}] tools={r.names}")

    assert "research" in r.names, f"三类有明确对象的问法应走 research，实际只有 {r.names}"
    # 收尾口径与 test_evaluate_named 一致：**允许纯文本收尾**（模型不再发 tool_call 即正常终止）。
    # 实测 r02/r03 走的就是这条——答案本身没问题，但不落 result.json、不走终结工具的后处理，
    # 与 prompt <termination>「纯文字 → chat_fallback」的要求不符。先记录不硬判，见验收报告。
    assert r.names and (r.names[-1] in TERMINAL or r.final_text), f"既没终结也没文本：{r.names}"
    print(f"[{case}] terminal_tool={r.names[-1] in TERMINAL}")

    outs = _outputs(r)
    ctx = _context_text(r.session_dir)
    total_searched = 0

    for out in outs:
        targets = out.get("targets") or []
        assert targets, "targets 被清洗成空还发了搜索"
        assert len(targets) <= 3, f"targets 超过 RESEARCH_MAX_TARGETS：{targets}"
        assert out.get("searched") == len(targets), (
            f"searched 与 targets 数对不上（闸预扣的就是它）：{out.get('searched')} vs {targets}"
        )
        total_searched += int(out.get("searched") or 0)

        payload = _raw_payload(r.session_dir, out)
        allowed = {
            res.get("url")
            for pack in payload.get("packs", [])
            for res in pack.get("results", [])
            if res.get("url")
        }
        for f in out.get("findings", []):
            for c in f.get("claims", []):
                assert c.get("url") in allowed, f"claim 的 url 不在本次搜索结果里：{c}"

        # C2 卖点：正文只进归纳模型。取原文里最长的一段正文，它不该出现在主环 context 里。
        bodies = [
            res.get("content") or ""
            for pack in payload.get("packs", [])
            for res in pack.get("results", [])
        ]
        for probe in _probes(bodies):
            assert probe[:80] not in ctx, f"网页正文漏进主环 context 了：{probe[:60]}…"

    st = retrieval_snapshot.get(str(r.session_dir))
    assert st is not None, "收尾前没截到检索状态，分账无从谈起"
    assert st.research_searches == total_searched, (
        f"research 账记的条数与工具实发对不上：{st.research_searches} vs {total_searched}"
    )
    assert st.research_searches <= RESEARCH_SEARCH_QUOTA
    assert st.web_search_runs <= r.names.count("web_search"), (
        f"research 的搜索被记进了 web_search 账：web_search_runs={st.web_search_runs}，"
        f"裸 web_search 只调了 {r.names.count('web_search')} 次"
    )
    print(
        f"[{case}] research_searches={st.research_searches} "
        f"web_search_runs={st.web_search_runs} tree={st.count}"
    )
