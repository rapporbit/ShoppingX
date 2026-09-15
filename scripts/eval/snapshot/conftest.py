"""快照评测（真 LLM）夹具：环境隔离 + 起跑 + 从落盘 session.json 抽本轮工具调用。

**为什么不放 tests/**：``tests/conftest.py`` 在 app 导入前把模型钉成哑值、召回钉成本地回退，那边
跑不了真链路。这里只强制覆盖四类副作用：账户 / 交易库（钉临时 SQLite，不往开发库写测试订单）、
Langfuse（不往生产项目打 trace）、整轮缓存（关，否则新会话命中旧结果、模型根本没跑）、鉴权（关）。
模型 / 召回 / reranker 走真实 ``.env``。

**跑前**：OrbStack 里的 Qdrant 容器要起着。预计 4 条一遍 ≈ $0.007 / 2.5 分钟（deepseek-v4-flash
每百万 token 0.14 / 0.28 / 缓存读 0.0028，round3 实测口径，2026-09-15）。每条实测花费追加到
``output/snapshot_runs.jsonl``。
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_DB = Path(tempfile.gettempdir()) / "globex-snapshot.db"
_DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB}"
os.environ["TURN_CACHE_ENABLED"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["AUTH_ENABLED"] = "false"
os.environ.setdefault("ASK_USER_TIMEOUT_SEC", "5")
os.environ.setdefault("LLM_REQUEST_TIMEOUT", "300")

from app.db.session import init_db  # noqa: E402

asyncio.run(init_db())

FIXTURES = Path(__file__).parent / "fixtures"
RUN_LOG = Path(__file__).resolve().parents[3] / "output" / "snapshot_runs.jsonl"


@dataclass
class SnapResult:
    thread_id: str
    user_id: str
    session_dir: Path
    final_text: str
    calls: list[tuple[str, dict[str, Any]]]  # 本轮 (工具名, 入参)，按发生顺序
    results: dict[str, list[str]]  # 工具名 → 本轮各次返回文本

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _text_of(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(str(b.get("text", "")) for b in output if isinstance(b, dict))
    return str(output or "")


def _extract(context: list[dict]) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, list[str]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    results: dict[str, list[str]] = {}
    for msg in context:
        content = msg.get("content")
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_call":
                try:
                    args = json.loads(b.get("input") or "{}")
                except ValueError:
                    args = {}
                calls.append((str(b.get("name")), args if isinstance(args, dict) else {}))
            elif b.get("type") == "tool_result":
                results.setdefault(str(b.get("name")), []).append(_text_of(b.get("output")))
    return calls, results


@pytest.fixture
def snap_run(request: pytest.FixtureRequest) -> Any:
    """``await snap_run(query, fixture=..., user_id=...)`` → :class:`SnapResult`。

    ``fixture`` 是 ``fixtures/<名>/`` 目录，里面的文件（session.json / bundle.json）原样拷进新会话
    目录；本轮工具调用 = 跑完落盘的 context 去掉 fixture 自带的那几条消息。
    """
    from app.agent.orchestrator import ensure_session_dir, run_agent

    async def _run(
        query: str, *, fixture: str | None = None, user_id: str | None = None
    ) -> SnapResult:
        tid = f"snap_{request.node.name[:40]}_{uuid.uuid4().hex[:6]}"
        uid = user_id or f"snap_user_{uuid.uuid4().hex[:6]}"
        sd = ensure_session_dir(tid)
        prior = 0
        if fixture:
            for f in (FIXTURES / fixture).iterdir():
                shutil.copy(f, sd / f.name)
            state = json.loads((sd / "session.json").read_text(encoding="utf-8"))
            state["session_id"] = tid
            (sd / "session.json").write_text(
                json.dumps(state, ensure_ascii=False), encoding="utf-8"
            )
            prior = len(state.get("context") or [])
        t0 = time.perf_counter()
        out = await run_agent(query, thread_id=tid, user_id=uid)
        wall = round(time.perf_counter() - t0, 1)
        saved = json.loads((sd / "session.json").read_text(encoding="utf-8"))
        calls, results = _extract((saved.get("context") or [])[prior:])
        tk = out.get("tokens") or {}
        rec = {
            "case": request.node.name,
            "thread_id": tid,
            "wall_s": wall,
            "model_calls": out.get("model_calls"),
            "input_tokens": tk.get("input"),
            "cost_usd": tk.get("cost_usd"),
            "tools": [n for n, _ in calls],
        }
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return SnapResult(tid, uid, sd, out.get("final_text") or "", calls, results)

    return _run


@pytest.fixture
def needs_qdrant() -> None:
    """要走召回的用例先探 Qdrant：不在线就当场失败，别花了模型钱才发现召回全空。"""
    import httpx
    from dotenv import load_dotenv

    load_dotenv()
    url = os.environ.get("QDRANT_URL", "").rstrip("/")
    coll = os.environ.get("QDRANT_COLLECTION", "")
    try:
        httpx.get(f"{url}/collections/{coll}", timeout=3).raise_for_status()
    except Exception as e:
        pytest.fail(f"Qdrant 不在线（{url}/collections/{coll}）：先打开 OrbStack 起容器。{e!r}")


@pytest.fixture
def seed_order() -> Any:
    """``await seed_order()`` → ``(user_id, order_id)``：给一个新用户落一张 CONFIRMED 订单。"""

    async def _seed() -> tuple[str, str]:
        from app.trade.address import Address
        from app.trade.money import Money
        from app.trade.order import Order, OrderLine
        from app.trade.repository_sql import order_repository

        repo = order_repository()
        uid = f"snap_user_{uuid.uuid4().hex[:6]}"
        order = Order(
            order_id=await repo.next_order_id(),
            user_id=uid,
            thread_id="snap-seed",
            lines=[
                OrderLine(
                    platform="amazon",
                    item_id="B0C64FNWPH",
                    title="Casual Daypack Backpack",
                    unit_price=Money(2600, "USD"),
                )
            ],
            address=Address(recipient="张三", line="上海市某路 1 号", country="CN"),
        )
        order.place()
        await repo.save(order)
        return uid, order.order_id

    return _seed


@pytest.fixture
def confirmations_of() -> Any:
    """``await confirmations_of(result)`` → 该会话、该用户名下的确认记录。"""

    async def _of(r: SnapResult) -> list[Any]:
        from app.trade.repository_sql import confirmation_repository

        return await confirmation_repository().list_by_thread(r.user_id, r.thread_id)

    return _of
