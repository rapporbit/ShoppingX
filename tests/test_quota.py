"""M18 用户级 credit 配额：记账 / 周期 / 入口闸 / 单任务上限联动。

**这些用例真正在问的是：「一个人到底能不能烧穿账单」。** token_budget 只保证单次任务不失控，
对「同一个人连发一百条合规 query」完全无感——本文件盯的就是那道跨会话的闸：账记不记得住（累加 /
并发不丢更新）、周期换没换（跨天回满）、闸拦不拦得住（402）、以及**最后一次任务能透支多少**
（被压低的 task cap）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app.db import quota

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _quota_on(monkeypatch: Any) -> AsyncIterator[None]:
    """配额只在**开着鉴权**时生效（关掉时没有可信身份可限），故整个文件在「鉴权已开」下跑。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("DAILY_QUOTA_USD", "1.0")  # = 1000 credits
    yield


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup(client: AsyncClient, username: str) -> tuple[str, dict[str, str]]:
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


async def test_fresh_user_has_full_quota(client: AsyncClient) -> None:
    """没记过账的人 = 一分没用（不必先插一行占位，get_quota 不写库）。"""
    _, headers = await _signup(client, "q-fresh")
    body = (await client.get("/api/quota", headers=headers)).json()
    assert body["enabled"] is True
    assert body["used_credits"] == 0
    assert body["remaining_credits"] == body["limit_credits"] == 1000
    assert body["exhausted"] is False


async def test_usage_accumulates_across_tasks(client: AsyncClient) -> None:
    """多次记账在**同一个周期行**上累加——这是「跨会话累计」的全部意义。"""
    uid, headers = await _signup(client, "q-accum")
    await quota.add_usage(uid, 0.1, input_tokens=1000, output_tokens=200)
    await quota.add_usage(uid, 0.25, input_tokens=2000, output_tokens=400)

    body = (await client.get("/api/quota", headers=headers)).json()
    assert body["used_credits"] == 350  # $0.35 → 350 credits
    assert body["remaining_credits"] == 650
    assert body["task_count"] == 2


async def test_quota_exhausted_blocks_new_task(client: AsyncClient) -> None:
    """额度耗尽 → ``POST /api/task`` 直接 402，任务根本不起（不占槽、不认领 thread）。"""
    uid, headers = await _signup(client, "q-empty")
    await quota.add_usage(uid, 1.0)  # 一把烧完当日额度

    resp = await client.post("/api/task", json={"query": "买个背包"}, headers=headers)
    assert resp.status_code == 402
    detail = resp.json()["detail"]
    assert detail["error"] == "quota_exhausted"
    assert detail["remaining_credits"] == 0
    assert detail["reset_at"]  # 前端要拿它告诉用户几点回满
    assert not server.active_tasks  # 关键：被拒的任务不留任何足迹


async def test_period_rollover_resets(client: AsyncClient, monkeypatch: Any) -> None:
    """跨到新的一天 = 换一行新账 → 额度自然回满，不需要任何定时任务。"""
    uid, headers = await _signup(client, "q-rollover")
    await quota.add_usage(uid, 1.0)
    assert (await client.get("/api/quota", headers=headers)).json()["exhausted"] is True

    monkeypatch.setattr(quota, "current_period", lambda: "2099-01-01")  # 时间来到「明天」
    body = (await client.get("/api/quota", headers=headers)).json()
    assert body["used_credits"] == 0
    assert body["remaining_credits"] == 1000
    assert body["exhausted"] is False


async def test_disabled_when_auth_off(client: AsyncClient, monkeypatch: Any) -> None:
    """鉴权关闭（demo 模式）→ 不设闸：既不记账，也不拦任务，前端见 enabled=false 隐藏余额条。"""
    uid, headers = await _signup(client, "q-demo")
    monkeypatch.setenv("AUTH_ENABLED", "false")

    await quota.add_usage(uid, 5.0)  # 远超上限，但鉴权关着 → 压根不该记进去
    assert (await client.get("/api/quota", headers=headers)).json()["enabled"] is False

    monkeypatch.setenv("AUTH_ENABLED", "true")  # 开回来验证：刚才那 5 刀确实没落库
    assert (await client.get("/api/quota", headers=headers)).json()["used_credits"] == 0


async def test_remaining_usd_caps_task_budget(monkeypatch: Any, tmp_path: Any) -> None:
    """最后一次任务只能透支到「额度刚好用尽」——单任务预算被压成 min(预算, 今日剩余)。

    没有这一压，剩余只够 $0.01 的用户照样能起一趟烧满 $0.50 的任务：入口闸放行了，之后就没人管。
    """
    from app.agent import token_budget
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setenv("TOKEN_BUDGET_USD", "0.50")
    uid = "q-cap-user"
    await quota.add_usage(uid, 0.97)  # 剩 $0.03

    left = await quota.remaining_usd(uid)
    assert left is not None and abs(left - 0.03) < 1e-6

    with thread_scope("t-cap", tmp_path):
        token_budget.set_task_cap(left)
        assert abs(token_budget.budget_cap_usd() - 0.03) < 1e-6  # 被剩余额度压低了
        token_budget.reset_tree()

    # 剩余额度比单任务预算还宽时，cap 不该被抬高——它是上限，不是目标值。
    with thread_scope("t-cap2", tmp_path):
        token_budget.set_task_cap(10.0)
        assert abs(token_budget.budget_cap_usd() - 0.50) < 1e-6
        token_budget.reset_tree()
