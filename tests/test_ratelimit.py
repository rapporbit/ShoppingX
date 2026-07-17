"""认证端点限流：**这是账单被刷爆之前的最后一道门**。

配额（M19）按人限，可开放注册之后「人」是能被脚本批量制造的——每个新号都合规地领一份日额度。
所以这里验的不是「限流器代码能跑」，而是「刷号这条路真的被堵死了」：注册撞 IP 闸、换 IP 撞每日
总量闸、登录撞撞库闸；以及反过来——**闸不能误伤正常用户**（关掉开关、或额度之内，必须照常放行）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

import app.api.ratelimit as rl
import app.api.server as server
from app.db.models import User
from app.db.session import session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _auth_and_limits_on(monkeypatch: Any) -> AsyncIterator[None]:
    """本文件是唯一把限流打开的地方（conftest 全局关着它，见那边注释）。

    每条用例前后都 ``reset_all()``：滑动窗口是**模块级单例**，不清就会把上一条用例用掉的额度
    带进下一条——断言随执行顺序忽红忽绿，正是最难查的那种假红。
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    rl.reset_all()
    yield
    rl.reset_all()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client: AsyncClient, username: str) -> int:
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    return resp.status_code


async def test_register_ip_window_blocks_script_signups(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """同一个 IP 连续注册，超过窗口额度即 429——「一台机器猛刷号」这条路。"""
    monkeypatch.setattr(rl, "_register_by_ip", rl.SlidingWindow(limit=3, window_s=3600))

    for i in range(3):
        assert await _register(client, f"rl-ok-{i}") == 200

    resp = await client.post(
        "/api/auth/register", json={"username": "rl-blocked", "password": "sup3r-secret"}
    )
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After")  # 告诉客户端等多久，而不是让它盲目重试
    # 关键：被 429 拦下的用户**没有被建出来**（闸在建号之前，事后再拦就晚了）
    assert await _login_status(client, "rl-blocked") == 401


async def _login_status(client: AsyncClient, username: str) -> int:
    resp = await client.post(
        "/api/auth/login", json={"username": username, "password": "sup3r-secret"}
    )
    return resp.status_code


async def test_daily_signup_cap_survives_ip_rotation(client: AsyncClient, monkeypatch: Any) -> None:
    """**换 IP 也没用**：每日总量闸查的是库里今天真多出来几行，与 IP 无关。

    IP 闸设得很宽（不让它先跳出来），只留每日总量这一道——模拟攻击者手握 IP 池的情形。

    上限按「当前基数 + 2」算而不是写死 2：``users`` 表**不在 conftest 的清理名单里**（账户测试
    靠不同用户名隔离），别的文件注册的用户同样计入今日总数——写死就会随执行顺序忽红忽绿。
    """
    monkeypatch.setattr(rl, "_register_by_ip", rl.SlidingWindow(limit=999, window_s=3600))
    async with session_factory()() as db:
        await rl.guard_daily_signups(db)  # 确认此刻还没到顶（否则下面的基数没意义）
        base = await db.scalar(select(func.count()).select_from(User))
    monkeypatch.setenv("MAX_NEW_USERS_PER_DAY", str((base or 0) + 2))

    assert await _register(client, "rl-cap-1") == 200
    assert await _register(client, "rl-cap-2") == 200

    resp = await client.post(
        "/api/auth/register", json={"username": "rl-cap-3", "password": "sup3r-secret"}
    )
    assert resp.status_code == 429
    assert "名额" in resp.json()["detail"]


async def test_login_window_blocks_credential_stuffing(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """撞库：对着一个账号猛试密码，超过窗口即 429（而不是让人无限次猜下去）。"""
    monkeypatch.setattr(rl, "_login_by_ip", rl.SlidingWindow(limit=3, window_s=900))
    monkeypatch.setattr(rl, "_register_by_ip", rl.SlidingWindow(limit=999, window_s=3600))
    assert await _register(client, "rl-victim") == 200

    for _ in range(3):
        resp = await client.post(
            "/api/auth/login", json={"username": "rl-victim", "password": "wrong-guess"}
        )
        assert resp.status_code == 401  # 密码错，但还没撞到闸

    resp = await client.post(
        "/api/auth/login", json={"username": "rl-victim", "password": "sup3r-secret"}
    )
    assert resp.status_code == 429  # 连正确密码此刻也进不来——闸在验密码之前


async def test_disabled_switch_lets_everyone_through(client: AsyncClient, monkeypatch: Any) -> None:
    """开关关掉就完全不拦——本地开发 / 测试套件靠它免疫（**闸不能误伤自己人**）。"""
    monkeypatch.setattr(rl, "_register_by_ip", rl.SlidingWindow(limit=1, window_s=3600))
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    for i in range(4):
        assert await _register(client, f"rl-off-{i}") == 200


def test_forged_xff_is_ignored_unless_proxy_trusted(monkeypatch: Any) -> None:
    """**默认不信 X-Forwarded-For**：否则攻击者每请求伪造一个新 IP，窗口永远撞不满，限流形同虚设。

    只有显式声明「我前面有反代」（生产的 Caddy）才采信，且取最后一个元素——反代把真实对端 append
    在末尾，前面那些可能是客户端自己塞的。
    """

    class _Req:
        headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        client = type("C", (), {"host": "10.0.0.1"})()

    req: Any = _Req()

    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    assert rl.client_ip(req) == "10.0.0.1"  # 伪造的 header 被无视，用真实对端

    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    assert rl.client_ip(req) == "5.6.7.8"  # 信任反代时取最后一个（反代写的那个）


def test_window_slides_rather_than_resetting_on_a_boundary() -> None:
    """滑动窗口本身：额度用满即拦，时间往前推过窗口长度后自动放行（不是固定窗口的边界跳变）。"""
    win = rl.SlidingWindow(limit=2, window_s=10)
    assert win.check("k") is None
    assert win.check("k") is None
    retry = win.check("k")
    assert retry is not None and 0 < retry <= 10  # 超限，并告知还要等几秒

    win._hits["k"] = type(win._hits["k"])([-100.0, -99.0])  # 把两次命中挪到远古
    assert win.check("k") is None  # 窗口滑过去了，重新放行
