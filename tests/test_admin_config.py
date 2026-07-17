"""后台管理：参数注册表 / 覆盖层 / admin API 的测试。

**本文件最要紧的一条是 test_registry_default_matches_code**：注册表里的 ``default`` 与各模块代码
里写的默认值是**两处独立的事实**，漂移了不会有任何人报错——页面上「恢复默认」会把参数恢复成一个
从未生效过的值，而 UI 上那行「默认 0.45」也在骗人。故这里显式重新表达一遍映射再比对：改了生产
代码的默认值而忘了改注册表，这个测试就红。
"""

from __future__ import annotations

import os
from functools import lru_cache
from unittest import mock

import pytest

from app.config import overrides
from app.config.registry import BY_KEY, PARAMS, RELOAD_MODULES
from app.tools import item_picker, item_search

# 模块级：本文件后半段的 admin API 用例是 async 的（同步用例不受影响）。
pytestmark = pytest.mark.anyio

# key → (模块, 模块里的常量名)。**故意手写而非从生产代码推导**——测试的价值就在于独立地把事实
# 再说一遍。只列「值住在模块级常量里」的参数；模型档位（LLM_*）是在 get_*_llm() 内部直读 env 的，
# 没有对应常量，另由 test_llm_reload_clears_cache 覆盖。
CONST_MAP: dict[str, tuple[object, str]] = {
    "ITEM_SEARCH_TOP_K": (item_search, "DEFAULT_TOP_K"),
    "ITEM_SEARCH_MAX_TOP_K": (item_search, "MAX_TOP_K"),
    "ITEM_SEARCH_SINGLE_POOL_K": (item_search, "SINGLE_PLATFORM_POOL_K"),
    "ITEM_SEARCH_RENDER_CAP": (item_search, "RENDER_CAP"),
    "RELEVANCE_FLOOR": (item_search, "RELEVANCE_FLOOR"),
    "CATEGORY_MATCH_FLOOR": (item_search, "CATEGORY_MATCH_FLOOR"),
    "ITEM_SEARCH_RETRY_MIN_HITS": (item_search, "RETRY_MIN_HITS"),
    "ITEM_SEARCH_EXCLUDE_BUFFER": (item_search, "EXCLUDE_FETCH_BUFFER"),
    "PICK_DISPLAY_CAP": (item_picker, "PICK_DISPLAY_CAP"),
    "PICK_REL_SHOW_RATIO": (item_picker, "PICK_REL_SHOW_RATIO"),
    "PICK_W_MATCH_HARD": (item_picker, "_W_MATCH_HARD"),
    "PICK_W_MATCH_HARD_SEM": (item_picker, "_W_MATCH_HARD_SEM"),
    "PICK_W_MATCH_SEM": (item_picker, "_W_MATCH_SEM"),
    "PICK_W_ATTEN_SEM": (item_picker, "_W_ATTEN_SEM"),
    "PICK_W_AFFINITY": (item_picker, "_W_AFFINITY"),
    "PICK_W_SPEC_CONFLICT": (item_picker, "_W_SPEC_CONFLICT"),
    "PICK_RERANK_FLOOR": (item_picker, "_RERANK_FLOOR"),
    "PICK_W_RERANK_MISS": (item_picker, "_W_RERANK_MISS"),
    "PICK_SLOT_RERANK_FLOOR": (item_picker, "_SLOT_RERANK_FLOOR"),
    "PICK_W_SLOT_RERANK": (item_picker, "_W_SLOT_RERANK"),
}


@pytest.fixture(autouse=True)
def _restore_params():
    """每个用例后把 env 与模块常量复原。

    **不加这个就会污染整个测试会话**：本文件会把 PICK_DISPLAY_CAP 之类改掉，而模块常量是进程级
    全局——后面 test_tools 里那些断言 `len(picks) == PICK_DISPLAY_CAP == 8` 的用例会莫名其妙地挂，
    且看起来像是它们自己的 bug。
    """
    snapshot = {p.key: os.environ.get(p.key) for p in PARAMS}
    yield
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    overrides._OVERRIDES.clear()
    item_search._load_params()
    item_picker._load_params()


def test_registry_default_matches_code():
    """注册表的 default == 模块代码里的默认值（清掉 env 后重新求值即是代码默认值）。"""
    for key in CONST_MAP:
        os.environ.pop(key, None)
    item_search._load_params()
    item_picker._load_params()

    for key, (module, const) in CONST_MAP.items():
        assert getattr(module, const) == BY_KEY[key].default, (
            f"{key}：注册表写 {BY_KEY[key].default}，但代码里 {const} 求值出 "
            f"{getattr(module, const)} —— 两处漂移了，「恢复默认」会恢复成一个假值"
        )


def test_registry_const_matches_handwritten_map():
    """注册表的 const 字段 == 本文件手写的映射。两份独立表达，漂了就红。"""
    declared = {p.key: p.const for p in PARAMS if p.const}
    handwritten = {key: const for key, (_, const) in CONST_MAP.items()}
    assert declared == handwritten


def test_ui_value_is_the_real_one_not_a_re_derivation():
    """页面显示的必须是**真实生效值**，不是拿 env 重新推导的值。

    真踩过的 bug：把召回条数改成 15，模块里 MAX_TOP_K 确实跟着变成 15（它的代码默认值取
    DEFAULT_TOP_K），但「读 env → 没配 → 回退注册表 default=10」的推导会得出 10。页面于是在
    「我刚改完、正想确认生效了没」的那一刻显示了个假值。
    """
    overrides.apply({"ITEM_SEARCH_TOP_K": 15})  # 只改这个，不碰 MAX_TOP_K
    assert item_search.MAX_TOP_K == 15  # 模块里真实生效的
    assert overrides.current_value(BY_KEY["ITEM_SEARCH_MAX_TOP_K"]) == 15  # 页面看到的


def test_llm_reload_clears_cache():
    """模型档位没有模块常量，靠清 lru_cache 生效：不清的话改完 LLM_MAIN 仍拿旧模型的实例。

    只断言「缓存被清了」，不真去建实例——建实例要连 endpoint，那是集成测试的事。
    """
    from app.agent import llm

    calls = {"n": 0}

    @lru_cache(maxsize=1)
    def fake_factory():
        calls["n"] += 1
        return object()

    # 用真的工厂名替身，验证 _load_params 会把它们全清一遍。
    with mock.patch.multiple(
        llm,
        get_llm=fake_factory,
        get_fast_llm=fake_factory,
        get_vision_llm=fake_factory,
        get_judge_llm=fake_factory,
    ):
        fake_factory()
        assert fake_factory.cache_info().currsize == 1
        llm._load_params()
        assert fake_factory.cache_info().currsize == 0, "改完模型档位没清缓存 = 改了不生效"


def test_registry_covers_only_reloadable_modules():
    """注册表里声明的每个模块都真的有 _load_params()，否则改了参数不生效且无人报错。"""
    import importlib

    for path in RELOAD_MODULES:
        module = importlib.import_module(path)
        assert callable(getattr(module, "_load_params", None)), f"{path} 缺 _load_params()"


def test_apply_reaches_module_constants_and_reset_restores():
    """覆盖层的本职：apply 后模块常量真的变了，reset 后真的退回默认。"""
    overrides.apply({"PICK_DISPLAY_CAP": 3, "RELEVANCE_FLOOR": 0.9})
    assert item_picker.PICK_DISPLAY_CAP == 3
    assert item_search.RELEVANCE_FLOOR == 0.9

    overrides.reset(["PICK_DISPLAY_CAP", "RELEVANCE_FLOOR"])
    assert item_picker.PICK_DISPLAY_CAP == BY_KEY["PICK_DISPLAY_CAP"].default
    assert item_search.RELEVANCE_FLOOR == BY_KEY["RELEVANCE_FLOOR"].default


def test_max_top_k_follows_top_k_within_one_reload():
    """MAX_TOP_K 默认取 DEFAULT_TOP_K —— 同一次 _load_params() 里按源码顺序求值才成立。

    这是「把参数搬进函数」而非「散在模块各处重新赋值」换来的性质：只改 TOP_K，封顶自动跟随，
    不会留下「召回 25 条却被旧的 10 封顶」这种半生效状态。
    """
    overrides.apply({"ITEM_SEARCH_TOP_K": 25})
    assert item_search.DEFAULT_TOP_K == 25
    assert item_search.MAX_TOP_K == 25


@pytest.mark.parametrize(
    "key,bad",
    [
        ("PICK_DISPLAY_CAP", 999),  # 越上界
        ("PICK_DISPLAY_CAP", 0),  # 越下界
        ("PICK_DISPLAY_CAP", "abc"),  # 类型不对
        ("PICK_DISPLAY_CAP", ""),  # 数值参数不接受留空
        ("LLM_MAIN", ""),  # allow_empty=False 的字符串
        ("NOT_A_REAL_PARAM", 1),  # 未知 key
    ],
)
def test_normalize_rejects_bad_values(key, bad):
    with pytest.raises(overrides.ParamValidationError):
        overrides.normalize(key, bad)


def test_apply_is_all_or_nothing():
    """一批里有一个非法就整批不生效——半生效比拒绝更糟：页面显示改了 3 项，实际只生效 2 项。"""
    before = item_picker.PICK_DISPLAY_CAP
    with pytest.raises(overrides.ParamValidationError):
        overrides.apply({"PICK_DISPLAY_CAP": 5, "PICK_W_AFFINITY": 999})
    assert item_picker.PICK_DISPLAY_CAP == before


def test_source_reflects_where_value_came_from(monkeypatch):
    """来源判定：override / env / default 三态。前端靠它标「已改」。"""
    # 基线快照是模块导入时拍的，故这里直接构造基线字典来模拟「.env 里配过」。
    monkeypatch.setitem(overrides._BASELINE_ENV, "PICK_W_AFFINITY", "0.5")
    assert overrides.source_of("PICK_W_AFFINITY") == "env"
    assert overrides.source_of("PICK_W_SPEC_CONFLICT") == "default"
    overrides.apply({"PICK_W_AFFINITY": 0.3})
    assert overrides.source_of("PICK_W_AFFINITY") == "override"


def test_bool_accepts_common_spellings():
    assert overrides.normalize("LLM_FAST_REASONING", "on") == "true"
    assert overrides.normalize("LLM_FAST_REASONING", False) == "false"
    assert overrides.normalize("LLM_FAST_REASONING", "0") == "false"


# --- admin API ------------------------------------------------------------
#
# 这几条问的是同一个问题：「不该进来的人能不能改模型」。后台能换模型、能改召回阈值，是本项目
# 权限最高的面，故三道门（鉴权开没开 / 有没有 token / 是不是白名单里的人）各测一条。


@pytest.fixture
async def api():
    from collections.abc import AsyncIterator  # noqa: F401

    from httpx import ASGITransport, AsyncClient

    import app.api.server as server

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup(api, username: str) -> dict[str, str]:
    resp = await api.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_admin_closed_when_auth_disabled(api, monkeypatch):
    """鉴权没开时后台整体不可用——此时身份是前端自报的，放行等于把改模型开放给所有人。"""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    resp = await api.get("/api/admin/config")
    assert resp.status_code == 403
    assert "AUTH_ENABLED" in resp.json()["detail"]


async def test_admin_requires_whitelist(api, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "the-boss")

    assert (await api.get("/api/admin/config")).status_code == 401  # 无 token

    peon = await _signup(api, "peon-user")
    assert (await api.get("/api/admin/config", headers=peon)).status_code == 403  # 非白名单

    boss = await _signup(api, "the-boss")
    resp = await api.get("/api/admin/config", headers=boss)
    assert resp.status_code == 200
    assert len(resp.json()["params"]) == len(PARAMS)


async def test_admin_no_admins_configured_closes_door(api, monkeypatch):
    """ADMIN_USERNAMES 为空 = 没有管理员，谁都进不去（不是「谁都能进」）。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "")
    headers = await _signup(api, "nobody-special")
    assert (await api.get("/api/admin/config", headers=headers)).status_code == 403


async def test_admin_update_and_reset_roundtrip(api, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "cfg-admin")
    boss = await _signup(api, "cfg-admin")

    resp = await api.put(
        "/api/admin/config", headers=boss, json={"values": {"PICK_DISPLAY_CAP": 4}}
    )
    assert resp.status_code == 200
    assert item_picker.PICK_DISPLAY_CAP == 4
    view = next(p for p in resp.json()["params"] if p["key"] == "PICK_DISPLAY_CAP")
    assert view["value"] == 4 and view["source"] == "override"

    bad = await api.put(
        "/api/admin/config", headers=boss, json={"values": {"PICK_DISPLAY_CAP": 999}}
    )
    assert bad.status_code == 400
    assert item_picker.PICK_DISPLAY_CAP == 4  # 拒绝掉的那次没留下痕迹

    resp = await api.post(
        "/api/admin/config/reset", headers=boss, json={"keys": ["PICK_DISPLAY_CAP"]}
    )
    assert resp.status_code == 200
    assert item_picker.PICK_DISPLAY_CAP == BY_KEY["PICK_DISPLAY_CAP"].default


async def test_load_from_db_survives_dirty_rows(caplog):
    """启动路径：库里的未知 key / 非法值只跳过并记 warning，**绝不让服务起不来**。

    这是真会发生的：注册表删掉一个参数、或把范围收紧（如权重上限从 10 收到 5），库里那条旧行
    立刻变成「非法」。若在 lifespan 里抛，整个服务起不来，而症状（起不来）与病因（一条旧配置）
    隔着十万八千里。
    """
    from app.config import store
    from app.db.models import ConfigOverride
    from app.db.session import session_factory

    async with session_factory()() as db:
        db.add(ConfigOverride(key="GONE_FROM_REGISTRY", value="1", updated_by="t"))
        db.add(ConfigOverride(key="PICK_DISPLAY_CAP", value="99999", updated_by="t"))  # 越界值
        db.add(ConfigOverride(key="PICK_W_AFFINITY", value="0.4", updated_by="t"))  # 好的那条
        await db.commit()

    try:
        applied = await store.load_into_memory()
        assert applied == 1  # 只有好的那条生效
        assert item_picker._W_AFFINITY == 0.4
        assert item_picker.PICK_DISPLAY_CAP == BY_KEY["PICK_DISPLAY_CAP"].default  # 脏值没生效
    finally:
        async with session_factory()() as db:
            from sqlalchemy import delete

            await db.execute(delete(ConfigOverride))
            await db.commit()


# --- 密钥（API key）------------------------------------------------------
#
# 三条规则各锁一条。这几条测的都是「密钥会不会漏 / 会不会被误抹」，错了不会崩，只会安静地把
# key 送出去或把全站 LLM 调用打挂。


def test_secret_never_echoes_plaintext():
    """current_value 对密钥恒返回空串——明文绝不出现在给页面的响应里。"""
    os.environ["OPENAI_API_KEY"] = "sk-super-secret-key-do-not-leak"
    p = BY_KEY["OPENAI_API_KEY"]
    assert p.secret is True
    assert overrides.current_value(p) == ""


def test_secret_mask_shows_enough_to_verify_not_enough_to_use():
    os.environ["OPENAI_API_KEY"] = "sk-abcdefghijklmnop1234"
    assert overrides.masked_value(BY_KEY["OPENAI_API_KEY"]) == "sk-…1234"

    # 短串不露内容：12 位以下露前 3 后 4 等于交出去大半。
    os.environ["OPENAI_API_KEY"] = "sk-short"
    assert overrides.masked_value(BY_KEY["OPENAI_API_KEY"]) == "已配置"

    os.environ.pop("OPENAI_API_KEY", None)
    assert overrides.masked_value(BY_KEY["OPENAI_API_KEY"]) == ""
    # 非密钥参数不该有掩码
    assert overrides.masked_value(BY_KEY["PICK_DISPLAY_CAP"]) == ""


async def test_blank_secret_means_unchanged_not_erased(api, monkeypatch):
    """密钥留空 = 不改，且**不连累同一批里的其它参数**。

    密钥在页面上天生是空的（不回显）。若后端不把空值剔掉，OPENAI_API_KEY（allow_empty=False）会让
    normalize 拒绝整批 → 管理员改任何别的参数都被这个空密钥连累成 400，页面根本没法用。若某个密钥
    是 allow_empty=True，后果换成更糟的那种：空串被当真值写进 env，key 直接被抹。
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "key-admin")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-original-key-must-survive")
    boss = await _signup(api, "key-admin")

    # 模拟前端连着密钥空值一起提交（改的是别的参数）
    resp = await api.put(
        "/api/admin/config",
        headers=boss,
        json={"values": {"OPENAI_API_KEY": "", "PICK_DISPLAY_CAP": 6}},
    )
    assert resp.status_code == 200
    assert os.environ["OPENAI_API_KEY"] == "sk-original-key-must-survive"  # 没被抹
    assert item_picker.PICK_DISPLAY_CAP == 6  # 同批里的正常参数照常生效


async def test_secret_can_be_set_and_is_masked_in_response(api, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "key-admin2")
    boss = await _signup(api, "key-admin2")

    resp = await api.put(
        "/api/admin/config",
        headers=boss,
        json={"values": {"OPENAI_API_KEY": "sk-brand-new-key-9876"}},
    )
    assert resp.status_code == 200
    assert os.environ["OPENAI_API_KEY"] == "sk-brand-new-key-9876"

    view = next(p for p in resp.json()["params"] if p["key"] == "OPENAI_API_KEY")
    assert view["value"] == ""  # 不回显
    assert view["masked"] == "sk-…9876"  # 够核对
    assert "sk-brand-new-key-9876" not in resp.text  # 整份响应里都不该有明文


async def test_blank_allow_empty_secret_also_survives(api, monkeypatch):
    """allow_empty=True 的密钥（视觉 key）留空同样是「不改」，不是「抹掉」。

    这条与上一条分开测：两者的失败模式不同（那条是整批 400，这条是值被静默清空），而「静默清空」
    连报错都没有——视觉档会悄悄回退去用主 key，直到某天你发现图片理解在用错的账单跑。
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "key-admin3")
    monkeypatch.setenv("VISION_API_KEY", "sk-vision-key-keep-me-1234")
    boss = await _signup(api, "key-admin3")

    resp = await api.put("/api/admin/config", headers=boss, json={"values": {"VISION_API_KEY": ""}})
    assert resp.status_code == 200
    assert os.environ["VISION_API_KEY"] == "sk-vision-key-keep-me-1234"


async def test_reset_is_the_way_to_clear_a_secret(api, monkeypatch):
    """密钥唯一的「清空」途径是恢复默认——留空不行（那是「不改」），得走 reset。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("ADMIN_USERNAMES", "key-admin4")
    monkeypatch.delenv("VISION_API_KEY", raising=False)  # 基线：没配过
    boss = await _signup(api, "key-admin4")

    await api.put(
        "/api/admin/config",
        headers=boss,
        json={"values": {"VISION_API_KEY": "sk-tmp-key-abcd1234"}},
    )
    assert os.environ["VISION_API_KEY"] == "sk-tmp-key-abcd1234"

    resp = await api.post(
        "/api/admin/config/reset", headers=boss, json={"keys": ["VISION_API_KEY"]}
    )
    assert resp.status_code == 200
    assert os.environ.get("VISION_API_KEY") is None  # 退回基线（没配）= 复用主 key
