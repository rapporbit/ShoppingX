"""测试夹具：让单测不依赖开发者本地的 ``.env``（gitignored，CI / 新 worktree 没有）。

``app.agent.llm.get_llm`` 需要 ``LLM_MAIN`` / ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``，
缺失会在构造期抛 ``KeyError``。单测里这些模型要么被 monkeypatch 成假模型、要么根本不会
真正 ``ainvoke``（如 fork 容错路径在拿到工具前就该失败），所以这里塞一组**哑值**让
``init_chat_model`` 能把客户端建出来（OpenAI 客户端构造不发网络，真请求才发）。

用 ``setdefault``：本地若已有真实 ``.env`` / 环境变量，保持原值不覆盖。
"""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("LLM_MAIN", "gpt-4o-mini")
os.environ.setdefault("LLM_JUDGE", "gpt-4o-mini")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:9/v1")

# 单测一律走 TowerClient 的本地确定性回退（无网络、可断言、维度自定）。把 EMBED_MODEL 钉成
# 空串：①TowerClient(model=None) 读到空串 → remote=False → 本地编码；②随后 app.agent.llm 的
# load_dotenv(override=False) 不会再用 .env 里的真实 EMBED_MODEL 覆盖它。否则开发者 .env 配了真实
# embedding 时，本地单测会改打网络、维度/确定性全变（正是这次 Phase 1 暴露的回归）。
# 仍用 setdefault：想跑真实联调可在 shell 显式 export EMBED_MODEL=... 覆盖。
os.environ.setdefault("EMBED_MODEL", "")

# 同理把检索后端钉成「本地回退」：开发者 .env 配了真 OpenSearch / 远程 reranker 时，单测仍走
# 进程内 hybrid + 确定性本地打分（无网络、可断言、不依赖 docker）。空串 → KBClient/RerankerClient
# 的 remote 判定为 False。想跑真实联调同样可在 shell 显式 export 覆盖。
os.environ.setdefault("OPENSEARCH_HOST", "")
os.environ.setdefault("RERANKER_ENDPOINT", "")

# 观测同理钉死为关：开发者 .env 里 LANGFUSE_ENABLED=true 时，单测（fork 容错、dispatch 那几条走
# 真实 apply_tracing 的用例）会往**生产 Langfuse 项目**里写 trace——实测每跑一次 pytest 就多出几条
# 「system: you are a test」的垃圾 trace，混在真实会话里污染成本 / 延迟统计。想在测试里验观测行为，
# 一律 monkeypatch _get_client（tests/test_tracing.py、test_rubric.py 就是这么做的）。
os.environ.setdefault("LANGFUSE_ENABLED", "false")

# 认证限流默认关：限流的 key 是**客户端 IP**，而整个测试套件共用一个假 IP（ASGITransport 下压根
# 没有 client），于是所有文件里的注册/登录会撞在同一个窗口上——test_accounts 注册第 6 个用户就开始
# 吃 429，红的却是别人的用例。它是「跨用例共享的全局计数器」，和 usage_ledger 一样必须在测试里中和。
# 限流本身的行为由 tests/test_ratelimit.py 显式打开开关来验。
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

# M16：账户库钉到临时文件，且**强制覆盖**（不是 setdefault）——单测会真的写库（注册用户、认领
# 会话），若落到开发者的 var/globex.db 上，跑一遍 pytest 就往真实账户表里塞一堆测试用户。每次
# pytest 启动先删掉旧的临时库，保证从空表开始（用例间的隔离则靠各自用不同用户名）。
# engine 在 app.db.session 被 import 时就按这个 URL 建好，所以必须在任何 app 导入之前设。
_TEST_DB = Path(tempfile.gettempdir()) / "globex-test-accounts.db"
_TEST_DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB}"

# 建表放这儿、而不是放某个测试文件的 fixture 里：库是全局的，碰它的不止 test_accounts
# （test_auth 起任务就会写归属表）。挂在单个文件的 fixture 上，就成了「先跑的文件顺手帮后跑的
# 建表」——换个跑法（只跑 test_auth、或用例重排）当场 "no such table"，红的还是测试自己。
# 这里用 asyncio.run 同步建掉：conftest 顶层没有事件循环，也不该为它引 session 级 async fixture
# （anyio 的 backend fixture 是 function 级，套不上）。
import asyncio  # noqa: E402

import pytest  # noqa: E402

from app.db.session import init_db  # noqa: E402

asyncio.run(init_db())


@pytest.fixture(autouse=True)
def _redirect_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """把 ``ensure_session_dir`` 的输出根钉到 tmp_path——测试不许往真实 ``output/`` 拉屎。

    改造前 ``run_agent`` 系测试用 "t-1" / "t-empty" 这类硬编码 thread_id 直接写进仓库的
    ``output/``，跑一遍 pytest 就留一堆残留目录，混在真实会话产物里（离线 trace 审计时得先
    人肉分辨哪些是测试垃圾）。``ensure_session_dir`` 在调用时读模块全局 ``OUTPUT_ROOT``，
    monkeypatch 模块属性即可全量重定向；``server.OUTPUT_ROOT`` 是 by-value import 的另一份
    绑定，碰它的测试（test_server / test_history）已各自 patch，不归这里管。
    """
    import app.utils.path_utils as path_utils

    monkeypatch.setattr(path_utils, "OUTPUT_ROOT", tmp_path / "output")


@pytest.fixture(autouse=True)
def _clean_memory_tables() -> Iterator[None]:
    """每个测试跑完清空四张记忆表（偏好 / 行为历史 / 收藏 / 对话正文）。

    **Mmem 之后必须有这道清理**：改造前每个测试拿 ``LocalFileStore(root=tmp_path)``，pytest 的
    tmp_path 天然一测一目录、互不可见；现在三类数据都进了**同一个共享库**，一个测试写的 "u1"
    偏好会被下一个测试读到——串出来的红是假红，且**先跑谁就变谁的锅**，极难查。

    ``messages`` 同理、且更容易串：测试里的 thread_id 常是 "t-1" 这类硬编码字面量，不清就会让
    下一个用例的「新会话」凭空带上一段前世的对话。

    ``usage_ledger`` 也在此列且尤其要清：它按 ``(user_id, 当天)`` 累加，不清的话「上一个用例烧了
    多少 credit」会直接算进下一个用例的额度里——配额相关的断言会随**用例执行顺序**忽红忽绿。

    只清这五张，不动 ``users`` / ``threads``（账户测试自己靠不同用户名隔离，且它们之间没有
    「同名 user 反复写」的问题）。
    """
    yield
    from sqlalchemy import text

    from app.db.session import session_factory

    async def _wipe() -> None:
        async with session_factory()() as db:
            for table in (
                "preferences",
                "history_records",
                "favorites",
                "messages",
                "usage_ledger",
            ):
                await db.execute(text(f"DELETE FROM {table}"))  # noqa: S608 —— 表名是字面量常量
            await db.commit()

    asyncio.run(_wipe())
