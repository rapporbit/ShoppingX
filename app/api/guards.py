"""HTTP 层共用的两道守卫：thread_id 路径校验 + 会话归属校验。

从 ``server.py`` 拆出来，供 ``server`` / ``files`` / ``preferences`` / ``orders`` 共用。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import (
    HTTPException,
)

from app.api.auth import (
    auth_enabled,
)
from app.db.accounts import assert_owner
from app.db.session import session_factory
from app.utils.path_utils import (
    safe_join,
)


def safe_session_dir(root: Path, thread_id: str) -> Path:
    """把 ``root/<thread_id>`` 经 ``safe_join`` 校验后返回——**thread_id 也是用户可控输入**。

    download 的 ``thread_id`` 来自 URL 段、upload 的来自表单，二者都可能塞 ``..``（如编码的
    ``%2e%2e`` 或表单里直接写 ``../../etc``）。若像最初那样 ``root / thread_id`` 直接拼，会在
    ``safe_join(filename)`` 之前就已逃出 root——文件名那道 safe_join 守的是错的那半截路径。
    这里对 thread_id 也走 safe_join，逃逸即 400（对齐 CONVENTIONS「文件路径一律 safe_join」）。
    """
    try:
        return safe_join(root, thread_id)
    except ValueError as exc:
        raise HTTPException(400, "非法会话标识") from exc


async def guard_thread(thread_id: str, auth_uid: str | None) -> None:
    """校验当前用户有权访问这个 thread，否则 403。

    **这是 auth.py 当初点名却没做的那半边洞。** 原先所有 thread 接口（history / files / ws /
    cancel / upload）只按 thread_id 寻址、不问归属：thread_id 会出现在 URL 和事件流里，谁拿到
    就能读别人的对话历史、下载他的产物、连他的实时事件流。有了归属表，这里一句校验就封死。

    鉴权关闭时直接放行——免鉴权模式下没人认领会话，校验无从谈起（见 accounts.assert_owner）。
    """
    if not auth_enabled() or auth_uid is None:
        return
    async with session_factory()() as db:
        try:
            await assert_owner(db, thread_id, auth_uid)
        except PermissionError as exc:
            raise HTTPException(403, "无权访问该会话") from exc
