"""长期记忆 Store 与用户偏好注入（M7 / Mmem）。

- :mod:`app.memory.domains`：``PrefDomain`` 封闭枚举——一条偏好「管哪个品类」，决定它在哪些轮次生效。
- :mod:`app.memory.store`：``PreferenceEntry`` / ``HistoryEntry`` / ``FavoriteItem`` +
  ``PreferenceStore``（后端是 :mod:`app.db` 的 SQLite）+ ``get_store()`` 工厂。
- :mod:`app.memory.session_state`：会话级短期状态 P_t（本轮约束，落 pt.json，不进长期库）。
- :mod:`app.memory.injector`：偏好读出格式化注入 + 唯一落库口 ``persist_new_preferences``。
- :mod:`app.memory.curator`：会话结束后独立跑的记忆管家——长期库的**唯一**判定 / 写入路径。
"""

from app.memory.domains import PrefDomain
from app.memory.injector import (
    build_preference_block,
    format_preferences,
    persist_new_preferences,
)
from app.memory.store import (
    FavoriteItem,
    HistoryEntry,
    PreferenceEntry,
    PreferenceStore,
    get_store,
)

__all__ = [
    "FavoriteItem",
    "HistoryEntry",
    "PrefDomain",
    "PreferenceEntry",
    "PreferenceStore",
    "build_preference_block",
    "format_preferences",
    "get_store",
    "persist_new_preferences",
]
