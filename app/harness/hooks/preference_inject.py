"""post_tool_call：planner 跑完之后，把**本轮域内**的长期偏好注入给模型。

**为什么非得等 planner。** 偏好块原先拼在当轮 human message 的最前面（``main_agent.
_inject_runtime_context``），可那一刻 planner 还没跑、``session_domains`` 还是空的，而
``injector._in_scope`` 对空域一律放行——于是模型看到的偏好块**必然是跨域全量**的。「买跑鞋时
不要皮革」（域=footwear）就这么出现在买旅行包的这一轮，模型很自觉地把 leather 转述进
``item_picker(exclude_keywords=...)``，硬淘汰绕过域闸生效。买旅行包时没出事纯属侥幸（召回里
没皮革包）；换成「想买个真皮公文包」，正确答案会被一条八竿子打不着的旧偏好全杀光。

时序上这在注入侧无解——注入的那一刻品类域还不存在。所以把注入**挪到域产生之后**：planner 判出
domains → 本 Hook 注入域内偏好。模型从此看不到跨域偏好，也就不存在转述它的可能。这比在消费侧
挨个堵参数（exclude / prefer / deprioritize / planner intent…）干净：**不给，胜过给了再管。**

闲聊轮不跑 planner → 不注入，这是对的：没有购物意图，就没有「哪些偏好相关」可言。
子 Agent（depth ≥ 1）不注入：它继承父的 session_dir / user_id，偏好由机制侧在工具内并入
（``memory.assemble``），不需要它自己再看一遍文本。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.api.context import get_user_id
from app.harness.middleware import harness_hook
from app.memory.injector import PREF_EMPTY, build_preference_block

logger = logging.getLogger("shoppingx.harness.preference_inject")


@harness_hook("post_tool_call", name="preference_inject", priority=50)
async def inject_domain_preferences(context: dict[str, Any]) -> dict[str, Any] | None:
    """planner 返回后注入域内长期偏好。一轮至多注入一次——阶段机保证 planner 只成功跑一次
    （跑完即离开 PLANNING，而 planner 不在后续阶段的白名单里）。"""
    if context.get("tool_name") != "planner" or current_fork_depth() >= 1:
        return None

    user_id = get_user_id() or ""
    if not user_id:
        return None  # 匿名用户没有长期偏好

    block = await build_preference_block(user_id)
    if not block or block == PREF_EMPTY:
        return None  # 本轮域内没有任何偏好 → 不塞空占位（省 token，也不给模型噪声）

    logger.info("注入域内长期偏好（planner 后）")
    context.setdefault("inject_messages", []).append(
        {
            "role": "system",
            "content": (
                "<user_long_term_preferences>\n"
                f"{block}\n"
                "</user_long_term_preferences>\n"
                "以上是该用户与**本轮品类相关**的长期偏好，已由系统自动生效（检索词与精挑打分里\n"
                "都已并入，见 memory.assemble）——**不要**再把它们转述进任何工具参数，重复一遍不会\n"
                "让它们更生效，只会让你替用户做了他没授权的决定。它们在这里只为一件事：让你在向\n"
                "用户解释「为什么选这几件」时，说得出是哪条偏好起了作用。"
            ),
        }
    )
    return context
