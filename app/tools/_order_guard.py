"""下单确认门：会话级记录「哪一组商品出过确认卡」。

`create_order` 的两段式（先出确认卡、用户同意后再真下单）本身只是**约定**——模型完全可以第一次
调用就传 `confirmed=True`，把确认这一步跳过去。这个模块把约定变成机制：真下单前查一下，这组
商品在本会话里确实出过确认卡；没出过就退回去先出卡。

**为什么不用框架原生的 `RequireUserConfirmEvent`。** 它是「工具调用前弹一个确认」，确认的载体是
前端要实现的一套事件协议；而本仓的确认卡就是对话流里的一张 OrderCard，用户回一句「确认」即可，
前端零改动，多轮恢复也走已经跑通的 history 通路。两条路都能防误下单，选了与现有交互一致的那条。
（L0 spike 验过原生通路可跨实例恢复且恰好执行一次，需要时可以换。）

按 `session_dir` 聚合，与候选登记表同一套口径：worker 继承父 session_dir，所以主 Agent 出的卡，
TradeAgent 认得。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from app.api.context import get_session_dir

# 确认卡有效期：出卡后超过这个时长，「确认」不再直接落库，而是重新出一张卡（价格 / 候选可能
# 已经变了，让用户再看一眼）。前端据 expires_at 画倒计时并在过期后灰掉按钮。
PREVIEW_TTL_SECONDS = 30 * 60

# session_dir(str) -> {商品组指纹: 出卡时刻（epoch 秒）}
_CONFIRMED_PREVIEWS: dict[str, dict[str, float]] = {}


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def preview_fingerprint(item_ids: list[str]) -> str:
    """一组商品的指纹（与顺序无关）。

    只按 item_id 不按数量：用户看完确认卡说「第二件要两个」是很自然的一句话，为此让他重看一遍
    卡片纯属找骂。数量变了金额会变，而金额在下单结果里还会再报一次。
    """
    return ",".join(sorted(set(item_ids)))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def mark_preview_shown(item_ids: list[str]) -> str:
    """记下「这组商品出过确认卡」，返回这张卡的失效时刻（UTC ISO）。"""
    shown_at = time.time()
    key = _key()
    if key is not None:
        _CONFIRMED_PREVIEWS.setdefault(key, {})[preview_fingerprint(item_ids)] = shown_at
    return _iso(shown_at + PREVIEW_TTL_SECONDS)


def preview_expired(item_ids: list[str]) -> bool:
    """这组商品出过确认卡、但卡已过了有效期。没出过卡返回 False（那是另一种情况）。"""
    key = _key()
    if key is None:
        return False
    shown_at = _CONFIRMED_PREVIEWS.get(key, {}).get(preview_fingerprint(item_ids))
    return shown_at is not None and time.time() - shown_at > PREVIEW_TTL_SECONDS


def preview_shown(item_ids: list[str]) -> bool:
    """这组商品在本会话出过确认卡吗。

    无会话作用域（离线脚本 / 单测直调）时返回 True：那些场景没有「用户」可确认，卡在这里只会
    让测试写不下去。真实链路一定有 session_dir。
    """
    key = _key()
    if key is None:
        return True
    shown_at = _CONFIRMED_PREVIEWS.get(key, {}).get(preview_fingerprint(item_ids))
    return shown_at is not None and time.time() - shown_at <= PREVIEW_TTL_SECONDS


def reset_order_guard() -> None:
    """清掉当前会话的记录（测试与会话回零用）。"""
    key = _key()
    if key is not None:
        _CONFIRMED_PREVIEWS.pop(key, None)
