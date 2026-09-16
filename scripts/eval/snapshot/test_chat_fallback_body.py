"""快照评测（真 LLM）：chat_fallback 收尾时，模型写的正文有没有活着到达用户。

这条守的是 2026-09-16 修掉的那个**用户可见的静默损失**：模型常把整篇回答写成 assistant 文本、
``message`` 入参里只留一句「以上就是…」。而主 loop 的文本不推前端（events.py 的刻意偏离）、
``final_text`` 只取终结工具那句——于是那篇回答用户一个字都看不到，产物与历史里也只剩客套话。
实测 r03_category 一次：1500+ 字符的选购指南，最终只存下 312 字节。

两处修：``chat_fallback`` 不再拿 fast 模型改写 message（原样透出）；``adapter`` 在收尾 Msg 上
把同一条消息里的正文并回最终答案（并完照常过 output_guard/output_audit）。

跑法（真 LLM，Qdrant 隧道见 conftest 模块头）：
    uv run pytest scripts/eval/snapshot/test_chat_fallback_body.py -q -s
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.llm, pytest.mark.asyncio(loop_scope="session")]

QUERY = "跨境网购的关税一般怎么算？先别搜商品，把规则给我讲讲就行"


def _fallback_turn_text(sd: Path) -> str:
    """调 chat_fallback 的那条 assistant 消息里的正文（与 adapter 的取法同一口径）。"""
    state = json.loads((sd / "session.json").read_text(encoding="utf-8"))
    for msg in reversed(state.get("context") or []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        names = {
            b.get("name") for b in content if isinstance(b, dict) and b.get("type") == "tool_call"
        }
        if "chat_fallback" in names:
            return "".join(
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
    return ""


async def test_chat_fallback_keeps_the_prose(snap_run: Any) -> None:
    r = await snap_run(QUERY)
    print(f"\n[chat_fallback_body] tools={r.names} final_len={len(r.final_text)}")

    if "chat_fallback" not in r.names:
        pytest.skip(f"这遍模型没走 chat_fallback（纯文本收尾也不丢答案）：{r.names}")

    message = ""
    for name, args in r.calls:
        if name == "chat_fallback":
            message = str(args.get("message") or "").strip()
    assert message, "chat_fallback 调用没带 message"
    # message 原样透出：工具不再改写（改写过的那版会把长答案换成一句客套话）。
    assert message in r.final_text, "message 没有原样出现在最终答案里，工具又在改写了"

    body = _fallback_turn_text(r.session_dir)
    if body and body not in message and message not in body:
        assert body in r.final_text, "模型写在正文里的那段没并进最终答案，用户看不到它"

    saved = (r.session_dir / "summary.md").read_text(encoding="utf-8")
    assert saved.strip() == r.final_text.strip(), "落盘产物与最终答案不一致"
    print(f"[chat_fallback_body] body_len={len(body)} message_len={len(message)}")
