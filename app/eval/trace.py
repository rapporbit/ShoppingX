"""评测 / 训练侧的**运行时中立**轨迹解析（批 0 / L6）。

评测与蒸馏脚本要回答的问题只有三个——「这轮按顺序调了哪些工具」「最后一句回复是什么」
「怎么从落盘的 ``history.json`` 把前两者读出来」。这三件事本身与底下跑的是 LangChain 还是
AgentScope 无关，可两套运行时的消息形态差得很远：

- **LangChain**：一条 ``AIMessage`` 顶层挂 ``tool_calls``（``{name, args}``），一次工具结果
  单独一条 ``ToolMessage``；落盘走 ``messages_to_dict``，元素形如 ``{"type": "ai", "data": {…}}``。
- **AgentScope**：**一次 reply = 一条 assistant ``Msg``**，本轮所有 ``tool_call`` / ``tool_result``
  block 全 ``extend`` 进它的 ``content`` 里（L5 起还会混进 ``HintBlock`` 注入）；落盘走
  ``Msg.model_dump()``，元素形如 ``{"role": "assistant", "content": [blocks], …}``。

与其让 ``rubric`` / ``distill_fewshot`` 各写一遍 if-else，不如把差异收进本模块，**先归一成中立
dict 再解析**。L8 摘掉 LangChain 时删的是这里的一个分支，上层一行不动（同 ``harness/_msgcompat``
的思路，但那个服务控制面、这个服务评测面，两边的输入根本不是同一批对象，不合并）。

刻意**不 import 任何一方的消息类**：解析全走鸭子判断，于是本模块在两个运行时下都可导入，
离线脚本（训练机上依赖装得很薄）也能用。
"""

import json
from pathlib import Path
from typing import Any

# 中立消息形态：``{"role": "assistant" | "user" | "tool" | "system", "text": str,
# "tool_calls": [{"name": str, "args": dict}]}``。故意用 dict 而不是 Pydantic——
# ``extract_tool_calls`` 的返回值是既有公开契约（``rubric.render_trajectory`` 与测试都按
# ``c["name"]`` / ``c["args"]`` 取），换成模型对象等于给所有调用点找活干。
NormMsg = dict[str, Any]

_ROLE_ALIASES = {"ai": "assistant", "human": "user", "tool": "tool", "system": "system"}


def _block_field(block: Any, key: str) -> Any:
    """从 block 取字段——AgentScope 的 block 是 TypedDict（运行时就是 dict），但落盘再读回来
    也可能是普通 dict，且 SDK 未来换成对象也不该让这里翻。两路都试。"""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _parse_args(raw: Any) -> dict[str, Any]:
    """``ToolCallBlock.input`` → 入参 dict。

    **它是 JSON 字符串不是 dict**：流式解析里工具入参是一段段 JSON 文本拼起来的
    （``ChatResponse.append_tool_call`` 直接 ``block.input += input``），落盘后还是字符串。
    当 dict 用会静默得到空入参——轨迹渲染成 ``planner()``，judge 看不到 Agent 到底传了什么。
    解析不出来时保留原文（模型偶尔会吐半截 JSON），别把这条调用整个丢掉。
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {}


def _from_blocks(content: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """解析 AgentScope 风格的 content blocks → ``(拼接后的文本, 工具调用序列)``。

    文本按出现顺序拼接：一轮里模型可能先说一段再调工具、调完再补一段，只取最后一段会丢内容。
    ``tool_result`` / ``thinking`` / ``HintBlock`` 不进文本——分别是观察、思维链、控制面注入，
    都不是模型说给用户听的话，混进去会让「最终回复」变成工具 JSON。
    """
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for block in content:
        btype = _block_field(block, "type")
        if btype == "text":
            text = _block_field(block, "text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        elif btype == "tool_call":
            calls.append(
                {
                    "name": _block_field(block, "name") or "?",
                    "args": _parse_args(_block_field(block, "input")),
                }
            )
    return "\n".join(texts), calls


def normalize_message(msg: Any) -> NormMsg:
    """把一条消息（任一运行时的对象 / 已落盘的 dict / 已归一的 :data:`NormMsg`）归一。

    **必须幂等**：典型用法是 ``extract_tool_calls(load_history(...))``，而 ``load_history`` 已经
    归一过一遍。不认已归一形态的话，第二遍会把 ``NormMsg`` 当成 AgentScope 的落盘 dict——它有
    ``role`` 没有 ``content``，于是文本与工具调用**双双被洗成空**，且一声不响（症状是轨迹恒空，
    judge 把每条 query 都判成「一个工具没调」）。
    """
    if isinstance(msg, dict) and "tool_calls" in msg and "text" in msg and "content" not in msg:
        return {
            "role": str(msg.get("role") or ""),
            "text": msg.get("text") or "",
            "tool_calls": list(msg.get("tool_calls") or []),
        }

    if isinstance(msg, dict) and "type" in msg and isinstance(msg.get("data"), dict):
        # LangChain ``messages_to_dict`` 的落盘形态
        data = msg["data"]
        role = _ROLE_ALIASES.get(str(msg.get("type")), str(msg.get("type")))
        content = data.get("content")
        return {
            "role": role,
            "text": content if isinstance(content, str) else "",
            "tool_calls": [
                {"name": tc.get("name", "?"), "args": tc.get("args") or {}}
                for tc in (data.get("tool_calls") or [])
            ],
        }

    if isinstance(msg, dict):
        # AgentScope ``Msg.model_dump()`` 的落盘形态
        role = str(msg.get("role") or "assistant")
        content = msg.get("content")
        if isinstance(content, list):
            text, calls = _from_blocks(content)
        else:
            text, calls = (content if isinstance(content, str) else ""), []
        return {"role": role, "text": text, "tool_calls": calls}

    # 运行时对象：AgentScope ``Msg`` 的 content 是 blocks 列表；LangChain 的是 str + 顶层 tool_calls
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        text, calls = _from_blocks(content)
        role = str(getattr(msg, "role", "") or "assistant")
        return {"role": role, "text": text, "tool_calls": calls}

    raw_role = str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")
    return {
        "role": _ROLE_ALIASES.get(raw_role, raw_role),
        "text": content if isinstance(content, str) else "",
        "tool_calls": [
            {"name": tc.get("name", "?"), "args": tc.get("args") or {}}
            for tc in (getattr(msg, "tool_calls", None) or [])
        ],
    }


def normalize_messages(messages: Any) -> list[NormMsg]:
    """批量归一。``messages`` 为空 / None 时返回空列表（评测脚本常拿到空轨迹）。"""
    return [normalize_message(m) for m in (messages or [])]


def extract_tool_calls(messages: Any) -> list[dict[str, Any]]:
    """从消息轨迹里抽出（父 loop 的）工具调用序列，按调用顺序。

    自洽地从 messages 取，不依赖 monitor/WS（离线评测脚本没有连接）。子 Agent 的内部调用在
    各自 thread 的 messages 里、不进父序列，这里看到的是主 loop 的编排轨迹——正是 P1 要评的对象。
    """
    calls: list[dict[str, Any]] = []
    for norm in normalize_messages(messages):
        calls.extend(norm["tool_calls"])
    return calls


def last_assistant_text(messages: Any) -> str:
    """最后一条有正文的 assistant 消息文本（找不到给空串）。"""
    for norm in reversed(normalize_messages(messages)):
        if norm["role"] == "assistant" and norm["text"]:
            return str(norm["text"])
    return ""


def load_history(path: str | Path) -> list[NormMsg]:
    """读 ``history.json``（两套运行时的落盘格式都吃），返回中立轨迹；文件不存在给空列表。

    不在这里做「哪个格式更可信」的判断——同一个 ``session_dir`` 只会被一个运行时写过，
    格式由 :func:`normalize_message` 逐条自识别即可。
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return normalize_messages(raw if isinstance(raw, list) else [])
