"""Agent Skill 的发放（批 4-3）—— 框架原生 ``LocalSkillLoader``，不自建 loader。

**Skill 是什么**：``skills/<name>/SKILL.md``，frontmatter 给 ``name`` / ``description``，正文是
一段写给模型看的领域打法。框架把**所有 skill 的 name + description + dir**（不是正文）拼成
``<agent-skills>`` 块接在 system prompt 后面，另外自动挂一个内置只读工具 ``Skill``；模型看
description 判断这次用不用得上，用得上才调 ``Skill(skill="…")`` 把正文读进来。

所以 skill 的成本模型是「**目录常驻、正文按需**」：三个 skill 常驻的只有三行描述（几百
token），一千多字的正文只有真被触发的那一轮才进上下文。这是它区别于「把知识写进 system
prompt」的唯一理由——后者是每轮都付钱。**description 因此是唯一的触发面**：它写不准，正文
写得再好也永远不会被读到。

**发放范围（与 ``tool_registry`` 同一套思路）**：skill 只发给 ``main``。三个 skill 讲的都是主
Agent 的活——到手价口径、槽位规划、图搜流程；SearchAgent 手上只有 ``item_search`` /
``web_search`` 与一条收窄过的 demands，读了也没有对应的工具去执行，只会白白多一个 ``Skill``
工具和三行描述。TradeAgent 同理。**发放范围不是提示词劝退**，这里少给一份，worker 的
Toolkit 里就根本没有那个入口。

**与批 4-2 的 ``on_system_prompt`` 钩子怎么相处**（口径，改这里前先读）：两者拼在 system
prompt 的**不同层**，顺序是框架定的，我们不去抢：

    [本仓定稿 prompt = 基线正文 + on_system_prompt 钩子追加的策略块] ← agents._assemble
      + [<agent-skills> 目录块]                                      ← 框架 _get_system_prompt
      + [workspace 指令（本仓未用）]

即**策略块在前、skill 目录块在后**。三条推论：

1. 两者不会互相覆盖——钩子只拿得到 ``self._system_prompt`` 那一段，压根看不见 skill 块，
   也就删不掉它；反之亦然。
2. 缓存账不亏。skill 块是**静态**的（只随 SKILL.md 文件变），策略块是**按 query 变**的。
   没匹配到策略时追加为空，整段 system prompt 逐字稳定，前缀缓存全程命中；匹配到策略时
   缓存本来就断在策略块那里，后面跟着的 skill 块是「已经断了之后」的字节，不额外多亏。
   反过来把 skill 块塞在策略块前面才是亏的——那要我们自己接管注入，还得放弃框架原生。
3. skill 块**每次模型调用都重新渲染**（框架在 ``_get_system_prompt`` 里现算，会重扫目录 +
   getmtime）。所以**跑任务期间别改 SKILL.md**：改一个字节，这一轮后续所有请求的 system
   prompt 就变了，前缀缓存从头失效。改完重开一轮即可。
"""

from __future__ import annotations

from pathlib import Path

from agentscope.skill import LocalSkillLoader, SkillLoaderBase

from app.utils.env import env_bool
from app.utils.path_utils import PROJECT_ROOT

#: skill 根目录。每个子目录一个 skill，必须含 ``SKILL.md``。
SKILLS_DIR: Path = PROJECT_ROOT / "skills"

#: 拿得到 skill 的角色。见模块 docstring「发放范围」。
SKILL_ROLES: frozenset[str] = frozenset({"main"})

#: 框架内置的 skill 阅读器工具名（``agentscope.tool._builtin.SkillViewer.name``）。
#: 注册了 skill 就会自动出现在可用工具表里，它是只读的、权限永远 ALLOW。
#: 白名单（``app/security/tool_whitelist.py``）要认得它，否则将来这类内置工具一旦接进
#: harness 的工具中间件，第一道闸就会把它当幻觉工具名拒掉。
SKILL_VIEWER_TOOL_NAME = "Skill"


def skill_loaders(role: str = "main") -> list[SkillLoaderBase]:
    """按角色返回 skill loader（``main`` 一个，其余空表）。

    ``scan_subdir=True`` 是必须的：``LocalSkillLoader`` 默认只在**给定目录自身**找
    ``SKILL.md``，而本仓的布局是 ``skills/<name>/SKILL.md``——不开这个开关会静默加载到 0 个
    skill（框架只打一行 info 日志），表现为「写了 skill 但模型从来不知道」。

    目录不存在时返回空表而不是让它去扫一个不存在的路径：loader 自己会 warning 后返回 []，
    但那条 warning 每次模型调用都打一遍，噪音比信息多。
    """
    if role not in SKILL_ROLES:
        return []
    if not env_bool("SKILLS_ENABLED", True):
        return []
    if not SKILLS_DIR.is_dir():
        return []
    return [LocalSkillLoader(str(SKILLS_DIR), scan_subdir=True)]
