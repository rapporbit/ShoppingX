"""Agent Skill（批 4-3）：目录加载 / 发放范围 / 注入位与批 4-2 策略块的相对次序。

不测「模型有没有按 description 触发」——那是模型行为，本单不跑真实 LLM。这里测的是**机制**：
三个 SKILL.md 真被框架的 ``LocalSkillLoader`` 读到、目录块真拼进了 system prompt、worker
拿不到、以及策略块与 skill 块的先后关系被钉死（缓存账建立在这个次序上，见 app/agent/skills.py）。
"""

import frontmatter
import pytest

from app.agent.skills import SKILL_VIEWER_TOOL_NAME, SKILLS_DIR, skill_loaders
from app.agent.tool_registry import build_toolkit

EXPECTED_SKILLS = {"cross-border-duty", "bundle-planning", "image-shopping"}


def test_skill_dirs_are_exactly_the_three() -> None:
    dirs = {p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()}
    assert dirs == EXPECTED_SKILLS


@pytest.mark.parametrize("name", sorted(EXPECTED_SKILLS))
def test_frontmatter_name_matches_dir(name: str) -> None:
    """``name`` 必须与目录同名，``description`` 非空。

    框架用 frontmatter 的 name 做 ``Skill(skill="…")`` 的键，用目录名定位文件。两者不一致时
    模型按目录名调就找不到、按 name 调则读不到文件——而 loader 只会静默跳过，没有红灯。
    """
    post = frontmatter.load(SKILLS_DIR / name / "SKILL.md")
    assert post.get("name") == name
    assert len(str(post.get("description") or "")) > 40
    assert post.content.strip()


async def test_loader_loads_all_three() -> None:
    """``scan_subdir=True`` 是必须的：默认只扫目录自身，会静默加载到 0 个。"""
    (loader,) = skill_loaders("main")
    skills = await loader.list_skills()
    assert {s.name for s in skills} == EXPECTED_SKILLS
    assert all(s.markdown.strip() for s in skills)


async def test_main_toolkit_injects_skill_directory() -> None:
    toolkit = await build_toolkit("main")
    block = await toolkit.get_skill_instructions(["basic"])
    assert block is not None
    for name in EXPECTED_SKILLS:
        assert f"<name>{name}</name>" in block
    # 目录块只带 name/description/dir，**不带正文**——正文按需读才是 skill 的成本模型。
    assert "Multiple-Choice Knapsack" not in block


async def test_skill_viewer_tool_is_available_to_main_only() -> None:
    main = await build_toolkit("main")
    assert SKILL_VIEWER_TOOL_NAME in {s["function"]["name"] for s in await main.get_tool_schemas()}
    for role in ("search", "trade"):
        toolkit = await build_toolkit(role)
        names = {s["function"]["name"] for s in await toolkit.get_tool_schemas()}
        assert SKILL_VIEWER_TOOL_NAME not in names
        assert await toolkit.get_skill_instructions(["basic"]) is None


async def test_skill_viewer_is_whitelisted() -> None:
    """内置阅读器要在 L1 白名单里——它今天走不到那道闸，但判据必须是对的。"""
    from app.security.tool_whitelist import allowed_tools, validate_tool_call

    allowed_tools.cache_clear()
    assert validate_tool_call(SKILL_VIEWER_TOOL_NAME)


def test_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLS_ENABLED", "0")
    assert skill_loaders("main") == []


def test_workers_never_get_skills() -> None:
    for role in ("search", "trade"):
        assert skill_loaders(role) == []


async def test_strategy_block_precedes_skill_block_in_final_prompt() -> None:
    """口径钉死：``[基线正文][策略块]`` 在前（本仓装配期拼），``<agent-skills>`` 在后（框架拼）。

    这条次序不是审美：缓存账建立在它上面（skill 块是静态的，策略块按 query 变；静态的排在
    变的后面，等于「反正已经断了」，不额外多亏）。次序若反过来，说明有人接管了框架的注入位，
    那笔账要重算。见 app/agent/skills.py 模块 docstring。
    """
    from app.agent.agents import build_main_agent
    from app.memory.strategies import Strategy, force_strategies

    strategy = Strategy(
        category="预算陷阱",
        trigger="用户给了明确预算",
        trigger_keywords=["预算"],
        actions=["先算到手价再排序"],
        evidence=["q26"],
    )
    with force_strategies([strategy]):
        agent, _ = await build_main_agent(original_query="预算 300 买个旅行包")
    prompt = await agent._get_system_prompt()
    i_strategy = prompt.index("<learned_strategies>")
    i_skills = prompt.index("<agent-skills>")
    assert i_strategy < i_skills
    # 基线正文仍是整段的逐字前缀（追加语义没被谁改成改写）。
    assert prompt.startswith(agent._system_prompt[:200])
