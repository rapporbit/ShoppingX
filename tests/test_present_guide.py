"""S3：present_guide 选购指南工具（终结）+ 它的 final_text 并回。

验收分三档：
1. **排版**：结构化入参 → 一份可读的 Markdown（块间空行、块内不空行，来源成可点链接）。
2. **入参容错与清洗**：模型把 sections / sources 写成 JSON 字符串或映射照样收；非 http 链接丢掉
   并如实回报条数（模型以为自己列了出处、实际被吞掉，答案里「据某评测」就成了无出处断言）。
3. **接线三处**：进 TERMINAL_TOOLS（批次原子闸随之覆盖）、进工具表、markdown 并回 final_text。
"""

from __future__ import annotations

from typing import Any

import app.tools.present_guide as mod
from app.tools.present_guide import GuideOutput, GuideSection, GuideSource, _render


def _sections() -> list[dict[str, Any]]:
    return [
        {"title": "清洁力", "points": ["声波每分钟 3 万次以上", "刷头小一点更够得着后牙"]},
        {"title": "续航", "points": ["按一天两次算，30 天以上才够出差用"]},
    ]


async def test_renders_markdown_with_sections_and_sources() -> None:
    """排版：标题 / 假设 / 逐节要点 / 结尾 / 来源，来源渲染成 Markdown 链接。"""
    out = await mod.present_guide.ainvoke(
        {
            "topic": "电动牙刷怎么挑",
            "sections": _sections(),
            "assumptions": ["按一个人用估的"],
            "sources": [{"title": "某评测", "url": "https://example.com/a"}],
            "closing": "预算 300 以内优先看清洁力。",
        }
    )
    md = out.markdown
    assert md.startswith("### 电动牙刷怎么挑")
    assert "#### 1. 清洁力" in md and "#### 2. 续航" in md
    assert "- 声波每分钟 3 万次以上" in md
    assert "按一个人用估的" in md
    assert "预算 300 以内优先看清洁力。" in md
    assert "**参考来源**\n- [某评测](https://example.com/a)" in md
    assert out.note == ""
    # 块内不空行：要点紧跟标题，否则 marked 会渲染成松散列表（每项套一层 <p>），行距翻倍。
    assert "#### 1. 清洁力\n- 声波每分钟 3 万次以上" in md


async def test_coerces_stringified_and_mapping_args() -> None:
    """入参容错：sections 写成 ``{标题: [要点]}`` 映射、sources 写成裸 url 列表照样收。"""
    out = await mod.present_guide.ainvoke(
        {
            "topic": "行李箱怎么挑",
            "sections": {"箱体": ["PC 比 ABS 抗摔"]},
            "sources": ["https://example.com/b"],
        }
    )
    assert [s.title for s in out.sections] == ["箱体"]
    assert out.sections[0].points == ["PC 比 ABS 抗摔"]
    assert [s.url for s in out.sources] == ["https://example.com/b"]
    # 没给标题的来源退回域名，不把裸 url 整条摊在正文里。
    assert "- [example.com](https://example.com/b)" in out.markdown


async def test_drops_non_http_sources_and_reports_count() -> None:
    """非 http(s) 来源丢掉并回报条数；重复 url 去重但不算「丢弃」（不是模型的错）。"""
    out = await mod.present_guide.ainvoke(
        {
            "topic": "t",
            "sections": _sections(),
            "sources": [
                {"url": "https://example.com/a"},
                {"url": "https://example.com/a"},
                {"url": "某评测网站"},
                {"url": ""},
            ],
        }
    )
    assert [s.url for s in out.sources] == ["https://example.com/a"]
    assert "已忽略 2 条来源" in out.note


async def test_truncates_sections_points_and_sources(monkeypatch: Any) -> None:
    """三处封顶：节数 / 每节要点数 / 来源数，超出截断而不是原样吐出去。"""
    monkeypatch.setattr(mod, "GUIDE_MAX_SECTIONS", 2)
    monkeypatch.setattr(mod, "GUIDE_MAX_POINTS", 1)
    monkeypatch.setattr(mod, "GUIDE_MAX_SOURCES", 1)
    out = await mod.present_guide.ainvoke(
        {
            "topic": "t",
            "sections": [{"title": f"s{i}", "points": ["p1", "p2"]} for i in range(4)],
            "sources": [{"url": f"https://example.com/{i}"} for i in range(3)],
        }
    )
    assert [s.title for s in out.sections] == ["s0", "s1"]
    assert out.sections[0].points == ["p1"]
    assert len(out.sources) == 1
    assert "已忽略 2 条来源" in out.note


async def test_empty_sections_still_terminates_with_note() -> None:
    """空 sections 不抛错：抛出去只会让模型换个工具把空内容再讲一遍。note 要说清它给了什么。"""
    out = await mod.present_guide.ainvoke({"topic": "t", "closing": "先说下预算？"})
    assert out.sections == []
    assert "sections 为空" in out.note
    assert out.markdown == "### t\n\n先说下预算？"


def test_render_skips_empty_blocks() -> None:
    """纯函数：没给的块不留空行；标题空但有要点的节保留（标题不是必需的信息）。"""
    md = _render(
        GuideOutput(sections=[GuideSection(title="", points=["只有要点"])], sources=[GuideSource()])
    )
    assert md == "#### 1.\n- 只有要点"


def test_is_terminal_and_registered() -> None:
    """接线：终结集合（批次原子闸靠它）+ 工具表（模型看得见才调得到）。"""
    from app.agent.constants import TERMINAL_TOOLS, is_terminal_call
    from app.agent.tool_registry import TOOLS_BY_NAME

    assert "present_guide" in TERMINAL_TOOLS
    assert is_terminal_call("present_guide") is True
    assert "present_guide" in TOOLS_BY_NAME
