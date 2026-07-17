"""image_understand（M20 图搜）：格式嗅探 / 降级路径 / 解析容错 / 会话目录定位。

看图这条腿的每个失败点都必须**降级而不是崩**——用户传了张糊图或没配 VL 模型，购物流程该照常
按文字意图往下走，而不是整条任务挂掉。这里逐个把失败点钉住。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.tools.image_understand import (
    ImageUnderstanding,
    _build_output,
    _extract_json,
    image_understand,
    sniff_image_mime,
)
from app.utils.thread_ctx import thread_scope

_PNG = b"\x89PNG\r\n\x1a\n" + b"body"
_JPEG = b"\xff\xd8\xff" + b"body"
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"body"


class TestSniffMime:
    def test_recognizes_common_formats(self) -> None:
        assert sniff_image_mime(_PNG) == "image/png"
        assert sniff_image_mime(_JPEG) == "image/jpeg"
        # WebP 魔数是分段的（RIFF....WEBP），最容易写错的一个——单独钉住。
        assert sniff_image_mime(_WEBP) == "image/webp"

    def test_rejects_non_image(self) -> None:
        """扩展名和 Content-Type 都可伪造，只有字节头作数。"""
        assert sniff_image_mime(b"%PDF-1.7") == ""
        assert sniff_image_mime(b"") == ""


class TestExtractJson:
    def test_parses_bare_json(self) -> None:
        assert _extract_json('{"category": "backpack"}') == {"category": "backpack"}

    def test_parses_fenced_json(self) -> None:
        """VL 模型很爱套 markdown 代码块，哪怕 prompt 明说不要。"""
        text = 'Here you go:\n```json\n{"category": "mug"}\n```\n'
        assert _extract_json(text) == {"category": "mug"}

    def test_returns_empty_on_garbage(self) -> None:
        assert _extract_json("对不起，我看不清这张图。") == {}


class TestBuildOutput:
    def test_degrades_when_not_a_product(self) -> None:
        """图里不是商品时不能拿 "not a product" 当检索词去搜——降级，让主 loop 去问用户。"""
        out = _build_output("x.png", {"subject": "not a product"})
        assert out.degraded
        assert not out.search_query

    def test_synthesizes_search_query_when_missing(self) -> None:
        """模型漏给 search_query 时用识别出的要素兜底拼一个，保证下游一定有检索词可用。"""
        out = _build_output(
            "x.png",
            {
                "subject": "一只黑色皮质双肩包",
                "category": "backpack",
                "color": "black",
                "material": "leather",
                "keywords": ["leather backpack"],
            },
        )
        assert not out.degraded
        assert "backpack" in out.search_query
        assert "black" in out.search_query

    def test_splits_comma_joined_keywords(self) -> None:
        """模型可能把数组吐成一坨逗号分隔的字符串。"""
        out = _build_output("x.png", {"category": "mug", "keywords": "ceramic, white, matte"})
        assert out.keywords == ["ceramic", "white", "matte"]

    def test_flags_multi_subject(self) -> None:
        """图里有多件可买商品时置 multi_subject —— 主 loop 据此决定要不要 ask_user。"""
        out = _build_output(
            "look.png",
            {
                "subject": "藏青色工装夹克",
                "category": "jacket",
                "objects": ["jacket: 藏青工装夹克", "sneakers: 白色板鞋", "backpack: 黑色双肩包"],
            },
        )
        assert out.multi_subject is True
        assert len(out.objects) == 3

    def test_single_object_is_not_ambiguous(self) -> None:
        """只有一件商品就别把主 loop 逼去问用户 —— 多余的一轮往返就是延迟。"""
        out = _build_output(
            "mug.png", {"subject": "白色马克杯", "category": "mug", "objects": ["mug: 白色马克杯"]}
        )
        assert out.multi_subject is False

    def test_no_objects_field_stays_unambiguous(self) -> None:
        """VL 没吐 objects（prompt 不是 schema 强约束）时保持旧行为：直接搜，不问。"""
        out = _build_output("x.png", {"subject": "白色马克杯", "category": "mug"})
        assert out.multi_subject is False
        assert out.objects == []


class TestImageUnderstandTool:
    async def _run(self, filename: str, tmp: Path) -> ImageUnderstanding:
        with thread_scope("t-img", tmp):
            result = await image_understand.ainvoke({"filename": filename})
        assert isinstance(result, ImageUnderstanding)
        return result

    async def test_degrades_without_vision_model(self, monkeypatch: Any) -> None:
        """没配 LLM_VISION → 不崩，如实说「看图不可用」，购物流程照常按文字走。"""
        monkeypatch.delenv("LLM_VISION", raising=False)
        tmp = Path(tempfile.mkdtemp())
        out = await self._run("ref.png", tmp)
        assert out.degraded
        assert "LLM_VISION" in out.note

    async def test_degrades_on_missing_file(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("LLM_VISION", "fake-vl")
        tmp = Path(tempfile.mkdtemp())
        out = await self._run("nope.png", tmp)
        assert out.degraded
        assert "读取上传图失败" in out.note

    async def test_degrades_on_path_traversal(self, monkeypatch: Any) -> None:
        """文件名是用户可控输入：../ 想读会话目录外的文件 → safe_join 抛错 → 降级，不泄露。"""
        monkeypatch.setenv("LLM_VISION", "fake-vl")
        tmp = Path(tempfile.mkdtemp())
        out = await self._run("../../etc/passwd", tmp)
        assert out.degraded

    async def test_degrades_on_non_image_bytes(self, monkeypatch: Any, tmp_path: Path) -> None:
        """混过上传口的非图片文件（或用户直接把 PDF 改名塞进来）不送模型，本地就挡下。"""
        monkeypatch.setenv("LLM_VISION", "fake-vl")
        from app.tools import image_understand as mod

        upload_dir = tmp_path / "uploads" / "t-img"
        upload_dir.mkdir(parents=True)
        (upload_dir / "evil.png").write_bytes(b"%PDF-1.7 not an image")
        monkeypatch.setattr(mod, "UPLOAD_ROOT", tmp_path / "uploads")

        out = await self._run("evil.png", tmp_path / "session" / "t-img")
        assert out.degraded
        assert "不是可识别的图片格式" in out.note


@pytest.mark.parametrize("payload", [_PNG, _JPEG, _WEBP])
async def test_reads_all_supported_formats(
    payload: bytes, monkeypatch: Any, tmp_path: Path
) -> None:
    """三种主流格式都要能走到「送进模型」那一步（这里 mock 掉模型，只验前面的读图链路）。"""
    monkeypatch.setenv("LLM_VISION", "fake-vl")
    from app.tools import image_understand as mod

    upload_dir = tmp_path / "uploads" / "t-img"
    upload_dir.mkdir(parents=True)
    (upload_dir / "ref.png").write_bytes(payload)
    monkeypatch.setattr(mod, "UPLOAD_ROOT", tmp_path / "uploads")

    class _FakeResp:
        content = '{"subject": "杯子", "category": "mug", "search_query": "white ceramic mug"}'

    class _FakeLLM:
        # 签名要收下 config：真实调用会挂 usage callback 记 token 账（漏账则成本闸/配额少算）。
        async def ainvoke(self, _messages: Any, config: Any = None) -> Any:
            return _FakeResp()

    monkeypatch.setattr(mod, "get_vision_llm", lambda: _FakeLLM())

    with thread_scope("t-img", tmp_path / "session" / "t-img"):
        out = await image_understand.ainvoke({"filename": "ref.png"})
    assert not out.degraded
    assert out.category == "mug"
    assert out.search_query == "white ceramic mug"
