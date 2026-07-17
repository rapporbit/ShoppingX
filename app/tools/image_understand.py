"""image_understand —— 看懂用户上传的参考图，翻译成可检索的结构化文本。

**为什么是「工具」而不是把图塞进主 loop**：主模型（``LLM_MAIN``）是纯文本的，多模态消息塞进
主 loop 的 messages 只会报错或被静默忽略。所以图片被关在这个工具内部——工具里调 VL 模型
（``LLM_VISION``），主 loop 只看到它吐出的文本结论。副作用是正好符合既有架构里「工具中间产物
不污染主上下文」的约定：一张几百 KB 的图不会挤进后续每一轮的 prompt。

**这不是「以图搜图」，是「看图说话再搜文本」**（诚实标注）：VL 模型把图读成品类/颜色/材质/风格
+ 检索词，检索词并入既有的 BGE-M3 文本召回链路。因此它找到的是**同品类同属性的相似品**，不是
「一模一样的那只」。对「想买类似这种风格的东西」够用；同款复刻做不到，别宣称。

**检索词优先出英文**：商品库正文以英文为主（amazon 屠版），英文词打英文库更准——虽然 BGE-M3
本身跨语言，但同语种检索的召回质量实测更稳。
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.agent.llm import get_vision_llm, vision_enabled
from app.agent.token_budget import charge_tool_llm_usage
from app.api import monitor
from app.api.context import get_session_dir, get_thread_id
from app.utils.env import env_int
from app.utils.path_utils import UPLOAD_ROOT, safe_join

# 按 magic bytes 认图（而非信任用户给的扩展名）：这张图要转 base64 送进 VL 模型，
# 一个改名成 .jpg 的 PDF 只会让 provider 报 400，不如在这里就拦下并给出可读原因。
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"),
]


def sniff_image_mime(raw: bytes) -> str:
    """从字节头认出图片类型，认不出返回空串（调用方据此降级）。"""
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            return mime
    # WebP 的魔数是分段的：RIFF....WEBP（中间 4 字节是文件长度）。
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return ""


class ImageUnderstanding(BaseModel):
    """image_understand 的结构化返回（字段直接喂给 planner / item_search）。"""

    filename: str
    subject: str = ""  # 主体物品是什么（一句话）
    category: str = ""  # 品类（英文，对齐商品库）
    color: str = ""
    material: str = ""
    style: str = ""
    attributes: list[str] = Field(default_factory=list)  # 显著属性（图案 / 形制 / 细节）
    keywords: list[str] = Field(default_factory=list)  # 建议检索词（英文优先）
    search_query: str = ""  # 拼好的检索词串，可直接交给 item_search
    note: str = ""  # 降级 / 异常说明
    degraded: bool = False

    # 多主体消歧（工业上这一步叫主体检测，这里用「如实枚举 + 交给主 loop 问」代替）：
    # 一张模特图上衣服、鞋、包都是可买的商品，VL 自己挑一个 subject 很可能挑错，而它**没有对话
    # 上下文**、判断不了用户到底要哪个。所以工具只负责如实枚举，要不要追问由主 loop 结合用户
    # 文本决定（见 main_agent 的 <reference_images> 块）。
    objects: list[str] = Field(default_factory=list)  # 图中可独立购买的商品，≥2 即存在歧义
    multi_subject: bool = False  # 由 objects 长度派生，给主 loop 一个直接可读的信号


_PROMPT = """你是电商选购助手的视觉理解模块。
用户上传了一张参考图，想买「和图里这个东西类似的商品」。

请看图并**只输出一个 JSON 对象**（不要任何解释文字、不要 markdown 代码块），字段如下：
- objects: 图中**可以独立作为购买目标**的商品数组，每项写成 "英文品类: 一句话中文描述"（如
  "backpack: 黑色帆布双肩包"）。判断标准：它本身是一件用户可能想买的商品。
  **要算进来**：模特身上的每件单品（上衣 / 裤子 / 鞋 / 包 / 表 / 帽），桌面平铺的每件物品。
  **不要算**：人体本身、背景陈设与环境（墙、地板、植物、道具）、看不清的远景物件、商品的组成
  部件（拉链、纽扣、鞋带不是独立商品）。
  只有一件就只给一项；宁可少列也不要把背景硬凑成商品。
- subject: 图中最主要的那件商品是什么，一句话中文（objects 里最显著的那个）
- category: 商品品类，**英文**，用电商站常见的叫法（如 "backpack" / "smart watch" / "ceramic mug"）
- color: 主色调，英文
- material: 材质，英文；看不出就留空字符串
- style: 风格，英文（如 "minimalist" / "vintage" / "outdoor rugged"）
- attributes: 显著属性数组，英文，3-6 个（形制、图案、功能细节等肉眼可见的）
- keywords: 建议检索词数组，**英文**，3-6 个，用来在英文商品库里搜到同类商品
- search_query: 把上面要素拼成一句可直接用于商品检索的英文短语

**除 objects 外，以上所有字段都只描述 subject 那一件商品**——图里有多件时不要把它们的属性拌在
一起（别把上衣的颜色和鞋的材质写进同一组字段）。用户到底要哪件由后续环节决定，你只管如实枚举。

只描述**你确实看得见的**东西。看不出的字段留空，不要编造品牌、价格、材质。
如果图里根本不是商品（截图、文档、人像、风景等），把 subject 写成 "not a product"，其余留空。"""


def _extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象。

    VL 模型不像文本模型那样能稳定吃 json_schema / function calling（DashScope 的 qwen-vl 尤其），
    所以这里不赌结构化输出，而是要求它吐 JSON 文本再宽松解析：先直接 parse，失败则剥 markdown
    代码块，再失败则正则抓第一个 {...}。三层都失败由调用方降级——比整条链路崩了强。
    """
    text = text.strip()
    try:
        return dict(json.loads(text))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return dict(json.loads(fenced.group(1)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return dict(json.loads(brace.group(0)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {}


def _resolve_upload_path(filename: str) -> Path:
    """把文件名解析到本会话的上传目录下的绝对路径。

    **必须按 session_dir 而不是 thread_id 定位**（fork 上下文的坑）：fork 出的子 Agent 有自己的
    thread_id（``uploaded/<子thread_id>/`` 根本不存在，子 Agent 一读图就扑空），但 session_dir 是
    **继承父的**（CLAUDE.md §6.3）。所以拿 session_dir 的目录名反推出会话属主，主 / 子 loop 才能
    读到同一份上传图。没有 session_dir（如裸测）时才退回 thread_id。

    文件名来自用户，一律走 safe_join 防 ``../`` 穿越。
    """
    session_dir = get_session_dir()
    owner = session_dir.name if session_dir else (get_thread_id() or "")
    if not owner:
        raise ValueError("当前无会话上下文，无法定位上传目录")
    return safe_join(UPLOAD_ROOT / owner, filename)


def _degraded(filename: str, note: str) -> ImageUnderstanding:
    """构造一个降级返回：图看不成也别崩主 loop，如实告诉它「这张图没读到」。"""
    return ImageUnderstanding(filename=filename, note=note, degraded=True)


@tool
async def image_understand(filename: str) -> ImageUnderstanding:
    """看懂用户上传的参考图，产出品类 / 颜色 / 材质 / 风格 + 可直接检索的英文关键词。

    何时调用：**常规情况下你不需要调它**——用户传的图已由系统在开局预读，结论就在上文的
    image_understand 工具消息里，直接用它的 search_query / keywords 检索即可。只有在上文没有
    任何图片识别结果、而用户又确实提到某张参考图时才调用（每调一次都要重新看一遍图，别白花）。
    参数：
      - filename：上传图的文件名（如 "ref.jpg"）。
    注意：它给的是「同类相似品」的检索线索，不是同款复刻；找不到一模一样的属正常。
    """
    await monitor.report_tool_start("image_understand", filename=filename)

    if not vision_enabled():
        out = _degraded(
            filename, "未配置 LLM_VISION，看图能力不可用；请让用户用文字描述想要的商品。"
        )
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    try:
        path = _resolve_upload_path(filename)
        raw = path.read_bytes()
    except (ValueError, OSError) as e:
        out = _degraded(
            filename, f"读取上传图失败（{type(e).__name__}），请让用户重新上传或改用文字描述。"
        )
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    max_mb = env_int("UPLOAD_MAX_IMAGE_MB", 8)
    if len(raw) > max_mb * 1024 * 1024:
        out = _degraded(filename, f"图片超过 {max_mb}MB 上限，未送入模型。")
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    mime = sniff_image_mime(raw)
    if not mime:
        out = _degraded(filename, "这个文件不是可识别的图片格式（支持 jpg/png/webp/gif/bmp）。")
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    try:
        # base64 直传，而不是把商品图的公网 URL 交给 provider 去下载：实测 DashScope 拉
        # Amazon CDN 会被 403 挡下（Failed to download multimodal content）。何况用户上传的
        # 图本来就只在我们盘上，没有公网 URL 可给。
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": _PROMPT},
            ]
        )
        # 挂 usage callback 并记账：这次调用不经 agent middleware（工具内部自己调 LLM），
        # 不挂就是漏账——成本闸与用户每日 credit 配额都会少算这一笔。一张图动辄上千 prompt
        # token，漏掉的不是零头（同 planner / chat_fallback / shopping_summary 的先例）。
        usage_cb = UsageMetadataCallbackHandler()
        resp = await get_vision_llm().ainvoke([message], config={"callbacks": [usage_cb]})
        charge_tool_llm_usage(usage_cb.usage_metadata)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _extract_json(text)
    except Exception as e:  # 外部依赖失败不该崩主 loop，转成可读 note + 标降级
        out = _degraded(
            filename, f"看图模型调用失败（{type(e).__name__}），请让用户改用文字描述商品。"
        )
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    if not data:
        out = _degraded(filename, "看图模型没有返回可解析的结构化结果，请让用户改用文字描述商品。")
        await monitor.report_tool_end(
            "image_understand", degraded=True, result=out.note, filename=filename
        )
        return out

    out = _build_output(filename, data)
    brief = out.note or f"{out.subject}｜检索词：{out.search_query}"
    if out.multi_subject:
        brief += f"｜图中有 {len(out.objects)} 件可买商品，存在歧义"
    await monitor.report_tool_end(
        "image_understand",
        degraded=out.degraded,
        result=brief,
        filename=filename,
    )
    return out


def _as_str_list(value: object, limit: int = 6) -> list[str]:
    """把模型给的字段规整成字符串列表（它可能吐字符串、也可能吐逗号分隔的一坨）。"""
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str) and value.strip():
        items = [p.strip() for p in re.split(r"[,，、]", value) if p.strip()]
    else:
        items = []
    return items[:limit]


def _build_output(filename: str, data: dict) -> ImageUnderstanding:
    """把 VL 模型吐的 dict 收成 ImageUnderstanding，并兜住两种「看了但没看出商品」的情况。"""
    subject = str(data.get("subject", "")).strip()
    keywords = _as_str_list(data.get("keywords"))
    category = str(data.get("category", "")).strip()
    search_query = str(data.get("search_query", "")).strip()

    # 模型判定图里压根不是商品（截图 / 文档 / 人像 / 风景）——如实上报，让主 loop 去问用户，
    # 而不是拿一句 "not a product" 当检索词去搜。
    # 「三个检索信号一个都没有」才算没看出商品：prompt 只是**要求**模型给 category/keywords，不是
    # schema 强约束，它完全可能只给 search_query。少了任一个就判降级，会把明明可用的检索词扔掉。
    if subject.lower() == "not a product" or not (category or keywords or search_query):
        return _degraded(
            filename,
            "这张图里没看出具体商品，请让用户确认上传的是想买的东西，或改用文字描述。",
        )

    # search_query 模型可能不给或给得敷衍，用识别出的要素兜底拼一个，保证下游一定有检索词可用。
    if not search_query:
        parts = [
            str(data.get("color", "")).strip(),
            str(data.get("material", "")).strip(),
            str(data.get("style", "")).strip(),
            category,
        ]
        search_query = " ".join(p for p in parts if p) or " ".join(keywords)

    # 上限放宽到 8：模特图上单品可以不少，截断成 6 会把本该消歧的那一件悄悄抹掉——
    # 宁可多列几项让主 loop 去问，也不要因为截断而漏问。
    objects = _as_str_list(data.get("objects"), limit=8)

    return ImageUnderstanding(
        filename=filename,
        subject=subject,
        category=category,
        color=str(data.get("color", "")).strip(),
        material=str(data.get("material", "")).strip(),
        style=str(data.get("style", "")).strip(),
        attributes=_as_str_list(data.get("attributes")),
        keywords=keywords,
        search_query=search_query,
        objects=objects,
        multi_subject=len(objects) >= 2,
    )
