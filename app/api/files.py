"""文件接口：产物下载 / 参考图上传 / 上传图回读。从 ``server.py`` 拆出。"""

from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.api.auth import (
    get_current_user_id,
)
from app.api.guards import guard_thread, safe_session_dir
from app.tools.image_understand import sniff_image_mime
from app.utils.env import env_int
from app.utils.path_utils import (
    OUTPUT_ROOT,
    UPLOAD_ROOT,
    safe_join,
)

router = APIRouter()


# 上传文件大小上限（参考图通常是截图；防一把超大文件打爆磁盘/内存）。
# **与 image_understand 读同一个 env**：两处各写一个数字的话，中间地带的图会「传得上去却看不了」——
# 上传口放行 9MB，工具侧按 8MB 判超限降级，用户只看到「传成功了但 Agent 说没看到图」。
MAX_UPLOAD_BYTES = env_int("UPLOAD_MAX_IMAGE_MB", 8) * 1024 * 1024


@router.get("/api/files/{thread_id}/{filename:path}")
async def download_file(
    thread_id: str, filename: str, auth_uid: str | None = Depends(get_current_user_id)
) -> FileResponse:
    """下载某次会话产物（summary.md / result.json）。

    ``filename`` 用 ``:path`` 转换器（允许子目录形式的名字），**正因如此** ``safe_join`` 才是
    真正起作用的防线：``../../`` 这类越权拼接会被它拦下返回 400，而不是靠路由「不匹配斜杠」
    侥幸挡住。

    ``safe_join`` 挡的是「越出目录」，属主校验挡的是「合法路径但不是你的会话」——
    两道防线管的是两件事，缺一不可。
    """
    await guard_thread(thread_id, auth_uid)
    session_dir = safe_session_dir(OUTPUT_ROOT, thread_id)
    if not session_dir.exists():
        raise HTTPException(404, "会话不存在")
    try:
        target = safe_join(session_dir, filename)
    except ValueError as exc:  # 路径穿越企图：当 400 拒绝，不暴露内部路径
        raise HTTPException(400, "非法文件名") from exc
    if not target.is_file():
        raise HTTPException(404, f"文件不存在：{filename}")
    return FileResponse(target, filename=target.name)


@router.post("/api/upload")
async def upload_file(
    thread_id: str = Form(...),
    file: UploadFile = File(...),
    auth_uid: str | None = Depends(get_current_user_id),
) -> dict[str, str]:
    """上传参考图（如复刻款截图）到本次会话目录 ``uploaded/<thread_id>/``。

    两道最小防护：``safe_join`` 净化文件名（恶意 ``../../etc/passwd`` 落不出上传目录）+ 大小
    上限（超限不落盘）。**注意**：这里先整文件读进内存再校验大小，挡的是「写爆磁盘」，
    挡不住「读爆内存」——真要防大文件得在读之前看 Content-Length / 流式分块校验，那属生产化
    硬化（限流 / 类型白名单同级），不在本项目主线。Starlette 的 UploadFile 超阈值会自动落临时
    文件而非全驻内存，已缓解大半。

    属主校验先于读文件：别人的会话目录不给写（否则可以往他的会话里塞图）。

    类型白名单：上传的图会被 image_understand 转 base64 送进 VL 模型，所以在**入口**就按
    magic bytes 认图——不认扩展名（改个名就绕过），不认 Content-Type（客户端随便填）。挡在这里，
    而不是等 provider 回一个 400 才知道用户传了个 PDF。"""
    await guard_thread(thread_id, auth_uid)
    # thread_id 来自表单、完全可控：**先**校验路径合法（否则 ../ 会建到 root 外），再读文件——
    # 路径都非法了就不必把请求体读进内存，且「非法会话标识」的返回码不会被后面的类型校验掩盖成 415。
    upload_dir = safe_session_dir(UPLOAD_ROOT, thread_id)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件过大（上限 {MAX_UPLOAD_BYTES // 1024 // 1024}MB）")
    if not sniff_image_mime(raw):
        raise HTTPException(415, "只支持图片（jpg / png / webp / gif / bmp）")
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        target = safe_join(upload_dir, file.filename or "upload.bin")
    except ValueError as exc:
        raise HTTPException(400, "非法文件名") from exc
    # 落盘是阻塞 IO，挪到线程池，别卡住事件循环（同 loop 还在推其他任务的事件 / 跑 agent）。
    await asyncio.to_thread(target.write_bytes, raw)
    return {"status": "ok", "filename": target.name}


@router.get("/api/uploads/{thread_id}/{filename:path}")
async def download_upload(
    thread_id: str, filename: str, auth_uid: str | None = Depends(get_current_user_id)
) -> FileResponse:
    """取回本会话上传的参考图，供前端在对话气泡里回显。

    与 ``/api/files`` 同构、但**根目录不同**（``uploaded/`` 而非 ``output/``）：那个口服务的是
    Agent 产出的结论文件，这个口服务的是用户传上来的输入。两道防线照旧——``safe_join`` 挡路径
    穿越，``_guard_thread`` 挡「路径合法但不是你的会话」（否则换个 thread_id 就能翻别人上传的图，
    而图往往比文字更私人）。

    为什么回看必须回服务端取、而不是前端缓一份 blob：blob URL 活不过一次刷新，而「我当时发的
    那张图」是对话的一部分——用户点回一段旧会话，图该还在。
    """
    await guard_thread(thread_id, auth_uid)
    upload_dir = safe_session_dir(UPLOAD_ROOT, thread_id)
    if not upload_dir.exists():
        raise HTTPException(404, "会话不存在")
    try:
        target = safe_join(upload_dir, filename)
    except ValueError as exc:
        raise HTTPException(400, "非法文件名") from exc
    if not target.is_file():
        raise HTTPException(404, f"图片不存在：{filename}")
    return FileResponse(target, filename=target.name)
