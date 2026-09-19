"""路径工具：统一解析上传 / 输出 / 会话目录，并防 ``../`` 路径穿越。

约定（CLAUDE.md §6.3）：
- 任务输出 → ``<ARTIFACT_ROOT>/output/<thread_id>/``
- 用户上传 → ``<ARTIFACT_ROOT>/uploaded/<thread_id>/``
- 读用户可控文件名时一律走 :func:`safe_join`，避免 ``../../etc/passwd`` 越权。

**为什么两个根要从同一个 ``ARTIFACT_ROOT`` 派生（阶段 1-4）。** 跑 Agent 的是 worker 进程，而这些
产物的读者是 API 进程（``GET /api/files`` 取 summary.md、``/api/uploads`` 取参考图、会话恢复读
``session.json``）。两个进程各自用「自己那份代码的项目根」算路径时，只要它们不在同一个文件系统位置
（两个容器、两台机器），worker 写完的东西 API 一律 404——而且不报错，表现为「产物莫名其妙没了」。
拆成一个可配置的根，部署侧只要把这一个路径指向同一个卷，两边就对得上；本地开发不配它，默认仍是
项目根，行为与从前一字不差。

跨主机（多台机器各挂各的盘）要的是 NFS / 对象存储，不是这个变量能解决的，本期不做。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根 = 本文件(app/utils/path_utils.py)向上三级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# **这里要自己 load 一次 .env**：本模块被 db / api / tools 到处 import，谁先 import 不定，而
# ``load_dotenv`` 此前只挂在 app.agent.llm 顶层。不 load 的话，本地开发写在 .env 里的
# ARTIFACT_ROOT 会时灵时不灵（取决于 import 顺序）——容器里是真环境变量，反而不受影响，于是这
# 种偏差只在本地出现、最难查。load_dotenv 默认不覆盖已存在的环境变量，重复调用无副作用。
load_dotenv()

# 会话产物的根。默认项目根 = 与改造前逐字相同的 output/ 与 uploaded/。
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT") or PROJECT_ROOT)
UPLOAD_ROOT = ARTIFACT_ROOT / "uploaded"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"


def ensure_session_dir(thread_id: str) -> Path:
    """获取或创建本次任务的输出目录 ``output/<thread_id>/``。"""
    session_dir = OUTPUT_ROOT / thread_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def ensure_upload_dir(thread_id: str) -> Path:
    """获取或创建本次任务的上传目录 ``uploaded/<thread_id>/``。"""
    upload_dir = UPLOAD_ROOT / thread_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def safe_join(base: Path, *parts: str) -> Path:
    """在 ``base`` 下安全拼接路径，越权（解析结果跳出 base）则抛 ``ValueError``。"""
    base_resolved = base.resolve()
    target = (base_resolved / Path(*parts)).resolve()
    # 用 is_relative_to 精确判断从属关系，避免前缀字符串误判（如 /a/b vs /a/bc）。
    if not target.is_relative_to(base_resolved):
        raise ValueError(f"路径越权: {target} 不在 {base_resolved} 内")
    return target
