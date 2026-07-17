"""路径工具：统一解析上传 / 输出 / 会话目录，并防 ``../`` 路径穿越。

约定（CLAUDE.md §6.3）：
- 任务输出 → ``output/<thread_id>/``
- 用户上传 → ``uploaded/<thread_id>/``
- 读用户可控文件名时一律走 :func:`safe_join`，避免 ``../../etc/passwd`` 越权。
"""

from pathlib import Path

# 项目根 = 本文件(app/utils/path_utils.py)向上三级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_ROOT / "uploaded"
OUTPUT_ROOT = PROJECT_ROOT / "output"


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
