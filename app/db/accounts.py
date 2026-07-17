"""账户与归属的读写（M16）——注册、登录校验、认领会话、列出会话、校验属主。

**密码怎么存。** bcrypt 摘要，不可逆：库被拖走也拿不到明文密码（用户往往在别处复用同一个密码，
这是「泄漏一个站 = 泄漏一批账号」的根因）。bcrypt 自带每用户随机盐，且**故意算得慢**——离线爆破
时慢就是防御，攻击者每猜一次都要付出同样的代价。

**登录失败为什么不区分「用户不存在」和「密码错」。** 两种都回同一句「用户名或密码错误」：一旦区分，
登录口就成了「用户名探测器」——攻击者可以先枚举出哪些账号存在，再对着真实账号集中撞库。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import bcrypt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Thread, User

# bcrypt 的输入上限是 72 字节，超出部分**被静默忽略**——不显式拦住的话，一个 100 字符的强密码
# 与它的前 72 字节等价，用户以为自己的长密码更安全，其实不是。故超长直接拒。
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LEN = 8
TITLE_MAX = 60


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        # 库里的摘要脏了（人工改过 / 迁移出错）——当作验证失败，不让异常冒到路由变成 500。
        return False


async def create_user(db: AsyncSession, username: str, password: str) -> User:
    """注册。用户名重复由数据库唯一索引挡（见 models.User），不靠应用层「先查再插」。"""
    user = User(id=uuid.uuid4().hex, username=username, password_hash=hash_password(password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise ValueError("用户名已被占用") from exc
    return user


async def authenticate(db: AsyncSession, username: str, password: str) -> User | None:
    """校验用户名 + 密码，通过返回 User，否则 None（调用方一律回同一句错误，见模块 docstring）。"""
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


async def claim_thread(db: AsyncSession, thread_id: str, user_id: str, title: str) -> None:
    """认领一段会话（起任务时调）：首轮登记归属 + 标题，后续轮只把 ``updated_at`` 顶到最新。

    **认领是一次性的**：已存在的 thread 绝不改 ``user_id``——否则「用别人的 thread_id 发一条消息」
    就成了把他人会话过户到自己名下的越权写。属主校验（assert_owner）挡的是读，这里挡的是写。
    """
    existing = await db.get(Thread, thread_id)
    if existing is not None:
        if existing.user_id != user_id:
            raise PermissionError("无权访问该会话")
        existing.title = existing.title or title[:TITLE_MAX]
        # 显式盖时间戳，把这段会话顶到侧栏最前（"最近聊过"）。不能指望 onupdate：值没真的变过时
        # SQLAlchemy 不认为对象是脏的，压根不会发 UPDATE，onupdate 也就不会触发。
        existing.updated_at = datetime.now(UTC)
        await db.commit()
        return
    db.add(Thread(id=thread_id, user_id=user_id, title=title[:TITLE_MAX]))
    try:
        await db.commit()
    except IntegrityError as exc:
        # 外键挡下来的：token 验签通过，但它的 sub 在 users 表里查无此人——账号被删了、库被换了、
        # 或这枚 token 是开发态发证口给一个不存在的 user_id 签的。此时该让用户重新登录（401），
        # 而不是把一个数据库约束错误当 500 甩到脸上。
        await db.rollback()
        raise LookupError("凭证对应的用户不存在") from exc


async def assert_owner(db: AsyncSession, thread_id: str, user_id: str) -> None:
    """校验某会话确属此人，否则抛 PermissionError（路由转 403/404）。

    **未登记的 thread 视为无主、放行**：鉴权关闭时（demo 模式）没人认领会话，若一律拒绝就把整个
    demo 锁死了。开启鉴权后所有会话都会在起任务时被认领，这条兜底自然失效。
    """
    row = await db.get(Thread, thread_id)
    if row is not None and row.user_id != user_id:
        raise PermissionError("无权访问该会话")


async def delete_thread(db: AsyncSession, thread_id: str, user_id: str) -> None:
    """把一段会话从我的清单里删掉（侧栏的删除）。

    **只删归属记录，不删 ``output/<tid>/`` 里的对话正文**——与改造前「只删前端索引」的语义一致：
    入口没了，磁盘上的东西还在。真要做「彻底删除我的数据」，得连会话目录一起清，那是另一件事
    （涉及正在跑的任务、产物文件、事件流），不在本次范围内，诚实标注。
    """
    row = await db.get(Thread, thread_id)
    if row is None:
        return  # 已经没了：删除本就该幂等，不报错
    if row.user_id != user_id:
        raise PermissionError("无权访问该会话")
    await db.delete(row)
    await db.commit()


async def list_threads(db: AsyncSession, user_id: str, limit: int = 50) -> list[Thread]:
    """某用户的会话清单，最近聊过的在前——侧栏历史的后端真源（取代前端 localStorage）。"""
    stmt = (
        select(Thread)
        .where(Thread.user_id == user_id)
        .order_by(Thread.updated_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())
