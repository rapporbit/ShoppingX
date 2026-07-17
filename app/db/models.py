"""账户与会话归属的关系表（M16）——「数据没丢，只是没绑到人身上」的那一层。

**为什么要建库。** 此前会话数据其实一直是持久的（偏好在 Redis + AOF、每轮对话落 ``output/`` 卷），
但**没有任何东西记录「哪个 thread 属于谁」**：侧栏历史清单纯靠浏览器 localStorage，换个浏览器 /
清缓存，后端数据还在却再也找不回来；且谁猜到 thread_id 就能读别人的会话（``auth.py`` 点名的洞）。
所以缺的不是持久化，是**归属**——两张表：``users``（人）与 ``threads``（会话归谁）。

**为什么用关系库而不是塞进已有的 Redis。** 密码与归属关系是账户系统的**真源**：要唯一约束
（用户名不重复）、要外键（会话必属于某个真实用户）、要事务。Redis 的 AOF 是「尽力而为」的持久化，
拿它当唯一真源，一次异常退出就可能丢掉刚注册的账号——偏好丢一条无所谓，账号丢一条是事故。

**为什么 SQLite 而不是 Postgres。** 部署是**单机单 worker**（Dockerfile 里就写死单 worker，因为
任务表是进程内状态），那台 VPS 已经扛着 OpenSearch + Qdrant + Redis，再加一个 Postgres 容器纯属
拿内存换用不上的并发写能力。SQLite 零容器、零运维、库文件躺在持久卷上。代价说清楚：多进程并发写
会锁表——真要多副本时，把 DSN 换成 Postgres 即可，ORM 层一行不用改（这正是用 SQLAlchemy 而非手写
SQL 的理由）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    """一个注册用户。

    ``id`` 用随机 hex 而非自增整数：它会被当作 ``user_id`` 写进 JWT 的 sub、进偏好 Store 的 key、
    出现在日志里。自增 id 会泄漏「这站一共几个用户、我是第几个注册的」，且一旦别处误信前端传的
    user_id，猜一个相邻整数就能撞到真实账号——随机 id 让「猜别人的身份」这条路直接不通。

    ``password_hash`` 存 bcrypt 摘要，**明文密码任何时候都不落库、不进日志**。
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # unique 约束是「用户名不重名」的唯一可信保证：应用层「先查再插」在并发下必然漏（两个请求同时
    # 查到「没人叫这个名」再同时插入），只有数据库的唯一索引能挡住，冲突时插入直接失败。
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    threads: Mapped[list[Thread]] = relationship(back_populates="owner")


class Thread(Base):
    """一段会话的归属与摘要——「这个 thread 是谁的、叫什么、什么时候更新的」。

    **只存元信息，不存正文。** 对话正文、P_t、候选集仍留在 ``output/<thread_id>/`` 的文件里（它们
    本来就在持久卷上，搬进库只是平白多一次迁移）。这张表负责两件文件系统答不了的事：**按用户列出
    他的会话**（侧栏历史），以及**校验属主**（挡住拿别人 thread_id 读历史 / 下产物 / 连事件流）。

    ``title`` 取该会话首轮提问的前若干字——省一次 LLM 调用，且用户自己说过的话最认得出是哪段对话。
    """

    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # index：侧栏每次都按 user_id 查会话清单，是本表最热的查询路径。
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # onupdate：每次这段会话有新一轮对话就自动刷新，侧栏据此按「最近聊过」倒序排。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=func.now()
    )

    owner: Mapped[User] = relationship(back_populates="threads")


# ── 用户级持久数据（Mmem：从 JSON 文件 / Redis 搬进关系库）──────────────────────────────
#
# **为什么这三张表的 user_id 都不加外键**（与 threads 相反）：鉴权关闭时（demo 模式）用户是
# 一个不在 users 表里的假身份（demo-user），外键会让每一次写偏好直接炸。而这三类数据的强度要求
# 本就低于账户——「偏好丢一条无所谓，账号丢一条是事故」（见模块 docstring）。只建 index 保查询。


class Preference(Base):
    """一条长期偏好（跨会话的一贯取向）。

    ``dedup_key`` 是**派生**的去重身份（由 polarity/category/domain/slug 拼出，见
    :class:`app.memory.store.PreferenceEntry`），不由 LLM 手拼——但它要在库里做唯一约束，所以
    冗余存一列。``(user_id, dedup_key)`` 唯一：同一身份的偏好只有一条，重复提及走覆盖合并。

    **``blocking`` 是这次重构的核心字段**：只有它为 True 的条目才会在 item_picker 里**硬淘汰**
    商品。而它**只能由用户在偏好页面显式勾选**（source="user"）——LLM 学到的偏好一律只减分。
    让一个每轮都在猜的模型去决定「这件商品用户永远不该看到」，风险和收益完全不匹配：猜错了，
    用户搜不到东西还归因不了。杀伤力必须由用户授予。
    """

    __tablename__ = "preferences"
    __table_args__ = (UniqueConstraint("user_id", "dedup_key", name="uq_pref_user_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    dedup_key: Mapped[str] = mapped_column(String(200))

    polarity: Mapped[str] = mapped_column(String(16), default="like")  # like / dislike
    category: Mapped[str] = mapped_column(String(32), default="other")  # PrefCategory
    domain: Mapped[str] = mapped_column(String(32), default="other")  # PrefDomain
    slug: Mapped[str] = mapped_column(String(64), default="")
    content: Mapped[str] = mapped_column(String(500))
    # 可硬过滤 / 减分的原子词。JSON 列：SQLite 原生支持，且这里只做整存整取，不按元素查询。
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)

    source: Mapped[str] = mapped_column(String(16), default="agent")  # agent / user
    blocking: Mapped[bool] = mapped_column(Boolean, default=False)

    source_session: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # 上次被确认（重复提及）的时间。**不再参与任何打分**（半衰期衰减已删）——只供偏好页面显示
    # 「这条 3 个月没用过了」，把「淡出」从一个没人能解释的隐式指数函数，变成用户看得见、能自己
    # 决定删不删的显式提示。系统不该偷偷把用户的偏好打七折。
    last_confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class HistoryRecord(Base):
    """一条行为历史（「上次搜了什么」）——事实快照，与偏好正交。

    与偏好的根本不同：偏好是「一贯的取向」（要去重 + 覆盖合并），历史是「做过什么」，**既不去重
    也不合并**，每种 kind 只保留最近若干条 + TTL 过期。故没有 dedup_key、没有唯一约束。
    """

    __tablename__ = "history_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # purchase / search
    content: Mapped[str] = mapped_column(String(500))
    source_session: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Favorite(Base):
    """用户收藏（♡）的一件商品。**纯展示数据——任何地方都不注入 prompt、不影响 Agent。**

    这是刻意的边界：收藏一件商品并不能可靠地推出任何偏好（可能只是想再比比价），拿它去改
    Agent 行为是过度解读。用户真正的「不要这个」信号走偏好库，不走这里。

    存**商品快照**而非只存 item_id：收藏要长期留着，而候选登记表随会话清理——换个会话按 id
    早就捞不回商品了。``(user_id, item_id)`` 唯一 → 重复点 ♡ 天然幂等。
    """

    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "item_id", name="uq_fav_user_item"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    item_id: Mapped[str] = mapped_column(String(128))

    title: Mapped[str] = mapped_column(String(500))
    platform: Mapped[str] = mapped_column(String(32), default="")
    price_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    landed_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_url: Mapped[str] = mapped_column(String(1000), default="")
    url: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UsageLedger(Base):
    """一个用户在**一个计费周期（UTC 自然日）**内的 LLM 用量与成本账本 —— credit 制的真源。

    **与已有两层记账的分工。** ``app/agent/usage.py`` 是事后测量（一轮的 token 聚合，进日志 /
    Langfuse）；``app/agent/token_budget.py`` 是**单次任务**（一棵 fork 树）的成本闸，进程内的
    模块级 dict，任务一结束就 ``reset_tree`` 清掉。两者都答不了「这个人这个月一共烧了多少、还剩
    多少能用」——那要跨会话、跨进程重启地累计，只能落库。

    **为什么按 (user_id, period_key) 一行而不是流水表。** 配额判定是最热的读路径（每次发任务都查
    一次、前端余额条也查），流水表要 ``SUM()`` 全表扫；这里一次主键命中就拿到累计值。真要审计
    「哪一轮烧的」，``messages.tokens`` 里每轮的 input/output/cost 都在，已经是流水。

    **``period_key`` = ``"YYYY-MM-DD"``（UTC），跨日靠「换一行」自然重置**，不需要任何定时任务：
    新的一天第一次记账时 upsert 出一行新的、从 0 起算。老行留着就是历史账单（将来要画「近 30 天
    用量」直接查它）。用 UTC 而非本地时区：服务器时区一改，用户的额度就会凭空多出或蒸发一段。

    ``cost_usd`` 是配额的**判定口径**（cache_read 享折扣的三档计价，见 token_budget），token 数
    只作展示与排障。前端不展示美元，按固定汇率换算成 credit（见 :mod:`app.db.quota`）。
    """

    __tablename__ = "usage_ledger"
    __table_args__ = (UniqueConstraint("user_id", "period_key", name="uq_usage_user_period"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # 不加外键，理由同 Preference：鉴权关闭时 user_id 是不在 users 表里的假身份。
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    period_key: Mapped[str] = mapped_column(String(16))  # "2026-07-14"（UTC 日）

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    task_count: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=func.now()
    )


class Message(Base):
    """一条对话消息（``user`` 或 ``assistant``）——**会话正文的真源**，原先落 ``turns.json``。

    **为什么它必须进库，而同目录的 pt.json / candidates.json / history.json 不必。** ``threads``
    表已经把「这段会话归谁、叫什么」搬进库了，正文却还躺在 ``output/<thread_id>/`` 的文件里——
    库与文件系统的生命周期一旦不一致（换机器、卷没挂上、多副本各写各的盘），侧栏就会列出一段
    点进去空白的会话：**元信息说它存在，正文说它不存在**。这不是「丢了点日志」，是用户回来发现
    自己聊过的东西没了。而那三份是过程产物：pt.json 是带 TTL 的会话缓存（过期就该当空开局）、
    candidates.json 随会话清理（商品真源在 Qdrant，要长留的已在 :class:`Favorite` 存了快照）、
    history.json 是排障用的完整轨迹（项目刻意不挂 checkpointer，没有「从中途恢复」的语义要它）。
    **判据是「用户回来还指望看到它吗」**，不是「它是不是状态」。

    ``seq`` 而非只靠 ``created_at`` 定序：同一轮的 user / assistant 两条在同一次写入里产生，
    时间戳可能撞到同一微秒，排序就成了不确定的。序号由「该 thread 现有条数」续号，读回来
    ``ORDER BY seq`` 逐字复现当时的顺序。

    ``items`` / ``activity`` / ``tokens`` 只挂 assistant 轮，是**回看**的还原源（商品卡、思考过程
    折叠区、token 消耗）；续聊回喂只取 role/content，对它们透明、不增 token。JSON 列整存整取，
    不按元素查询。

    ``thread_id`` **不加外键**（与 :class:`Thread` 相反，理由同 :class:`Preference`）：鉴权关闭的
    demo 模式下压根不建 ``threads`` 行，外键会让每一轮对话落库直接炸。只建 index。
    """

    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("thread_id", "seq", name="uq_msg_thread_seq"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # index：读一段会话永远是「按 thread_id 捞全部、按 seq 排」，这是本表唯一的查询路径。
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)

    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    # Text 而非 String(n)：结论文案动辄上千字（一份带理由的商品清单），截断就是毁数据。
    content: Mapped[str] = mapped_column(Text, default="")

    items: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    activity: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    tokens: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 本轮参考图的**文件名**（不是内容），唯一挂在 user 轮上——它属于用户说的话的一部分。
    # 存名不存图：图本体在 uploaded/<thread_id>/ 下，回看时前端拿名去 GET /api/uploads 取。
    # 图动辄几 MB，塞进 JSON 列会让每次读会话都把它们全拖出来，而回看只需要一个 <img src>。
    images: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ConfigOverride(Base):
    """后台管理页面改过的运行时参数（见 :mod:`app.config.registry`）。

    **只存「被改过的」，不存全量快照**：注册表里 30 多个参数绝大多数常年是默认值，把它们全落一遍
    会让「默认值改了」这件事悄悄失效——库里那份旧的固化值会一直盖住新代码的默认值。这里存的是
    「相对基线的差集」，恢复默认 = 删行，之后该参数就重新跟随 ``.env`` / 代码默认值走。

    值一律存字符串：它的归宿是 ``os.environ``（覆盖层经 env 中转，见 overrides.py 模块 docstring），
    类型信息在注册表里，不在库里——库里存 int/float 反而要为每种类型开一列。

    表很小（行数 ≤ 参数个数）且只在启动与改参时读写，不设索引，主键即查询路径。
    """

    __tablename__ = "config_overrides"

    # 主键即 env 变量名（如 PICK_DISPLAY_CAP）。同一参数只可能有一条覆盖，天然幂等 upsert。
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(200))
    # 留痕：谁在什么时候改的。调参调出问题时，这是唯一能回答「这值哪来的」的地方。
    updated_by: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
