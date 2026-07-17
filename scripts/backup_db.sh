#!/usr/bin/env bash
# globex 生产库备份（在 VPS 上跑，由 cron 每天调用）。
#
# **只备份一个东西：那个 SQLite 文件。** 全站不可重建的数据恰好都在里面——users（注册的人）、
# threads（会话归谁）、preferences（长期偏好）、history_records、favorites、usage_ledger（credit
# 账本）、messages（对话正文）。丢了没有任何别处能重建出来。
#
# 其余的都**故意不备**，因为都能重来：Qdrant / OpenSearch 索引可由 data/ + scripts/build_*.py 重建
# （代价是一笔 embedding 费用和若干小时，不是「数据没了」）；Redis 里只有事件回放和语义缓存，丢了
# 用户最多少看一段回放。把备份面收窄到这一个文件，是为了让恢复这件事在真出事那天足够简单。
#
# **不能直接 cp 那个 .db 文件**：数据库可能正被写到一半，拷出来的是个撕裂的快照——恢复时才发现
# 它坏了，而那时已经晚了。走 SQLite 的在线备份 API（``Connection.backup()``），它在写事务之间
# 取一致性快照，容器照常服务不用停。
#
# 用法：
#   ./scripts/backup_db.sh              # 备份到 ~/backups/globex/
#   BACKUP_DIR=/mnt/x ./scripts/backup_db.sh
#   RETENTION_DAYS=30 ./scripts/backup_db.sh

set -euo pipefail

CONTAINER="${CONTAINER:-globex-backend}"
DB_PATH="${DB_PATH:-/app/var/globex.db}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups/globex}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT="$BACKUP_DIR/globex-$STAMP.db.gz"

log() { echo "[backup $(date -u +%H:%M:%S)] $*"; }
die() { echo "[backup ERROR] $*" >&2; exit 1; }

mkdir -p "$BACKUP_DIR"

docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true \
    || die "容器 $CONTAINER 没在跑——备份没做，别以为做了"

# ① 容器内取一致性快照。integrity_check 就地做掉：备份文件本身是坏的却一路存下来、直到恢复那天
#    才发现，是备份这件事最典型的失败方式。这里当场验，不通过就不落盘。
# `docker exec` **必须带 -i**：不带的话容器里根本没有 stdin，heredoc 的脚本送不进去，`python -`
# 读到一个空程序、干干净净地退出 0——快照没生成，脚本却一路绿灯往下走。第一次跑就踩了这个。
log "从 $CONTAINER:$DB_PATH 取一致性快照..."
docker exec -i "$CONTAINER" python - "$DB_PATH" <<'PY' || die "快照失败"
import sqlite3, sys
src_path = sys.argv[1]
src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
dst = sqlite3.connect("/tmp/globex-backup.db")
with dst:
    src.backup(dst)          # 在线备份 API：写事务之间取快照，不用停服务
ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
users = dst.execute("SELECT count(*) FROM users").fetchone()[0]
dst.close(); src.close()
if ok != "ok":
    sys.exit(f"integrity_check 不通过：{ok}")
print(f"快照 OK，users={users}")
PY

# ② 拷出容器并压缩
docker cp "$CONTAINER:/tmp/globex-backup.db" "$BACKUP_DIR/.staging.db" >/dev/null \
    || die "docker cp 失败"
docker exec "$CONTAINER" rm -f /tmp/globex-backup.db || true
gzip -c "$BACKUP_DIR/.staging.db" > "$OUT"
rm -f "$BACKUP_DIR/.staging.db"

# ③ 落盘后再验一次压缩包本身（gzip 截断过的文件在这里就会暴露，而不是在恢复那天）
gzip -t "$OUT" || die "备份文件损坏：$OUT"
SIZE="$(du -h "$OUT" | cut -f1)"
log "已备份 → $OUT ($SIZE)"

# ④ 过期清理。**先确认新备份是好的再删旧的**——顺序反过来的话，某天备份失败又刚好把旧的删了，
#    就会在你毫无察觉时进入「一份备份都没有」的状态。
find "$BACKUP_DIR" -name 'globex-*.db.gz' -type f -mtime "+$RETENTION_DAYS" -print -delete \
    | while read -r f; do log "清理过期备份 $(basename "$f")"; done

COUNT="$(find "$BACKUP_DIR" -name 'globex-*.db.gz' -type f | wc -l | tr -d ' ')"
log "完成。现存 $COUNT 份备份（保留 $RETENTION_DAYS 天）"
