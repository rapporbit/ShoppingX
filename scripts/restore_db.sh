#!/usr/bin/env bash
# 从备份恢复 globex 的 SQLite 库（配合 scripts/backup_db.sh）。
#
# **写这个脚本的理由是：没恢复过的备份不算备份。** 备份脚本每天绿灯跑着，不代表那些 .gz 真能变回
# 一个能用的库——真出事那天才第一次尝试恢复，是最坏的时机。建议现在就拿它演练一次（用 --dry-run
# 恢复到临时文件、看看用户数对不对），别等到需要它的时候。
#
# 用法：
#   ./scripts/restore_db.sh --dry-run ~/backups/globex/globex-20260714-030000.db.gz  # 演练：只解压 + 验，不动线上
#   ./scripts/restore_db.sh           ~/backups/globex/globex-20260714-030000.db.gz  # 真恢复（会停容器）

set -euo pipefail

CONTAINER="${CONTAINER:-globex-backend}"
DB_PATH="${DB_PATH:-/app/var/globex.db}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=true; shift; }
ARCHIVE="${1:-}"

log() { echo "[restore] $*"; }
die() { echo "[restore ERROR] $*" >&2; exit 1; }

[[ -n "$ARCHIVE" && -f "$ARCHIVE" ]] || die "用法：$0 [--dry-run] <备份文件.db.gz>"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

log "解压 $ARCHIVE ..."
gzip -dc "$ARCHIVE" > "$TMP/restored.db" || die "解压失败——这份备份是坏的"

# 恢复前先验：拿一个坏库去覆盖线上那个好库，是能把「一次事故」升级成「两次事故」的操作。
log "校验..."
python3 - "$TMP/restored.db" <<'PY' || die "备份文件校验不通过，拒绝用它覆盖线上库"
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
ok = db.execute("PRAGMA integrity_check").fetchone()[0]
if ok != "ok":
    sys.exit(f"integrity_check: {ok}")
for t in ("users", "threads", "preferences", "usage_ledger"):
    n = db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    print(f"  {t:16} {n:6d} 行")
PY

if $DRY_RUN; then
    log "演练完毕（--dry-run）：这份备份是好的，能恢复出上面这些数据。线上库一根毫毛都没动。"
    exit 0
fi

read -r -p "⚠️  将用它覆盖 $CONTAINER:$DB_PATH，线上现有数据会被替换。输入 yes 继续：" ans
[[ "$ans" == "yes" ]] || die "已取消"

# 停容器再覆盖：库正被进程打开着就替换文件，会留下 WAL / journal 与主库不匹配的残局。
log "停止 $CONTAINER ..."
docker stop "$CONTAINER" >/dev/null

# 把当前这份先留一手——万一恢复的是错的那份备份，还能退回来。
docker cp "$CONTAINER:$DB_PATH" "$TMP/pre-restore.db" 2>/dev/null && {
    KEEP="$HOME/backups/globex/pre-restore-$(date -u +%Y%m%d-%H%M%S).db"
    mkdir -p "$(dirname "$KEEP")"; cp "$TMP/pre-restore.db" "$KEEP"
    log "被覆盖的旧库已留档 → $KEEP"
}

docker cp "$TMP/restored.db" "$CONTAINER:$DB_PATH" || die "写入失败（容器已停，需手动 docker start）"

# SQLite 的 -wal / -shm 是**旧库的尾巴**：留着就会被当成刚换上这个主库的未提交事务重放，轻则脏数据
# 重则打不开。必须删。不能用 docker exec —— 容器此刻是停的，exec 会直接失败（而失败被 || true 吞掉
# 的话，wal 就悄悄留下了，正是那种「脚本全绿、数据是坏的」的结局）。改用一次性容器挂同一批卷来删。
docker run --rm --volumes-from "$CONTAINER" alpine:3 \
    sh -c "rm -f ${DB_PATH}-wal ${DB_PATH}-shm" \
    || die "WAL 清理失败——**别启动容器**，先手动删掉 ${DB_PATH}-wal / -shm 再启"

log "启动 $CONTAINER ..."
docker start "$CONTAINER" >/dev/null
log "恢复完成。去站点上登录一个已知账号确认一下，别只信这条日志。"
