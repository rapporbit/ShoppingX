#!/usr/bin/env bash
# ShoppingX 一键启动脚本
# 用法：
#   ./start.sh          # 启动全部（Docker + 后端 + 前端）
#   ./start.sh backend  # 只启后端（含 Docker）
#   ./start.sh stop     # 停止全部

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── 颜色 ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

# ── 前置检查 ──────────────────────────────────────────────
preflight() {
    local missing=()
    command -v docker  >/dev/null 2>&1 || missing+=(docker)
    command -v uv      >/dev/null 2>&1 || missing+=(uv)
    command -v node    >/dev/null 2>&1 || missing+=(node)
    command -v npm     >/dev/null 2>&1 || missing+=(npm)
    if (( ${#missing[@]} > 0 )); then
        err "缺少依赖：${missing[*]}"
        exit 1
    fi

    if [[ ! -f .env ]]; then
        err "缺少 .env 文件，请参考 .env.example 创建"
        exit 1
    fi
}

# ── Docker 基础设施 ───────────────────────────────────────
start_docker() {
    info "启动 Docker 容器（Qdrant / OpenSearch / Redis）..."
    docker compose -f docker/docker-compose.yml up -d

    info "等待服务就绪..."
    local retries=0 max_retries=30

    # Qdrant
    while ! curl -sf http://localhost:6333/readyz >/dev/null 2>&1; do
        ((retries++))
        if (( retries >= max_retries )); then
            err "Qdrant 未能在 ${max_retries}s 内就绪"; exit 1
        fi
        sleep 1
    done
    ok "Qdrant        :6333"

    # OpenSearch
    retries=0
    while ! curl -sf http://localhost:9200/_cluster/health >/dev/null 2>&1; do
        ((retries++))
        if (( retries >= max_retries )); then
            warn "OpenSearch 未就绪，category_insight 将退化到本地回退"
            break
        fi
        sleep 1
    done
    if (( retries < max_retries )); then
        ok "OpenSearch     :9200"
    fi

    # Redis
    retries=0
    while ! docker exec shoppingx-redis redis-cli ping 2>/dev/null | grep -q PONG; do
        ((retries++))
        if (( retries >= max_retries )); then
            warn "Redis 未就绪，长期记忆将退化到本地 JSON 文件"
            break
        fi
        sleep 1
    done
    if (( retries < max_retries )); then
        ok "Redis          :6379"
    fi
}

stop_docker() {
    info "停止 Docker 容器..."
    docker compose -f docker/docker-compose.yml down
    ok "Docker 容器已停止"
}

# ── Python 依赖 ───────────────────────────────────────────
sync_python() {
    info "同步 Python 依赖..."
    uv sync --quiet
    ok "Python 依赖就绪"
}

# ── 数据索引检查 ──────────────────────────────────────────
check_data() {
    # 检查 Qdrant 是否有商品索引（collection 存在且非空）
    local has_items=false
    if curl -sf http://localhost:6333/collections 2>/dev/null | grep -q '"name"'; then
        has_items=true
    fi

    if [[ "$has_items" == false ]]; then
        warn "Qdrant 中未检测到商品索引"
        echo -e "    首次使用请运行：${CYAN}uv run python scripts/build_item_index.py${NC}"
    fi

    # 检查 OpenSearch 品类知识库
    if curl -sf http://localhost:9200/_cat/indices 2>/dev/null | grep -q 'category'; then
        ok "品类知识库索引存在"
    else
        warn "OpenSearch 中未检测到品类知识库索引"
        echo -e "    首次使用请运行：${CYAN}uv run python scripts/build_category_kb.py${NC}"
    fi
}

# ── 后端 ──────────────────────────────────────────────────
start_backend() {
    info "启动后端 (FastAPI :8000)..."
    uv run uvicorn app.api.server:app --reload --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$ROOT_DIR/.pid_backend"

    # 等待后端响应
    local retries=0
    while ! curl -sf http://127.0.0.1:8000/docs >/dev/null 2>&1; do
        ((retries++))
        if (( retries >= 20 )); then
            err "后端未能在 20s 内启动"
            kill "$BACKEND_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    ok "后端已启动     http://127.0.0.1:8000"
}

# ── 前端 ──────────────────────────────────────────────────
start_frontend() {
    info "启动前端 (Vite :5173)..."
    cd "$ROOT_DIR/frontend"

    if [[ ! -d node_modules ]]; then
        info "安装前端依赖..."
        npm install --silent
    fi

    npm run dev &
    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$ROOT_DIR/.pid_frontend"
    cd "$ROOT_DIR"

    sleep 3
    ok "前端已启动     http://localhost:5173"
}

# ── 停止全部 ──────────────────────────────────────────────
stop_all() {
    info "停止所有服务..."

    for pidfile in .pid_backend .pid_frontend; do
        if [[ -f "$ROOT_DIR/$pidfile" ]]; then
            local pid
            pid=$(cat "$ROOT_DIR/$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                ok "已停止 PID $pid ($pidfile)"
            fi
            rm -f "$ROOT_DIR/$pidfile"
        fi
    done

    # 也杀掉可能残留的 uvicorn / vite 进程
    pkill -f "uvicorn app.api.server:app" 2>/dev/null || true
    pkill -f "vite.*5173" 2>/dev/null || true

    stop_docker
    ok "全部服务已停止"
}

# ── 优雅退出 ──────────────────────────────────────────────
cleanup() {
    echo ""
    info "收到中断信号，正在停止..."
    kill "$BACKEND_PID"  2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    rm -f "$ROOT_DIR/.pid_backend" "$ROOT_DIR/.pid_frontend"
    exit 0
}

# ── 主流程 ────────────────────────────────────────────────
main() {
    local mode="${1:-all}"

    case "$mode" in
        stop)
            stop_all
            exit 0
            ;;
        backend)
            preflight
            sync_python
            start_docker
            check_data
            start_backend
            echo ""
            ok "后端就绪。按 Ctrl+C 停止。"
            trap cleanup INT TERM
            wait "$BACKEND_PID"
            ;;
        all)
            preflight
            sync_python
            start_docker
            check_data
            start_backend
            start_frontend
            echo ""
            echo -e "${GREEN}════════════════════════════════════════${NC}"
            echo -e "${GREEN}  ShoppingX 全部就绪${NC}"
            echo -e "  后端  ${CYAN}http://127.0.0.1:8000${NC}"
            echo -e "  前端  ${CYAN}http://localhost:5173${NC}"
            echo -e "  API 文档  ${CYAN}http://127.0.0.1:8000/docs${NC}"
            echo -e "${GREEN}════════════════════════════════════════${NC}"
            echo -e "  按 ${YELLOW}Ctrl+C${NC} 停止前后端（Docker 容器保持运行）"
            echo -e "  运行 ${YELLOW}./start.sh stop${NC} 停止全部（含 Docker）"
            echo ""
            trap cleanup INT TERM
            wait "$BACKEND_PID" "$FRONTEND_PID"
            ;;
        *)
            echo "用法：./start.sh [all|backend|stop]"
            exit 1
            ;;
    esac
}

main "$@"
