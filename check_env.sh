#!/bin/bash
# ============================================================================
# 迷宫训练框架 - Docker 环境检测与修复脚本
# 用法：bash check_env.sh [--fix]
# 运行环境：WSL2 Ubuntu
# ============================================================================

set -e

# --- 参数解析 ---
FIX_MODE=false
if [ "$1" = "--fix" ] || [ "$1" = "-fix" ]; then
    FIX_MODE=true
fi

# --- 颜色输出 ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[  OK  ]${NC} $1"; }
fail() { echo -e "${RED}[ FAIL ]${NC} $1"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $1"; }
info() { echo -e "${CYAN}[ INFO ]${NC} $1"; }

# --- 检测结果收集 ---
ALL_PASSED=true

echo ""
echo "============================================"
echo "  迷宫训练框架 - Docker 环境检测"
echo "============================================"
echo ""

# ============================================================================
# 1. 系统环境
# ============================================================================
info "--- 系统环境 ---"

# 检测是否在 WSL 中
if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "WSL2 环境检测通过"
else
    info "非 WSL 环境（原生 Linux）"
fi

# 检测 OS 版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
    ok "操作系统: $PRETTY_NAME"
else
    warn "无法检测操作系统版本"
fi

echo ""

# ============================================================================
# 2. Docker 环境
# ============================================================================
info "--- Docker 环境 ---"

# 检测 Docker Engine
if command -v docker &> /dev/null; then
    docker_ver=$(docker --version 2>&1)
    ok "Docker Engine : $docker_ver"
else
    fail "Docker Engine : 未安装"
    ALL_PASSED=false

    if $FIX_MODE; then
        info "正在安装 Docker Engine CE..."

        # 移除可能冲突的旧包
        for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
            sudo apt-get remove -y "$pkg" 2>/dev/null || true
        done

        # 安装依赖
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg

        # 添加 Docker 官方 GPG key
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg

        # 添加 Docker 仓库
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
          sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        # 安装 Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

        ok "Docker Engine CE 安装完成"
    fi
fi

# 检测 docker compose
if docker compose version &> /dev/null; then
    compose_ver=$(docker compose version 2>&1)
    ok "Docker Compose : $compose_ver"
else
    fail "Docker Compose : 未安装"
    ALL_PASSED=false
fi

# 检测 Docker daemon 是否运行
if docker info &> /dev/null; then
    ok "Docker daemon : 运行中"
else
    warn "Docker daemon : 未运行"

    if $FIX_MODE; then
        # WSL 中 init 可能不是 systemd，需要手动启动
        INIT_PROC=$(ps -p 1 -o comm= 2>/dev/null || echo "unknown")
        if [ "$INIT_PROC" = "systemd" ]; then
            info "使用 systemd 启动 Docker..."
            sudo systemctl start docker
            sudo systemctl enable docker
        else
            info "非 systemd 环境，手动启动 dockerd..."
            sudo dockerd &> /tmp/dockerd.log &
            sleep 3

            if docker info &> /dev/null; then
                ok "Docker daemon 已启动"
            else
                fail "Docker daemon 启动失败，请检查 /tmp/dockerd.log"
            fi
        fi
    fi
fi

# 检测当前用户是否在 docker 组
if groups | grep -q docker; then
    ok "用户 docker 组 : 已加入（可免 sudo 使用 docker）"
else
    warn "用户未加入 docker 组（需要 sudo 运行 docker）"

    if $FIX_MODE; then
        info "将当前用户加入 docker 组..."
        sudo usermod -aG docker "$USER"
        ok "已加入 docker 组（需要重新登录 WSL 生效）"
    fi
fi

echo ""

# ============================================================================
# 结果汇总
# ============================================================================
echo "============================================"

if $ALL_PASSED; then
    echo ""
    ok "Docker 环境检测通过！可以使用 docker compose 构建和运行项目。"
    echo ""
    info "快速开始："
    echo -e "  ${CYAN}cd /mnt/e/RL-LoaclServer${NC}"
    echo -e "  ${CYAN}docker compose build maze-client${NC}"
    echo -e "  ${CYAN}docker compose run maze-client${NC}"
    echo ""
else
    echo ""
    fail "Docker 环境未就绪。"
    echo ""
    info "运行以下命令自动修复："
    echo -e "  ${CYAN}bash check_env.sh --fix${NC}"
    echo ""
fi
