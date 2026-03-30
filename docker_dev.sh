#!/bin/bash
# ============================================================================
# 迷宫训练框架 - 一键 Docker 开发环境入口
# 用法：bash docker_dev.sh <服务名> [选项]
# 服务名：
#   client       进入 Client 容器（交互式 bash）
#   aiserver     启动 AIServer 后台服务 + 进入容器
# 选项：
#   (无参数)     检测镜像 → 按需构建 → 启动/进入容器
#   --build      强制重新构建镜像
#   --restart    重启已有容器（清理缓存，解决残留服务问题）
#   --check      仅检测，不启动/进入容器
# 运行环境：WSL2 Ubuntu
# ============================================================================

set -e

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

# --- 项目路径 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- 服务名映射 ---
declare -A SERVICE_MAP=(
    ["client"]="maze-client"
    ["aiserver"]="maze-aiserver"
)

declare -A DOCKERFILE_MAP=(
    ["client"]="docker/Dockerfile.client"
    ["aiserver"]="docker/Dockerfile.aiserver"
)

# --- 使用说明 ---
usage() {
    echo "用法: bash docker_dev.sh <服务名> [选项]"
    echo ""
    echo "服务名:"
    echo "  client       进入 Client 容器（交互式 bash）"
    echo "  aiserver     启动 AIServer 后台服务 + 进入容器"
    echo ""
    echo "选项:"
    echo "  (无)         检测镜像 → 按需构建 → 启动/进入容器"
    echo "  --build      强制重新构建镜像"
    echo "  --restart    重启已有容器并进入（清理缓存）"
    echo "  --check      仅检测，不进入容器"
    echo ""
    echo "容器内操作（进入容器后已在对应项目目录）:"
    echo "  ./build.sh    编译项目"
    echo "  ./run.sh      运行项目"
    exit 1
}

# --- 参数解析 ---
if [ $# -lt 1 ]; then
    usage
fi

TARGET="$1"
shift

# 验证服务名
if [ -z "${SERVICE_MAP[$TARGET]}" ]; then
    fail "未知服务名: $TARGET"
    usage
fi

SERVICE_NAME="${SERVICE_MAP[$TARGET]}"
DOCKERFILE="${SCRIPT_DIR}/${DOCKERFILE_MAP[$TARGET]}"
HASH_FILE="${SCRIPT_DIR}/.docker_build_hash_${TARGET}"

# 解析选项
FORCE_BUILD=false
CHECK_ONLY=false
RESTART_MODE=false

for arg in "$@"; do
    case "$arg" in
        --build)    FORCE_BUILD=true ;;
        --check)    CHECK_ONLY=true ;;
        --restart)  RESTART_MODE=true ;;
        *)          echo "未知参数: $arg"; usage ;;
    esac
done

echo ""
echo "============================================"
echo "  迷宫训练框架 - Docker 开发环境"
echo "  服务: ${TARGET} (${SERVICE_NAME})"
echo "============================================"
echo ""

# ============================================================================
# 1. 前置检查：Docker 是否可用
# ============================================================================

if ! command -v docker &> /dev/null; then
    fail "Docker 未安装，请先运行 bash check_env.sh --fix"
    exit 1
fi

if ! docker info &> /dev/null; then
    fail "Docker daemon 未运行，请先启动 Docker"
    info "尝试: sudo service docker start"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    fail "Docker Compose 未安装"
    exit 1
fi

ok "Docker 环境就绪"

# ============================================================================
# 2. 重启模式：停止旧容器 → 清理 → 重新启动并进入
# ============================================================================

if $RESTART_MODE; then
    echo ""
    info "--- 重启模式：清理并重启容器 ---"

    cd "${SCRIPT_DIR}"

    # 停止并移除该服务的容器（包括孤儿容器）
    info "停止并清理容器..."
    docker compose stop "${SERVICE_NAME}" 2>/dev/null || true
    docker compose rm -f "${SERVICE_NAME}" 2>/dev/null || true

    # 清理该项目的孤儿容器
    docker compose down --remove-orphans 2>/dev/null || true

    ok "容器已清理"

    # 重新启动依赖服务（如 aiserver）
    info "启动依赖服务..."
    docker compose up -d --no-recreate 2>/dev/null || true

    # 进入交互式 bash
    echo ""
    info "--- 进入容器交互式 bash ---"
    info "退出容器: 输入 exit 或按 Ctrl+D"
    echo ""

    docker compose run --rm "${SERVICE_NAME}" bash
    exit 0
fi

# ============================================================================
# 3. 检测镜像是否存在
# ============================================================================

# 从 docker-compose.yml 推断镜像名（项目名-服务名）
PROJECT_DIR_NAME=$(basename "${SCRIPT_DIR}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
IMAGE_NAME="${PROJECT_DIR_NAME}-${SERVICE_NAME}"

IMAGE_EXISTS=false
if docker image inspect "${IMAGE_NAME}" &> /dev/null; then
    IMAGE_EXISTS=true
    ok "镜像 ${IMAGE_NAME} 已存在"
else
    warn "镜像 ${IMAGE_NAME} 不存在"
fi

# ============================================================================
# 4. 检测 Dockerfile 是否有变化
# ============================================================================

DOCKERFILE_CHANGED=false

if [ -f "${DOCKERFILE}" ]; then
    CURRENT_HASH=$(md5sum "${DOCKERFILE}" | awk '{print $1}')

    if [ -f "${HASH_FILE}" ]; then
        SAVED_HASH=$(cat "${HASH_FILE}")
        if [ "${CURRENT_HASH}" != "${SAVED_HASH}" ]; then
            DOCKERFILE_CHANGED=true
            warn "Dockerfile 已变更（上次构建后有修改）"
        else
            ok "Dockerfile 未变更"
        fi
    else
        if $IMAGE_EXISTS; then
            info "首次记录 Dockerfile hash"
        else
            DOCKERFILE_CHANGED=true
        fi
    fi
else
    fail "Dockerfile 不存在: ${DOCKERFILE}"
    exit 1
fi

# ============================================================================
# 5. 决定是否需要构建
# ============================================================================

NEED_BUILD=false

if $FORCE_BUILD; then
    NEED_BUILD=true
    info "强制构建模式"
elif ! $IMAGE_EXISTS; then
    NEED_BUILD=true
    info "镜像不存在，需要构建"
elif $DOCKERFILE_CHANGED; then
    NEED_BUILD=true
    info "Dockerfile 已变更，需要重新构建"
fi

# ============================================================================
# 6. 执行构建（如果需要）
# ============================================================================

if $NEED_BUILD; then
    echo ""
    info "--- 开始构建镜像 ---"
    echo ""

    cd "${SCRIPT_DIR}"
    docker compose build "${SERVICE_NAME}"

    # 构建成功后保存 Dockerfile hash
    CURRENT_HASH=$(md5sum "${DOCKERFILE}" | awk '{print $1}')
    echo "${CURRENT_HASH}" > "${HASH_FILE}"

    echo ""
    ok "镜像构建完成"
else
    ok "镜像已是最新，无需构建"
fi

# ============================================================================
# 7. 仅检测模式：输出状态后退出
# ============================================================================

if $CHECK_ONLY; then
    echo ""
    echo "============================================"
    ok "检测完成（--check 模式，不进入容器）"
    echo ""
    info "镜像状态: $(if $IMAGE_EXISTS || $NEED_BUILD; then echo '就绪'; else echo '未构建'; fi)"
    info "进入容器: bash docker_dev.sh ${TARGET}"
    echo ""
    exit 0
fi

# ============================================================================
# 8. 启动/进入容器（根据服务类型区分行为）
# ============================================================================

echo ""
cd "${SCRIPT_DIR}"

case "$TARGET" in
    aiserver)
        # AIServer：后台启动服务 + 进入交互式 bash
        info "--- 启动 AIServer 后台服务 ---"
        docker compose up -d maze-aiserver
        echo ""
        ok "AIServer 已后台启动（容器名: maze-aiserver）"
        echo ""
        info "--- 进入 AIServer 容器交互式 bash ---"
        info "退出容器: 输入 exit 或按 Ctrl+D（服务仍在后台运行）"
        info "停止服务: docker compose stop maze-aiserver"
        echo ""
        info "编译项目: ./build.sh"
        info "运行项目: ./run.sh"
        echo ""
        docker compose exec maze-aiserver bash
        ;;
    client)
        # Client：交互式 bash（临时容器）
        info "--- 进入 Client 容器交互式 bash ---"
        info "退出容器: 输入 exit 或按 Ctrl+D"
        echo ""
        info "编译项目: ./build.sh"
        info "运行项目: ./run.sh"
        echo ""
        docker compose run --rm maze-client bash
        ;;
esac
