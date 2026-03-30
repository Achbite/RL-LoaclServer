#!/bin/bash
# ============================================================================
# AIServer 运行脚本（Docker 容器内执行）
# 用法：cd /workspace/AIServer && ./run.sh
# 前提：已通过 ./build.sh 完成编译
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/maze_aiserver"
CONFIG_PATH="${SCRIPT_DIR}/configs/server_config.yaml"

# ---- 检查可执行文件是否存在 ----
if [ ! -f "${EXECUTABLE}" ]; then
    echo "[Run] 可执行文件不存在: ${EXECUTABLE}"
    echo "[Run] 请先运行: ./build.sh"
    exit 1
fi

echo "============================================"
echo "  迷宫训练框架 - AIServer 启动"
echo "============================================"
echo ""

# ---- 切换到 AIServer 目录（配置文件使用相对路径）----
cd "${SCRIPT_DIR}"

# ---- 启动 ----
echo "[Run] 启动 AIServer..."
echo "[Run] 配置文件: ${CONFIG_PATH}"
echo ""

"${EXECUTABLE}" "${CONFIG_PATH}"
