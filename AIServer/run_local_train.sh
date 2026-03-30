#!/bin/bash
# ============================================================================
# AIServer 训练模式专用启动脚本（Docker 容器内执行）
# 用法：cd /workspace/AIServer && ./run_local_train.sh
# 说明：强制以 run_mode=1（训练模式）启动，通过命令行参数覆盖配置文件
# 前提：已通过 ./build.sh 完成编译
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/maze_aiserver"
CONFIG_PATH="${SCRIPT_DIR}/configs/server_config.yaml"

# ---- 检查可执行文件是否存在 ----
if [ ! -f "${EXECUTABLE}" ]; then
    echo "[Train] 可执行文件不存在: ${EXECUTABLE}"
    echo "[Train] 请先运行: ./build.sh"
    exit 1
fi

echo "============================================"
echo "  迷宫训练框架 - AIServer 训练模式"
echo "============================================"
echo ""

# ---- 切换到 AIServer 目录 ----
cd "${SCRIPT_DIR}"

# ---- 启动（传递 --train 参数强制训练模式）----
echo "[Train] 启动 AIServer（训练模式, run_mode=1）..."
echo "[Train] 配置文件: ${CONFIG_PATH}"
echo ""

"${EXECUTABLE}" "${CONFIG_PATH}" --train
