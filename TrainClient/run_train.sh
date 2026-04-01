#!/bin/bash
# ============================================================================
# TrainClient 并行训练运行脚本（Docker 容器内执行）
# 用法：cd /workspace/TrainClient && ./run_train.sh
# 前提：已通过 ./build.sh 完成编译
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/maze_train"
CONFIG_PATH="${SCRIPT_DIR}/configs/client_config.yaml"

# ---- 检查可执行文件是否存在 ----
if [ ! -f "${EXECUTABLE}" ]; then
    echo "[Run] 可执行文件不存在: ${EXECUTABLE}"
    echo "[Run] 请先运行: ./build.sh"
    exit 1
fi

echo "============================================"
echo "  迷宫训练框架 - TrainClient 并行训练"
echo "============================================"
echo ""

# ---- 切换到 TrainClient 目录（配置文件使用相对路径）----
cd "${SCRIPT_DIR}"

# ---- 自动启动可视化 HTTP 服务（后台运行，不占用终端）----
VIZ_SERVER="${SCRIPT_DIR}/tools/viz_player/maze_viz_server.py"
VIZ_LOG="${SCRIPT_DIR}/log/viz_server.log"

if [ -f "${VIZ_SERVER}" ] && command -v python3 &>/dev/null; then
    if ! pgrep -f "maze_viz_server.py" > /dev/null 2>&1; then
        mkdir -p "${SCRIPT_DIR}/log"
        nohup python3 "${VIZ_SERVER}" --dir "${SCRIPT_DIR}/log/viz" --port 9004 \
            > "${VIZ_LOG}" 2>&1 &
        VIZ_PID=$!
        echo "[Run] 可视化服务已启动（PID: ${VIZ_PID}，端口 9004）"
        echo "[Run] 浏览器访问: http://localhost:9004"
        echo "[Run] 服务日志: ${VIZ_LOG}"
    else
        echo "[Run] 可视化服务已在运行中"
    fi
else
    echo "[Run] 跳过可视化服务（未找到 ${VIZ_SERVER} 或 python3）"
fi
echo ""

# ---- 启动 ----
echo "[Run] 启动并行训练..."
echo "[Run] 配置文件: ${CONFIG_PATH}"
echo ""

"${EXECUTABLE}" "${CONFIG_PATH}"
