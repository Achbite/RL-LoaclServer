#!/bin/bash
# ============================================================================
# RL-Learner 运行脚本（Docker 容器内执行）
# 用法：cd /workspace/RL-Learner && ./run.sh [--model_path <path>]
# 流程：生成 Proto Python 绑定 → 启动 Learner gRPC 服务
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/learner_config.yaml"

echo "============================================"
echo "  迷宫训练框架 - RL-Learner 启动"
echo "============================================"
echo ""

# ---- 1. 生成 Proto Python 绑定 ----
echo "[Run] 生成 Proto Python 绑定..."
bash "${SCRIPT_DIR}/proto/gen_proto.sh"
echo ""

# ---- 2. 检查配置文件 ----
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[Run] 配置文件不存在: ${CONFIG_PATH}"
    exit 1
fi

# ---- 3. 启动训练指标面板服务（后台）----
METRICS_DIR="${SCRIPT_DIR}/logs/metrics"
DASHBOARD_PORT=9005
if ! pgrep -f "metrics_server.py" > /dev/null 2>&1; then
    echo "[Run] 启动训练指标面板服务..."
    mkdir -p "${METRICS_DIR}"
    nohup python3 "${SCRIPT_DIR}/tools/dashboard/metrics_server.py" \
        --dir "${METRICS_DIR}" \
        --port ${DASHBOARD_PORT} \
        > "${SCRIPT_DIR}/logs/metrics_server.log" 2>&1 &
    echo "[Run] 指标面板: http://localhost:${DASHBOARD_PORT}"
else
    echo "[Run] 训练指标面板服务已在运行"
fi
echo ""

# ---- 4. 启动 Learner 服务（透传所有命令行参数）----
echo "[Run] 启动 Learner gRPC 服务..."
echo "[Run] 配置文件: ${CONFIG_PATH}"
echo ""

cd "${SCRIPT_DIR}"
python3 -m main.train --config "${CONFIG_PATH}" "$@"
