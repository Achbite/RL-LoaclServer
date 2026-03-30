#!/bin/bash
# ============================================================================
# RL-Learner 运行脚本（Docker 容器内执行）
# 用法：cd /workspace/RL-Learner && ./run.sh
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

# ---- 3. 启动 Learner 服务 ----
echo "[Run] 启动 Learner gRPC 服务..."
echo "[Run] 配置文件: ${CONFIG_PATH}"
echo ""

cd "${SCRIPT_DIR}"
python3 -m main.train --config "${CONFIG_PATH}"
