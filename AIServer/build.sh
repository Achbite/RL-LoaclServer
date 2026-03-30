#!/bin/bash
# ============================================================================
# AIServer 编译脚本（Docker 容器内执行）
# 用法：cd /workspace/AIServer && ./build.sh
# ============================================================================

set -e

echo "============================================"
echo "  迷宫训练框架 - AIServer 编译"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ---- 1. 生成 Proto 代码 ----
echo "[Build] 生成 Protobuf + gRPC 代码..."
bash /workspace/Client/proto/gen_proto.sh

# ---- 2. CMake 配置 ----
echo ""
echo "[Build] CMake 配置..."
mkdir -p "${BUILD_DIR}"

# 清理旧缓存（路径不匹配时自动清理）
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    CACHED_SRC=$(grep "CMAKE_HOME_DIRECTORY" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    if [ -n "${CACHED_SRC}" ] && [ "${CACHED_SRC}" != "${SCRIPT_DIR}" ]; then
        echo "[Build] 检测到旧缓存路径: ${CACHED_SRC}，清理中..."
        rm -rf "${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
    fi
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

# ---- 3. 编译 ----
echo ""
echo "[Build] 编译中..."
cmake --build "${BUILD_DIR}" --parallel

echo ""
echo "============================================"
echo "[Build] 编译完成！可执行文件: ${BUILD_DIR}/maze_aiserver"
echo "============================================"
