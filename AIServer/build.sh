#!/bin/bash
# ============================================================================
# AIServer 编译脚本（Docker 容器内执行）
# 用法：cd /workspace/AIServer && ./build.sh [--clean]
# 选项：
#   (无参数)   增量编译（利用 ccache + Ninja 缓存，仅重编变更文件）
#   --clean    全量编译（清理 build 目录后重新编译）
# ============================================================================

set -e

echo "============================================"
echo "  迷宫训练框架 - AIServer 编译"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ---- 解析参数 ----
CLEAN_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN_BUILD=true ;;
    esac
done

# ---- 全量编译模式：清理 build 目录 ----
if $CLEAN_BUILD; then
    echo "[Build] --clean 模式：清理编译产物..."
    rm -rf "${BUILD_DIR}"
fi

# ---- 1. 生成 Proto 代码 ----
echo "[Build] 生成 Protobuf + gRPC 代码..."
bash /workspace/TrainClient/proto/gen_proto.sh

# ---- 2. CMake 配置（仅在需要时执行）----
mkdir -p "${BUILD_DIR}"

# 路径不匹配时自动清理旧缓存
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    CACHED_SRC=$(grep "CMAKE_HOME_DIRECTORY" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    if [ -n "${CACHED_SRC}" ] && [ "${CACHED_SRC}" != "${SCRIPT_DIR}" ]; then
        echo "[Build] 检测到旧缓存路径: ${CACHED_SRC}，清理中..."
        rm -rf "${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
    fi
fi

# 仅在 CMakeCache 不存在时执行 configure（增量编译跳过此步）
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo ""
    echo "[Build] CMake 配置..."
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
else
    echo "[Build] CMake 缓存有效，跳过 configure"
fi

# ---- 3. 编译（Ninja 自动增量编译，仅重编变更文件）----
echo ""
echo "[Build] 编译中..."
cmake --build "${BUILD_DIR}" --parallel

# ---- 4. 显示 ccache 统计（便于确认缓存命中情况）----
if command -v ccache &>/dev/null; then
    echo ""
    echo "[Build] ccache 统计:"
    ccache -s 2>/dev/null | grep -E "(Hits|Misses|cache size)" || true
fi

echo ""
echo "============================================"
echo "[Build] 编译完成！可执行文件: ${BUILD_DIR}/maze_aiserver"
echo "============================================"
