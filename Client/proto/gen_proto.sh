#!/bin/bash
# ============================================================================
# C++ Protobuf + gRPC 代码生成脚本
# 用法：bash Client/proto/gen_proto.sh
# 运行环境：Docker 容器（Ubuntu 24.04）
# ============================================================================

# Linux / Docker 环境路径
export PATH="/usr/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_FILE="${SCRIPT_DIR}/maze.proto"

# 检查 protoc 是否可用
if ! command -v protoc &> /dev/null; then
    echo "[错误] protoc 未找到，请先安装 protobuf 编译器"
    exit 1
fi

# 检查 grpc_cpp_plugin 是否可用
GRPC_PLUGIN=$(command -v grpc_cpp_plugin 2>/dev/null)
if [ -z "${GRPC_PLUGIN}" ]; then
    echo "[错误] grpc_cpp_plugin 未找到，请先安装 protobuf-compiler-grpc"
    exit 1
fi

# ---- 1. 生成 Client 端 C++ 代码（Protobuf + gRPC）----
CLIENT_OUT="${SCRIPT_DIR}/../src/proto_gen"
mkdir -p "${CLIENT_OUT}"

protoc \
    --proto_path="${SCRIPT_DIR}" \
    --cpp_out="${CLIENT_OUT}" \
    --grpc_out="${CLIENT_OUT}" \
    --plugin=protoc-gen-grpc="${GRPC_PLUGIN}" \
    "${PROTO_FILE}"

echo "[Client] C++ protobuf + gRPC 代码已生成到: ${CLIENT_OUT}"

# ---- 2. 生成 AIServer 端 C++ 代码（Protobuf + gRPC）----
AISERVER_OUT="${SCRIPT_DIR}/../../AIServer/src/proto_gen"
mkdir -p "${AISERVER_OUT}"

protoc \
    --proto_path="${SCRIPT_DIR}" \
    --cpp_out="${AISERVER_OUT}" \
    --grpc_out="${AISERVER_OUT}" \
    --plugin=protoc-gen-grpc="${GRPC_PLUGIN}" \
    "${PROTO_FILE}"

echo "[AIServer] C++ protobuf + gRPC 代码已生成到: ${AISERVER_OUT}"

echo "完成。"
