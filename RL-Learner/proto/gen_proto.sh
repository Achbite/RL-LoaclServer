#!/bin/bash
# ============================================================================
# Python Protobuf + gRPC 代码生成脚本
# 用法：在 WSL 中执行 bash RL-Learner/proto/gen_proto.sh
# 前置依赖：pip install grpcio-tools protobuf
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_SRC="${SCRIPT_DIR}/../../Client/proto/maze.proto"
PYTHON_OUT="${SCRIPT_DIR}"

# 生成 Python protobuf + gRPC 绑定
python3 -m grpc_tools.protoc \
    --proto_path="$(dirname "${PROTO_SRC}")" \
    --python_out="${PYTHON_OUT}" \
    --grpc_python_out="${PYTHON_OUT}" \
    "${PROTO_SRC}"

# 修复 Python 相对导入（grpc_tools 生成的代码使用绝对导入，需要修正）
sed -i 's/^import maze_pb2/from . import maze_pb2/' "${PYTHON_OUT}/maze_pb2_grpc.py" 2>/dev/null || true

echo "[Learner] Python protobuf + gRPC 代码已生成到: ${PYTHON_OUT}"
echo "生成文件："
echo "  - maze_pb2.py        (Protobuf 消息定义)"
echo "  - maze_pb2_grpc.py   (gRPC 服务桩代码)"
echo "完成。"
