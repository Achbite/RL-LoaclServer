# ============================================================================
# AIServer - AI 决策服务（gRPC 服务端）
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(maze_aiserver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖查找 ---
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# --- 源文件 ---
set(SOURCES
    main/main.cpp
    src/grpc/maze_service.cpp
    src/ai/maze_reward.cpp
    src/ai/astar_solver.cpp
    src/config/config_loader.cpp
    src/session/session_manager.cpp
    src/ai/onnx_inferencer.cpp
)

# --- Proto 生成文件（由 gen_proto.sh 预先生成）---
set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_gen)
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/maze.pb.cc
    ${PROTO_GEN_DIR}/maze.grpc.pb.cc
)

# --- 可执行文件 ---
add_executable(${PROJECT_NAME} ${SOURCES} ${PROTO_SOURCES})

# --- 头文件搜索路径 ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
    /usr/local/include/onnxruntime      # ONNX Runtime 头文件
)

# --- 链接库 ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    protobuf::libprotobuf
    ${GRPC_LIBRARIES}
    pthread
    onnxruntime                          # ONNX Runtime 推理库
)
