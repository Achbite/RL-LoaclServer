# ============================================================================
# TrainClient - 迷宫环境模拟器
# 构建目标：
#   maze_client  — 单 Episode 调试模式
#   maze_train   — 并行训练模式（线程池 + 对象池）
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(maze_trainclient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 依赖查找 ---
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# --- 共享源文件（不含 main）---
set(SHARED_SOURCES
    src/grpc_client.cpp
    src/maze_env.cpp
    src/config_loader.cpp
    src/viz_recorder.cpp
    src/episode_pool.cpp
)

# --- Proto 生成文件（由 gen_proto.sh 预先生成）---
set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_gen)
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/maze.pb.cc
    ${PROTO_GEN_DIR}/maze.grpc.pb.cc
)

# --- 公共头文件搜索路径 ---
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

# --- 公共链接库 ---
set(COMMON_LIBS
    protobuf::libprotobuf
    ${GRPC_LIBRARIES}
    pthread
)

# ---- 目标 1：maze_client（单 Episode 调试）----
add_executable(maze_client main/main.cpp ${SHARED_SOURCES} ${PROTO_SOURCES})
target_include_directories(maze_client PRIVATE ${COMMON_INCLUDES})
target_link_libraries(maze_client PRIVATE ${COMMON_LIBS})

# ---- 目标 2：maze_train（并行训练）----
add_executable(maze_train main/train_main.cpp ${SHARED_SOURCES} ${PROTO_SOURCES})
target_include_directories(maze_train PRIVATE ${COMMON_INCLUDES})
target_link_libraries(maze_train PRIVATE ${COMMON_LIBS})
