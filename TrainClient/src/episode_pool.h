#pragma once

#include "maze_env.h"
#include "grpc_client.h"
#include "config_loader.h"

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

// ---- Episode Worker（单个并行执行单元）----
// 每个 Worker 持有独立的 MazeEnv 和 GrpcClient，
// 可在线程池中独立运行一个完整 Episode。
struct EpisodeWorker {
    int          worker_id  = 0;        // Worker 编号
    MazeEnv      env;                   // 独立环境实例
    GrpcClient   client;                // 独立 gRPC 连接
    int          session_id = 0;        // AIServer 分配的会话 ID
    bool         in_use     = false;    // 是否正在使用
};

// ---- Episode 对象池 ----
// 管理固定数量的 EpisodeWorker，线程安全的借出/归还。
class EpisodePool {
public:
    EpisodePool() = default;
    ~EpisodePool() = default;

    // 初始化对象池：创建 pool_size 个 Worker，每个建立独立 gRPC 连接
    // 返回成功初始化的 Worker 数量
    int Init(int pool_size, const ClientConfig& cfg);

    // 借出一个空闲 Worker（阻塞等待直到有可用 Worker）
    EpisodeWorker* Acquire();

    // 归还 Worker
    void Release(int worker_id);

    // 关闭对象池：断开所有 Worker 的 gRPC 连接，释放资源
    void Shutdown();

    // 获取池大小
    int GetPoolSize() const { return static_cast<int>(workers_.size()); }

private:
    std::vector<std::unique_ptr<EpisodeWorker>> workers_;  // 堆分配，避免移动语义问题
    std::queue<int>            free_ids_;       // 空闲 Worker ID 队列
    std::mutex                 mutex_;
    std::condition_variable    cv_;
};
