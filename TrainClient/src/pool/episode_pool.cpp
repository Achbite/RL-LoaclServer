#include "pool/episode_pool.h"
#include "log/logger.h"

// ---- 初始化对象池 ----
int EpisodePool::Init(int pool_size, const ClientConfig& cfg) {
    workers_.reserve(pool_size);
    int success_count = 0;

    for (int i = 0; i < pool_size; ++i) {
        // 堆分配 Worker，避免 GrpcClient(unique_ptr) 的移动构造问题
        auto w = std::make_unique<EpisodeWorker>();
        w->worker_id = i;
        w->in_use = false;

        // 初始化独立环境实例
        w->env.Init(cfg);

        // 建立独立 gRPC 连接
        if (w->client.Connect(cfg.network.server_host, cfg.network.server_port)) {
            // 使用 worker_id 作为 session_id（从 1 开始，0 保留给单 Episode 模式）
            w->session_id = i + 1;
            free_ids_.push(i);
            success_count++;
            LOG_INFO("EpisodePool", "Worker[%d] 初始化成功, session_id=%d", i, w->session_id);
        } else {
            LOG_ERROR("EpisodePool", "Worker[%d] gRPC 连接失败", i);
        }

        workers_.push_back(std::move(w));
    }

    LOG_INFO("EpisodePool", "对象池初始化完成: %d/%d 个 Worker 就绪", success_count, pool_size);
    return success_count;
}

// ---- 借出一个空闲 Worker ----
EpisodeWorker* EpisodePool::Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !free_ids_.empty(); });

    int id = free_ids_.front();
    free_ids_.pop();
    workers_[id]->in_use = true;

    LOG_FILE("EpisodePool", "借出 Worker[%d], 剩余空闲=%zu", id, free_ids_.size());
    return workers_[id].get();
}

// ---- 归还 Worker ----
void EpisodePool::Release(int worker_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        workers_[worker_id]->in_use = false;
        free_ids_.push(worker_id);
        LOG_FILE("EpisodePool", "归还 Worker[%d], 空闲=%zu", worker_id, free_ids_.size());
    }
    cv_.notify_one();
}

// ---- 关闭对象池，断开所有 gRPC 连接 ----
void EpisodePool::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& w : workers_) {
        if (w) {
            w->client.Disconnect();
        }
    }
    LOG_INFO("EpisodePool", "对象池已关闭，%zu 个 Worker 的 gRPC 连接已断开",
             workers_.size());
}
