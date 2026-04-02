#pragma once

#include "maze.grpc.pb.h"
#include "ai/astar_solver.h"
#include "config/config_loader.h"
#include "session/session_manager.h"
#include "ai/onnx_inferencer.h"
#include "ai/maze_reward.h"

#include <grpcpp/grpcpp.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>

// ---- MazeService gRPC 服务实现（含 A* 测试策略 + 训练模式样本收集）----
class MazeServiceImpl final : public maze::MazeService::Service {
public:
    explicit MazeServiceImpl(const AIServerConfig& config);

    ~MazeServiceImpl();

    // gRPC 接口
    grpc::Status Init(grpc::ServerContext* ctx,
                        const maze::InitReq* req,
                        maze::InitRsp* rsp) override;

    grpc::Status Update(grpc::ServerContext* ctx,
                        const maze::UpdateReq* req,
                        maze::UpdateRsp* rsp) override;

    grpc::Status EndEpisode(grpc::ServerContext* ctx,
                            const maze::EpisodeEndReq* req,
                            maze::EpisodeEndRsp* rsp) override;

private:
    AIServerConfig config_;

    // ---- 会话管理器（替代全局 agents_ 实现并行 Episode 隔离）----
    SessionManager session_mgr_;
    std::mutex mutex_;

    // 初始化单个 Agent 的寻路器
    void InitAgentSolver(SessionManager::AgentRuntime& agent,
                            const SessionManager::Session& session);

    // --- Learner gRPC 客户端（训练模式）---
    std::shared_ptr<grpc::Channel>                    learner_channel_;
    std::unique_ptr<maze::LearnerService::Stub>       learner_stub_;
    bool learner_connected_ = false;

    // 训练模式辅助方法
    void ConnectLearner();                                          // 连接 Learner gRPC 服务
    void CollectSample(SessionManager::Session& session,            // 收集单帧样本
                        int agent_id, int gx, int gy,
                        int action, float old_log_prob,
                        float old_vpred, bool is_done);
    void FlushAgentSamples(SessionManager::Session& session,        // 批量发送指定 Agent 的样本到 Learner
                            int agent_id, bool is_episode_end);
    int  ChooseRandomAction();                                      // 随机策略采样
    int  SampleAction(const std::vector<float>& probs);             // 按概率分布采样动作
    void BuildObs(const SessionManager::Session& session,           // 构建观测向量（13 维：5 导航 + 8 射线）
                    int gx, int gy, std::vector<float>& obs);

    // ---- ONNX 推理 + 模型热更新 ----
    OnnxInferencer onnx_inferencer_;                                // ONNX 推理器
    std::thread    model_watcher_;                                  // 模型监控线程
    std::atomic<bool> watcher_running_{false};                      // 监控线程运行标志
    std::string    last_model_path_;                                // 上次加载的模型路径
    std::time_t    last_model_mtime_ = 0;                           // 上次模型文件修改时间

    void StartModelWatcher();                                       // 启动模型监控线程
    void StopModelWatcher();                                        // 停止模型监控线程
};
