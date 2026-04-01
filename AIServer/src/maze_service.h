#pragma once

#include "maze.grpc.pb.h"
#include "astar_solver.h"
#include "config_loader.h"
#include "session_manager.h"

#include <grpcpp/grpcpp.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

// ---- MazeService gRPC 服务实现（含 A* 测试策略 + 训练模式样本收集）----
class MazeServiceImpl final : public maze::MazeService::Service {
public:
    explicit MazeServiceImpl(const AIServerConfig& config);

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
                       int action, bool is_done);
    void FlushSamples(SessionManager::Session& session,             // 批量发送样本到 Learner
                      bool is_episode_end);
    int  ChooseRandomAction();                                      // 随机策略采样
    void BuildObs(const SessionManager::Session& session,           // 构建观测向量
                  int gx, int gy, std::vector<float>& obs);
    float CalcReward(const SessionManager::Session& session,        // 计算奖励
                     int agent_id, int gx, int gy, bool is_done);
};
