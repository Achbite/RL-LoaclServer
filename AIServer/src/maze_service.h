#pragma once

#include "maze.grpc.pb.h"
#include "astar_solver.h"
#include "config_loader.h"

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
    // ---- Agent 运行时状态（服务端维护）----
    struct AgentRuntime {
        AStarSolver solver;         // 每个 Agent 独立的寻路器
        int         last_action = 0;
        bool        path_valid  = false;

        // 训练模式：帧样本缓存
        int   prev_grid_x = -1;    // 上一帧网格坐标（用于计算距离变化）
        int   prev_grid_y = -1;
        bool  reached_goal = false; // 本 Episode 是否到达终点
    };

    AIServerConfig config_;

    // 地图参数（从 InitReq 获取）
    float map_width_  = 0.0f;
    float map_height_ = 0.0f;
    float start_x_    = 0.0f;
    float start_y_    = 0.0f;
    float end_x_      = 0.0f;
    float end_y_      = 0.0f;
    bool  initialized_ = false;

    // 网格参数（Init 时计算）
    int end_gx_ = 0;               // 终点网格坐标
    int end_gy_ = 0;
    int grid_cols_ = 0;             // 网格列数
    int grid_rows_ = 0;             // 网格行数

    // Agent 运行时数据
    std::unordered_map<int, AgentRuntime> agents_;
    std::mutex mutex_;

    // 初始化单个 Agent 的寻路器
    void InitAgentSolver(AgentRuntime& agent);

    // --- Learner gRPC 客户端（训练模式）---
    std::shared_ptr<grpc::Channel>                    learner_channel_;
    std::unique_ptr<maze::LearnerService::Stub>       learner_stub_;
    bool learner_connected_ = false;

    // --- 样本缓存（训练模式）---
    std::vector<maze::Sample> sample_cache_;    // 全局样本缓存，积累后批量发送
    int current_episode_id_ = 0;                // 当前 Episode ID

    // 训练模式辅助方法
    void ConnectLearner();                                          // 连接 Learner gRPC 服务
    void CollectSample(int agent_id, int gx, int gy,               // 收集单帧样本
                       int action, bool is_done);
    void FlushSamples();                                            // 批量发送样本到 Learner
    int  ChooseRandomAction();                                      // 随机策略采样
    void BuildObs(int gx, int gy, std::vector<float>& obs);        // 构建观测向量（简化版）
    float CalcReward(int agent_id, int gx, int gy, bool is_done);  // 计算奖励（简化版）
};
