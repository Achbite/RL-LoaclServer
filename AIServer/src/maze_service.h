#pragma once

#include "maze.grpc.pb.h"
#include "astar_solver.h"
#include "config_loader.h"

#include <grpcpp/grpcpp.h>
#include <unordered_map>
#include <mutex>

// ---- MazeService gRPC 服务实现（含 A* 测试策略）----
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

    // Agent 运行时数据
    std::unordered_map<int, AgentRuntime> agents_;
    std::mutex mutex_;

    // 初始化单个 Agent 的寻路器
    void InitAgentSolver(AgentRuntime& agent);
};
