#include "maze_service.h"
#include "logger.h"

#include <cstdlib>

// ---- 构造函数 ----
MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : config_(config) {
    LOG_INFO("MazeService", "策略模式: %s", config_.strategy.mode.c_str());
}

// ---- 初始化单个 Agent 的寻路器 ----
void MazeServiceImpl::InitAgentSolver(AgentRuntime& agent) {
    agent.solver.Init(map_width_, map_height_, config_.strategy.grid_size);

    // TODO: 后续从地图文件加载墙壁，当前使用与 test_maze.json 一致的硬编码墙壁
    // 外围边界
    agent.solver.AddWall(0, 0, map_width_, 0, 100);
    agent.solver.AddWall(map_width_, 0, map_width_, map_height_, 100);
    agent.solver.AddWall(map_width_, map_height_, 0, map_height_, 100);
    agent.solver.AddWall(0, map_height_, 0, 0, 100);

    // 内部隔墙（与 test_maze.json 一致）
    agent.solver.AddWall(5000, 0, 5000, 14000, 100);
    agent.solver.AddWall(10000, 6000, 10000, 20000, 100);
    agent.solver.AddWall(15000, 0, 15000, 14000, 100);

    // 规划路径
    agent.path_valid = agent.solver.PlanPath(start_x_, start_y_, end_x_, end_y_);
}

// ---- Init RPC：接收 Client 的初始化请求 ----
grpc::Status MazeServiceImpl::Init(grpc::ServerContext* ctx,
                                   const maze::InitReq* req,
                                   maze::InitRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);

    map_width_  = req->map_size().x();
    map_height_ = req->map_size().y();
    start_x_    = req->start_pos().x();
    start_y_    = req->start_pos().y();
    end_x_      = req->end_pos().x();
    end_y_      = req->end_pos().y();

    int agent_num = req->agent_num();

    LOG_INFO("MazeService", "Init: agent_num=%d, map=%.0fx%.0f, start=(%.0f,%.0f), end=(%.0f,%.0f)",
             agent_num, map_width_, map_height_, start_x_, start_y_, end_x_, end_y_);

    // 写入文件详细日志
    LOG_FILE("RPC:Init", "InitReq: agent_num=%d, map_size=(%.1f,%.1f), start_pos=(%.1f,%.1f), end_pos=(%.1f,%.1f)",
             agent_num, map_width_, map_height_, start_x_, start_y_, end_x_, end_y_);

    // 初始化每个 Agent 的寻路器
    agents_.clear();
    for (int i = 0; i < agent_num; ++i) {
        AgentRuntime& agent = agents_[i];
        if (config_.strategy.mode == "astar") {
            InitAgentSolver(agent);
        }
    }

    initialized_ = true;
    rsp->set_ret_code(0);

    LOG_FILE("RPC:Init", "InitRsp: ret_code=0");

    return grpc::Status::OK;
}

// ---- Update RPC：接收帧状态，返回动作 ----
grpc::Status MazeServiceImpl::Update(grpc::ServerContext* ctx,
                                     const maze::UpdateReq* req,
                                     maze::UpdateRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        LOG_WARN("MazeService", "Update 调用时尚未初始化");
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "未初始化");
    }

    int frame_id = req->frame_id();

    // 文件日志：记录请求帧信息
    LOG_FILE("RPC:Update", "UpdateReq: frame_id=%d, agents_size=%d", frame_id, req->agents_size());

    for (int i = 0; i < req->agents_size(); ++i) {
        const auto& state = req->agents(i);
        int agent_id = state.agent_id();
        float pos_x = state.pos().x();
        float pos_y = state.pos().y();
        bool is_done = state.is_done();

        // 文件日志：记录每个 Agent 的状态
        LOG_FILE("RPC:Update", "  Agent[%d]: pos=(%.1f,%.1f), done=%s",
                 agent_id, pos_x, pos_y, is_done ? "true" : "false");

        auto* action = rsp->add_actions();
        action->set_agent_id(agent_id);

        // 已结束的 Agent 不下发动作
        if (is_done) {
            action->set_action_id(0);
            LOG_FILE("RPC:Update", "  Agent[%d]: action=0 (done)", agent_id);
            continue;
        }

        // 根据策略模式选择动作
        auto it = agents_.find(agent_id);
        if (it == agents_.end()) {
            action->set_action_id(0);
            LOG_WARN("MazeService", "Agent[%d] 未找到运行时数据", agent_id);
            continue;
        }

        AgentRuntime& agent = it->second;

        if (config_.strategy.mode == "astar") {
            // A* 策略：定期重新规划
            if (config_.strategy.replan_interval > 0 &&
                frame_id > 0 &&
                frame_id % config_.strategy.replan_interval == 0) {
                agent.path_valid = agent.solver.PlanPath(pos_x, pos_y, end_x_, end_y_);
                if (!agent.path_valid) {
                    LOG_WARN("MazeService", "Agent[%d] 路径规划失败 frame=%d pos=(%.1f,%.1f)",
                             agent_id, frame_id, pos_x, pos_y);
                }
            }

            if (agent.path_valid) {
                int act = agent.solver.GetAction(pos_x, pos_y);
                action->set_action_id(act);
                agent.last_action = act;
                LOG_FILE("RPC:Update", "  Agent[%d]: action=%d (astar)", agent_id, act);
            } else {
                action->set_action_id(0);
                LOG_FILE("RPC:Update", "  Agent[%d]: action=0 (no_path)", agent_id);
            }
        } else {
            // 随机策略
            int act = std::rand() % 9;
            action->set_action_id(act);
            LOG_FILE("RPC:Update", "  Agent[%d]: action=%d (random)", agent_id, act);
        }
    }

    return grpc::Status::OK;
}

// ---- EndEpisode RPC：Episode 结束处理 ----
grpc::Status MazeServiceImpl::EndEpisode(grpc::ServerContext* ctx,
                                         const maze::EpisodeEndReq* req,
                                         maze::EpisodeEndRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);

    int ep_id = req->episode_id();
    LOG_INFO("MazeService", "Episode %d 结束", ep_id);
    LOG_FILE("RPC:EndEpisode", "EpisodeEndReq: episode_id=%d", ep_id);

    // 重新规划所有 Agent 的路径（下一个 Episode 使用）
    if (config_.strategy.mode == "astar") {
        for (auto& [id, agent] : agents_) {
            agent.path_valid = agent.solver.PlanPath(start_x_, start_y_, end_x_, end_y_);
        }
    }

    rsp->set_ret_code(0);
    LOG_FILE("RPC:EndEpisode", "EpisodeEndRsp: ret_code=0");

    return grpc::Status::OK;
}
