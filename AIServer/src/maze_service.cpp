#include "maze_service.h"
#include "logger.h"

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>

// ---- 构造函数 ----
MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : config_(config) {
    LOG_INFO("MazeService", "运行模式: %d (%s)", config_.server.run_mode,
             config_.server.run_mode == 1 ? "训练" :
             config_.server.run_mode == 2 ? "推理" : "A*测试");

    // 初始化随机种子
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // 训练模式下连接 Learner
    if (config_.server.run_mode == 1) {
        ConnectLearner();
    }
}

// ---- 连接 Learner gRPC 服务 ----
void MazeServiceImpl::ConnectLearner() {
    std::string target = config_.learner.host + ":" + std::to_string(config_.learner.port);
    LOG_INFO("MazeService", "连接 Learner: %s", target.c_str());

    learner_channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    learner_stub_ = maze::LearnerService::NewStub(learner_channel_);

    // 尝试等待连接就绪（最多 5 秒）
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    if (learner_channel_->WaitForConnected(deadline)) {
        learner_connected_ = true;
        LOG_INFO("MazeService", "Learner 连接成功");
    } else {
        learner_connected_ = true;  // 仍然标记为已连接，gRPC 会自动重连
        LOG_WARN("MazeService", "Learner 连接超时，将在发送时重试");
    }
}

// ---- 随机策略采样 ----
int MazeServiceImpl::ChooseRandomAction() {
    return std::rand() % 9;
}

// ---- 构建观测向量（简化版 5 维，验证数据流用）----
void MazeServiceImpl::BuildObs(int gx, int gy, std::vector<float>& obs) {
    obs.clear();
    obs.resize(5, 0.0f);

    // [0-1] 归一化网格位置
    obs[0] = (grid_cols_ > 0) ? static_cast<float>(gx) / grid_cols_ : 0.0f;
    obs[1] = (grid_rows_ > 0) ? static_cast<float>(gy) / grid_rows_ : 0.0f;

    // [2-3] 目标方向（归一化）
    float dx = static_cast<float>(end_gx_ - gx);
    float dy = static_cast<float>(end_gy_ - gy);
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0.0f) {
        obs[2] = dx / dist;
        obs[3] = dy / dist;
    }

    // [4] 目标距离（归一化）
    float max_dist = std::sqrt(static_cast<float>(grid_cols_ * grid_cols_ + grid_rows_ * grid_rows_));
    obs[4] = (max_dist > 0.0f) ? dist / max_dist : 0.0f;
}

// ---- 计算奖励（简化版：仅通关奖励 + 步数惩罚）----
float MazeServiceImpl::CalcReward(int agent_id, int gx, int gy, bool is_done) {
    float reward = 0.0f;

    // 步数惩罚
    reward -= 0.005f;

    // 通关奖励
    if (is_done && gx == end_gx_ && gy == end_gy_) {
        reward += 10.0f;
    }

    return reward;
}

// ---- 收集单帧样本 ----
void MazeServiceImpl::CollectSample(int agent_id, int gx, int gy, int action, bool is_done) {
    // 构建观测向量
    std::vector<float> obs;
    BuildObs(gx, gy, obs);

    // 计算奖励
    float reward = CalcReward(agent_id, gx, gy, is_done);

    // 构建 Sample
    maze::Sample sample;
    for (float v : obs) {
        sample.add_obs(v);
    }
    sample.set_action(action);
    sample.set_reward(reward);
    sample.set_old_log_prob(-2.197f);   // log(1/9) ≈ -2.197，随机策略占位
    sample.set_old_vpred(0.0f);         // 占位
    sample.set_advantage(0.0f);         // 占位（Learner 端重算）
    sample.set_td_return(0.0f);         // 占位（Learner 端重算）
    sample.set_mask(1.0f);              // 有效样本

    sample_cache_.push_back(std::move(sample));

    LOG_FILE("Sample", "Agent[%d] grid=(%d,%d) action=%d reward=%.3f cache_size=%zu",
             agent_id, gx, gy, action, reward, sample_cache_.size());

    // 达到批量大小时发送
    if (static_cast<int>(sample_cache_.size()) >= config_.learner.sample_batch_size) {
        FlushSamples();
    }
}

// ---- 批量发送样本到 Learner ----
void MazeServiceImpl::FlushSamples() {
    if (sample_cache_.empty()) return;

    if (!learner_stub_) {
        LOG_WARN("MazeService", "Learner 未连接，丢弃 %zu 个样本", sample_cache_.size());
        sample_cache_.clear();
        return;
    }

    // 构建 SampleBatch
    maze::SampleBatch batch;
    batch.set_episode_id(current_episode_id_);
    batch.set_agent_id(0);
    for (auto& s : sample_cache_) {
        *batch.add_samples() = std::move(s);
    }

    int batch_size = batch.samples_size();
    sample_cache_.clear();

    // 发送
    maze::SampleResponse response;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    grpc::Status status = learner_stub_->SendSamples(&ctx, batch, &response);

    if (status.ok()) {
        LOG_INFO("MazeService", "样本发送成功: %d 个, Learner 模型版本: %d",
                 batch_size, response.model_version());
    } else {
        LOG_WARN("MazeService", "样本发送失败: %s (code=%d)",
                 status.error_message().c_str(), static_cast<int>(status.error_code()));
    }
}

// ---- 初始化单个 Agent 的寻路器 ----
void MazeServiceImpl::InitAgentSolver(AgentRuntime& agent) {
    agent.solver.Init(map_width_, map_height_, config_.strategy.grid_size);

    // 内部隔墙（与 Client 端 LoadWalls 一致，不添加外围边界墙壁，网格越界检查天然阻止越界）
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

    // 计算网格参数
    int gs = config_.strategy.grid_size;
    grid_cols_ = static_cast<int>(std::ceil(map_width_ / gs));
    grid_rows_ = static_cast<int>(std::ceil(map_height_ / gs));
    end_gx_ = static_cast<int>(end_x_ / gs);
    end_gy_ = static_cast<int>(end_y_ / gs);

    LOG_INFO("MazeService", "Init: agent_num=%d, map=%.0fx%.0f, start=(%.0f,%.0f), end=(%.0f,%.0f)",
             agent_num, map_width_, map_height_, start_x_, start_y_, end_x_, end_y_);
    LOG_INFO("MazeService", "网格: %dx%d, 终点网格=(%d,%d)", grid_cols_, grid_rows_, end_gx_, end_gy_);

    LOG_FILE("RPC:Init", "InitReq: agent_num=%d, map_size=(%.1f,%.1f), start_pos=(%.1f,%.1f), end_pos=(%.1f,%.1f)",
             agent_num, map_width_, map_height_, start_x_, start_y_, end_x_, end_y_);

    // 初始化每个 Agent 的运行时状态
    agents_.clear();
    for (int i = 0; i < agent_num; ++i) {
        AgentRuntime& agent = agents_[i];
        agent.prev_grid_x = -1;
        agent.prev_grid_y = -1;
        agent.reached_goal = false;

        if (config_.server.run_mode == 3) {
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

    LOG_FILE("RPC:Update", "UpdateReq: frame_id=%d, agents_size=%d", frame_id, req->agents_size());

    for (int i = 0; i < req->agents_size(); ++i) {
        const auto& state = req->agents(i);
        int agent_id = state.agent_id();
        float pos_x = state.pos().x();
        float pos_y = state.pos().y();
        bool is_done = state.is_done();

        int cur_gx = static_cast<int>(pos_x);
        int cur_gy = static_cast<int>(pos_y);

        LOG_FILE("RPC:Update", "  Agent[%d]: pos=(%.1f,%.1f) grid=(%d,%d) done=%s",
                 agent_id, pos_x, pos_y, cur_gx, cur_gy, is_done ? "true" : "false");

        auto* action = rsp->add_actions();
        action->set_agent_id(agent_id);

        // 已结束的 Agent 不下发动作
        if (is_done) {
            action->set_action_id(0);
            LOG_FILE("RPC:Update", "  Agent[%d]: action=0 (done)", agent_id);
            continue;
        }

        // 查找 Agent 运行时数据
        auto it = agents_.find(agent_id);
        if (it == agents_.end()) {
            action->set_action_id(0);
            LOG_WARN("MazeService", "Agent[%d] 未找到运行时数据", agent_id);
            continue;
        }

        AgentRuntime& agent = it->second;
        int chosen_action = 0;

        // ---- 根据运行模式选择动作 ----
        if (config_.server.run_mode == 3) {
            // A* 测试模式
            if (config_.strategy.replan_interval > 0 &&
                frame_id > 0 &&
                frame_id % config_.strategy.replan_interval == 0) {
                float world_x = (cur_gx + 0.5f) * config_.strategy.grid_size;
                float world_y = (cur_gy + 0.5f) * config_.strategy.grid_size;
                agent.path_valid = agent.solver.PlanPath(world_x, world_y, end_x_, end_y_);
                if (!agent.path_valid) {
                    LOG_WARN("MazeService", "Agent[%d] 路径规划失败 frame=%d grid=(%d,%d)",
                             agent_id, frame_id, cur_gx, cur_gy);
                }
            }

            if (agent.path_valid) {
                chosen_action = agent.solver.GetAction(cur_gx, cur_gy);
            }
            LOG_FILE("RPC:Update", "  Agent[%d]: action=%d (astar) grid=(%d,%d)",
                     agent_id, chosen_action, cur_gx, cur_gy);
        } else {
            // 随机策略（训练模式默认）
            chosen_action = ChooseRandomAction();
            LOG_FILE("RPC:Update", "  Agent[%d]: action=%d (random) grid=(%d,%d)",
                     agent_id, chosen_action, cur_gx, cur_gy);
        }

        action->set_action_id(chosen_action);
        agent.last_action = chosen_action;

        // ---- 训练模式：收集样本 ----
        if (config_.server.run_mode == 1) {
            CollectSample(agent_id, cur_gx, cur_gy, chosen_action, is_done);
            agent.prev_grid_x = cur_gx;
            agent.prev_grid_y = cur_gy;
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
    current_episode_id_ = ep_id;

    LOG_INFO("MazeService", "Episode %d 结束", ep_id);
    LOG_FILE("RPC:EndEpisode", "EpisodeEndReq: episode_id=%d", ep_id);

    // 训练模式：发送剩余样本
    if (config_.server.run_mode == 1) {
        if (!sample_cache_.empty()) {
            LOG_INFO("MazeService", "Episode %d 结束，发送剩余 %zu 个样本",
                     ep_id, sample_cache_.size());
            FlushSamples();
        }
    }

    // 重置 Agent 状态
    if (config_.server.run_mode == 3) {
        for (auto& [id, agent] : agents_) {
            agent.path_valid = agent.solver.PlanPath(start_x_, start_y_, end_x_, end_y_);
        }
    }

    // 重置训练状态
    for (auto& [id, agent] : agents_) {
        agent.prev_grid_x = -1;
        agent.prev_grid_y = -1;
        agent.reached_goal = false;
    }

    rsp->set_ret_code(0);
    LOG_FILE("RPC:EndEpisode", "EpisodeEndRsp: ret_code=0");

    return grpc::Status::OK;
}
