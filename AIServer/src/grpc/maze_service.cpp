#include "grpc/maze_service.h"
#include "log/logger.h"

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <numeric>
#include <sys/stat.h>

// ---- 构造函数 ----
MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : config_(config) {
    LOG_INFO("MazeService", "运行模式: %d (%s)", config_.server.run_mode,
             config_.server.run_mode == 1 ? "训练" :
             config_.server.run_mode == 2 ? "推理" : "A*测试");

    // 初始化随机种子
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // 训练模式下连接 Learner 并启动模型监控
    if (config_.server.run_mode == 1) {
        ConnectLearner();
        StartModelWatcher();
    }

    // 推理模式下直接加载本地模型
    if (config_.server.run_mode == 2) {
        std::string model_path = config_.model.local_dir + "/" +
                                 config_.model.save_name + ".onnx";
        if (!onnx_inferencer_.LoadModel(model_path)) {
            LOG_ERROR("MazeService", "推理模式加载模型失败: %s", model_path.c_str());
        }
    }
}

// ---- 析构函数 ----
MazeServiceImpl::~MazeServiceImpl() {
    StopModelWatcher();
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

// ---- 按概率分布采样动作 ----
int MazeServiceImpl::SampleAction(const std::vector<float>& probs) {
    // 生成 [0, 1) 随机数
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    // 累积概率采样
    float cumsum = 0.0f;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return i;
        }
    }
    return static_cast<int>(probs.size()) - 1;
}

// ---- 启动模型监控线程 ----
void MazeServiceImpl::StartModelWatcher() {
    watcher_running_.store(true);
    model_watcher_ = std::thread([this]() {
        std::string model_path = config_.model.p2p_dir + "/" +
                                 config_.model.save_name + ".onnx";
        int poll_interval = config_.model.poll_interval;

        LOG_INFO("ModelWatcher", "模型监控启动: path=%s, interval=%ds",
                 model_path.c_str(), poll_interval);

        while (watcher_running_.load()) {
            // 检查文件是否存在且修改时间变化
            struct stat file_stat;
            if (stat(model_path.c_str(), &file_stat) == 0) {
                std::time_t mtime = file_stat.st_mtime;
                if (mtime != last_model_mtime_) {
                    // 等待 1 秒确保文件写入完成（shutil.copy2 非原子操作）
                    std::this_thread::sleep_for(std::chrono::seconds(1));

                    if (onnx_inferencer_.LoadModel(model_path)) {
                        last_model_mtime_ = mtime;
                        LOG_INFO("ModelWatcher", "新模型加载成功: %s", model_path.c_str());
                    } else {
                        LOG_WARN("ModelWatcher", "新模型加载失败，保留旧模型");
                    }
                }
            }

            // 轮询等待（每秒检查一次运行标志，支持快速退出）
            for (int i = 0; i < poll_interval && watcher_running_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        LOG_INFO("ModelWatcher", "模型监控已停止");
    });
}

// ---- 停止模型监控线程 ----
void MazeServiceImpl::StopModelWatcher() {
    if (watcher_running_.load()) {
        watcher_running_.store(false);
        if (model_watcher_.joinable()) {
            model_watcher_.join();
        }
    }
}

// ---- 构建观测向量（13 维：5 导航 + 8 射线）----
void MazeServiceImpl::BuildObs(const SessionManager::Session& session,
                               int gx, int gy,
                               const std::vector<float>& client_obs,
                               std::vector<float>& obs) {
    obs.clear();
    obs.resize(13, 0.0f);

    // [0-1] 归一化网格位置
    obs[0] = (session.grid_cols > 0) ? static_cast<float>(gx) / session.grid_cols : 0.0f;
    obs[1] = (session.grid_rows > 0) ? static_cast<float>(gy) / session.grid_rows : 0.0f;

    // [2-3] 目标方向（归一化）
    float dx = static_cast<float>(session.end_gx - gx);
    float dy = static_cast<float>(session.end_gy - gy);
    float dist = std::sqrt(dx * dx + dy * dy);
    if (dist > 0.0f) {
        obs[2] = dx / dist;
        obs[3] = dy / dist;
    }

    // [4] 目标距离（归一化）
    float max_dist = std::sqrt(static_cast<float>(
        session.grid_cols * session.grid_cols + session.grid_rows * session.grid_rows));
    obs[4] = (max_dist > 0.0f) ? dist / max_dist : 0.0f;

    // [5-12] 八方向射线障碍物检测（使用 Client 端传来的射线特征）
    if (client_obs.size() >= 8) {
        for (int d = 0; d < 8; ++d) {
            obs[5 + d] = client_obs[d];
        }
    }
    // 若 Client 未传射线数据（向后兼容），射线维度保持 0.0
}

// ---- 收集单帧样本 ----
void MazeServiceImpl::CollectSample(SessionManager::Session& session,
                                    int agent_id, int gx, int gy,
                                    int action, float old_log_prob,
                                    float old_vpred, bool is_done,
                                    const std::vector<float>& client_obs) {
    // 构建观测向量
    std::vector<float> obs;
    BuildObs(session, gx, gy, client_obs, obs);

    // 计算奖励（独立模块，含分项明细）
    int agent_num = static_cast<int>(session.agents.size());
    RewardDetail reward_detail = MazeReward::Calculate(session, agent_id, gx, gy, is_done, agent_num);
    float reward = reward_detail.total;

    // 更新探索/徘徊辅助状态（奖励计算完成后更新，确保当前帧判断基于历史数据）
    auto agent_it = session.agents.find(agent_id);
    if (agent_it != session.agents.end()) {
        int key = gy * session.grid_cols + gx;
        agent_it->second.visited.insert(key);
        agent_it->second.recent_positions.push_back(key);
        if (static_cast<int>(agent_it->second.recent_positions.size()) > 8) {
            agent_it->second.recent_positions.pop_front();
        }
    }

    // 构建 Sample
    maze::Sample sample;
    for (float v : obs) {
        sample.add_obs(v);
    }
    sample.set_action(action);
    sample.set_reward(reward);

    // 填充奖励分项明细（用于 Dashboard 可视化追踪）
    auto* details_map = sample.mutable_reward_details();
    for (const auto& item : reward_detail.items) {
        (*details_map)[item.first] = item.second;
    }
    sample.set_old_log_prob(old_log_prob);   // 模型推理的真实 log 概率
    sample.set_old_vpred(old_vpred);         // 模型推理的真实价值估计
    sample.set_advantage(0.0f);         // 占位（Learner 端重算）
    sample.set_td_return(0.0f);         // 占位（Learner 端重算）
    sample.set_mask(1.0f);              // 有效样本

    session.agent_sample_caches[agent_id].push_back(std::move(sample));

    LOG_FILE("Sample", "S:%d Agent[%d] grid=(%d,%d) action=%d reward=%.3f cache_size=%zu",
             session.session_id, agent_id, gx, gy, action, reward,
             session.agent_sample_caches[agent_id].size());

    // 达到批量大小时发送（TMax 截断，非 Episode 结束）
    if (static_cast<int>(session.agent_sample_caches[agent_id].size()) >= config_.learner.sample_batch_size) {
        FlushAgentSamples(session, agent_id, false);
    }
}

// ---- 批量发送指定 Agent 的样本到 Learner ----
void MazeServiceImpl::FlushAgentSamples(SessionManager::Session& session,
                                        int agent_id, bool is_episode_end) {
    auto& cache = session.agent_sample_caches[agent_id];

    // 非 Episode 结束时，缓存为空则跳过
    if (cache.empty() && !is_episode_end) return;

    if (!learner_stub_) {
        LOG_WARN("MazeService", "Learner 未连接，丢弃 S:%d Agent[%d] 的 %zu 个样本",
                 session.session_id, agent_id, cache.size());
        cache.clear();
        return;
    }

    // 构建 SampleBatch（agent_id 正确标识）
    maze::SampleBatch batch;
    batch.set_episode_id(session.current_episode_id);
    batch.set_agent_id(agent_id);
    batch.set_is_episode_end(is_episode_end);
    batch.set_session_id(session.session_id);
    for (auto& s : cache) {
        *batch.add_samples() = std::move(s);
    }

    int batch_size = batch.samples_size();
    cache.clear();

    // 发送（失败时重试，利用 gRPC 同步调用的背压机制保证零丢弃）
    int max_retries = config_.learner.max_retries;
    int send_timeout = config_.learner.send_timeout;
    grpc::Status status;

    for (int retry = 0; retry <= max_retries; ++retry) {
        maze::SampleResponse response;
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(send_timeout));

        status = learner_stub_->SendSamples(&ctx, batch, &response);

        if (status.ok()) {
            LOG_INFO("MazeService", "样本发送成功: S:%d Agent[%d] %d 个 ep_end=%s, Learner 模型版本: %d",
                     session.session_id, agent_id, batch_size,
                     is_episode_end ? "true" : "false",
                     response.model_version());
            return;
        }

        // 最后一次重试仍失败，不再 sleep
        if (retry < max_retries) {
            LOG_WARN("MazeService", "样本发送失败，重试 %d/%d: S:%d %s (code=%d)",
                     retry + 1, max_retries, session.session_id,
                     status.error_message().c_str(), static_cast<int>(status.error_code()));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // 所有重试均失败
    LOG_ERROR("MazeService", "样本发送最终失败: S:%d %d 个样本丢失, %s (code=%d)",
              session.session_id, batch_size,
              status.error_message().c_str(), static_cast<int>(status.error_code()));
}

// ---- 初始化单个 Agent 的寻路器 ----
void MazeServiceImpl::InitAgentSolver(SessionManager::AgentRuntime& agent,
                                      const SessionManager::Session& session) {
    agent.solver.Init(session.map_width, session.map_height, config_.strategy.grid_size);

    // 内部隔墙（与 TrainClient 端 LoadWalls 一致）
    agent.solver.AddWall(5000, 0, 5000, 14000, 100);
    agent.solver.AddWall(10000, 6000, 10000, 20000, 100);
    agent.solver.AddWall(15000, 0, 15000, 14000, 100);

    // 规划路径
    agent.path_valid = agent.solver.PlanPath(session.start_x, session.start_y,
                                             session.end_x, session.end_y);
}

// ---- Init RPC：接收 TrainClient 的初始化请求 ----
grpc::Status MazeServiceImpl::Init(grpc::ServerContext* ctx,
                                   const maze::InitReq* req,
                                   maze::InitRsp* rsp) {
    int session_id = req->session_id();

    // 获取或创建会话
    auto* session = session_mgr_.GetOrCreateSession(session_id);
    if (!session) {
        LOG_ERROR("MazeService", "无法创建会话 session_id=%d", session_id);
        rsp->set_ret_code(-1);
        return grpc::Status::OK;
    }

    // 填充地图参数
    session->map_width  = req->map_size().x();
    session->map_height = req->map_size().y();
    session->start_x    = req->start_pos().x();
    session->start_y    = req->start_pos().y();
    session->end_x      = req->end_pos().x();
    session->end_y      = req->end_pos().y();

    int agent_num = req->agent_num();

    // 计算网格参数
    int gs = config_.strategy.grid_size;
    session->grid_cols = static_cast<int>(std::ceil(session->map_width / gs));
    session->grid_rows = static_cast<int>(std::ceil(session->map_height / gs));
    session->end_gx = static_cast<int>(session->end_x / gs);
    session->end_gy = static_cast<int>(session->end_y / gs);

    LOG_INFO("MazeService", "Init: S:%d agent_num=%d, map=%.0fx%.0f, start=(%.0f,%.0f), end=(%.0f,%.0f)",
             session_id, agent_num, session->map_width, session->map_height,
             session->start_x, session->start_y, session->end_x, session->end_y);
    LOG_INFO("MazeService", "网格: S:%d %dx%d, 终点网格=(%d,%d)",
             session_id, session->grid_cols, session->grid_rows,
             session->end_gx, session->end_gy);

    LOG_FILE("RPC:Init", "S:%d InitReq: agent_num=%d, map_size=(%.1f,%.1f), start_pos=(%.1f,%.1f), end_pos=(%.1f,%.1f)",
             session_id, agent_num, session->map_width, session->map_height,
             session->start_x, session->start_y, session->end_x, session->end_y);

    // 初始化每个 Agent 的运行时状态
    session->agents.clear();
    for (int i = 0; i < agent_num; ++i) {
        auto& agent = session->agents[i];
        agent.prev_grid_x = -1;
        agent.prev_grid_y = -1;
        agent.reached_goal = false;
        agent.done_collected = false;

        if (config_.server.run_mode == 3) {
            InitAgentSolver(agent, *session);
        }
    }

    // 初始化网格障碍物（仅 A* 测试模式需要，训练/推理模式不需要地图数据）
    if (config_.server.run_mode == 3) {
        session->InitBlocked(gs);
    }

    session->initialized = true;
    rsp->set_ret_code(0);

    LOG_FILE("RPC:Init", "S:%d InitRsp: ret_code=0", session_id);

    return grpc::Status::OK;
}

// ---- Update RPC：接收帧状态，返回动作 ----
grpc::Status MazeServiceImpl::Update(grpc::ServerContext* ctx,
                                     const maze::UpdateReq* req,
                                     maze::UpdateRsp* rsp) {
    int session_id = req->session_id();

    // 获取会话
    auto* session = session_mgr_.GetSession(session_id);
    if (!session) {
        LOG_WARN("MazeService", "Update: 会话 S:%d 不存在", session_id);
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "会话不存在");
    }

    if (!session->initialized) {
        LOG_WARN("MazeService", "Update: 会话 S:%d 尚未初始化", session_id);
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "未初始化");
    }

    int frame_id = req->frame_id();

    LOG_FILE("RPC:Update", "S:%d UpdateReq: frame_id=%d, agents_size=%d",
             session_id, frame_id, req->agents_size());

    for (int i = 0; i < req->agents_size(); ++i) {
        const auto& state = req->agents(i);
        int agent_id = state.agent_id();
        float pos_x = state.pos().x();
        float pos_y = state.pos().y();
        bool is_done = state.is_done();

        int cur_gx = static_cast<int>(pos_x);
        int cur_gy = static_cast<int>(pos_y);

        // 提取 Client 端传来的射线观测特征
        std::vector<float> client_obs(state.obs().begin(), state.obs().end());

        LOG_FILE("RPC:Update", "  S:%d Agent[%d]: pos=(%.1f,%.1f) grid=(%d,%d) done=%s",
                 session_id, agent_id, pos_x, pos_y, cur_gx, cur_gy,
                 is_done ? "true" : "false");

        auto* action = rsp->add_actions();
        action->set_agent_id(agent_id);

        // 已结束的 Agent：记录排名 + 收集终止帧样本后跳过动作下发
        if (is_done) {
            action->set_action_id(0);

            // 训练模式：记录排名并收集终止帧样本
            if (config_.server.run_mode == 1) {
                auto it = session->agents.find(agent_id);
                if (it != session->agents.end() && !it->second.done_collected) {
                    // 标记终止帧已收集（仅收集一次）
                    it->second.done_collected = true;
                    it->second.reached_goal = (cur_gx == session->end_gx && cur_gy == session->end_gy);

                    // 记录完成排名（先完成的排在前面）
                    session->ranking_order.push_back(agent_id);
                    if (session->first_done_frame < 0) {
                        session->first_done_frame = frame_id;
                    }
                    LOG_INFO("MazeService", "S:%d Agent[%d] 完成排名 #%d (goal=%s frame=%d)",
                             session_id, agent_id,
                             static_cast<int>(session->ranking_order.size()),
                             it->second.reached_goal ? "true" : "false", frame_id);

                    CollectSample(*session, agent_id, cur_gx, cur_gy,
                                  0, -2.197f, 0.0f, true, client_obs);
                    it->second.prev_grid_x = cur_gx;
                    it->second.prev_grid_y = cur_gy;
                    LOG_FILE("RPC:Update", "  S:%d Agent[%d]: 终止帧样本已收集 (done rank=#%d)",
                             session_id, agent_id, static_cast<int>(session->ranking_order.size()));
                }
            }

            LOG_FILE("RPC:Update", "  S:%d Agent[%d]: action=0 (done)", session_id, agent_id);
            continue;
        }

        // 查找 Agent 运行时数据
        auto it = session->agents.find(agent_id);
        if (it == session->agents.end()) {
            action->set_action_id(0);
            LOG_WARN("MazeService", "S:%d Agent[%d] 未找到运行时数据", session_id, agent_id);
            continue;
        }

        auto& agent = it->second;
        int chosen_action = 0;

        // ---- 根据运行模式选择动作 ----
        if (config_.server.run_mode == 3) {
            // A* 测试模式
            if (config_.strategy.replan_interval > 0 &&
                frame_id > 0 &&
                frame_id % config_.strategy.replan_interval == 0) {
                float world_x = (cur_gx + 0.5f) * config_.strategy.grid_size;
                float world_y = (cur_gy + 0.5f) * config_.strategy.grid_size;
                agent.path_valid = agent.solver.PlanPath(world_x, world_y,
                                                         session->end_x, session->end_y);
                if (!agent.path_valid) {
                    LOG_WARN("MazeService", "S:%d Agent[%d] 路径规划失败 frame=%d grid=(%d,%d)",
                             session_id, agent_id, frame_id, cur_gx, cur_gy);
                }
            }

            if (agent.path_valid) {
                chosen_action = agent.solver.GetAction(cur_gx, cur_gy);
            }
            LOG_FILE("RPC:Update", "  S:%d Agent[%d]: action=%d (astar) grid=(%d,%d)",
                     session_id, agent_id, chosen_action, cur_gx, cur_gy);
        } else if (config_.server.run_mode == 1 || config_.server.run_mode == 2) {
            // 训练/推理模式：优先 ONNX 推理，无模型时回退随机策略
            float old_log_prob = -2.197f;   // log(1/9) 随机策略默认值
            float old_vpred = 0.0f;

            if (onnx_inferencer_.IsLoaded()) {
                // ONNX 模型推理
                std::vector<float> obs;
                BuildObs(*session, cur_gx, cur_gy, client_obs, obs);

                std::vector<float> action_probs;
                float value = 0.0f;

                if (onnx_inferencer_.Infer(obs, static_cast<int>(obs.size()),
                                          action_probs, value)) {
                    chosen_action = SampleAction(action_probs);
                    old_log_prob = std::log(action_probs[chosen_action] + 1e-8f);
                    old_vpred = value;
                    LOG_FILE("RPC:Update", "  S:%d Agent[%d]: action=%d (onnx) prob=%.3f value=%.3f grid=(%d,%d)",
                             session_id, agent_id, chosen_action,
                             action_probs[chosen_action], value, cur_gx, cur_gy);
                } else {
                    // 推理失败，回退随机策略
                    chosen_action = ChooseRandomAction();
                    LOG_FILE("RPC:Update", "  S:%d Agent[%d]: action=%d (random-fallback) grid=(%d,%d)",
                             session_id, agent_id, chosen_action, cur_gx, cur_gy);
                }
            } else {
                // 无模型，随机策略
                chosen_action = ChooseRandomAction();
                LOG_FILE("RPC:Update", "  S:%d Agent[%d]: action=%d (random-no-model) grid=(%d,%d)",
                         session_id, agent_id, chosen_action, cur_gx, cur_gy);
            }

            action->set_action_id(chosen_action);
            agent.last_action = chosen_action;

            // 训练模式：收集样本
            if (config_.server.run_mode == 1) {
                CollectSample(*session, agent_id, cur_gx, cur_gy,
                              chosen_action, old_log_prob, old_vpred, is_done, client_obs);
                agent.prev_grid_x = cur_gx;
                agent.prev_grid_y = cur_gy;
            }
            continue;   // 跳过下方的通用 action 设置
        }

        action->set_action_id(chosen_action);
        agent.last_action = chosen_action;
    }

    return grpc::Status::OK;
}

// ---- EndEpisode RPC：Episode 结束处理 ----
grpc::Status MazeServiceImpl::EndEpisode(grpc::ServerContext* ctx,
                                         const maze::EpisodeEndReq* req,
                                         maze::EpisodeEndRsp* rsp) {
    int session_id = req->session_id();
    int ep_id = req->episode_id();

    // 获取会话
    auto* session = session_mgr_.GetSession(session_id);
    if (!session) {
        LOG_WARN("MazeService", "EndEpisode: 会话 S:%d 不存在", session_id);
        rsp->set_ret_code(-1);
        return grpc::Status::OK;
    }

    session->current_episode_id = ep_id;

    LOG_INFO("MazeService", "S:%d Episode %d 结束", session_id, ep_id);
    LOG_FILE("RPC:EndEpisode", "S:%d EpisodeEndReq: episode_id=%d", session_id, ep_id);

    // 训练模式：遍历所有 Agent，发送各自剩余样本（标记 is_episode_end=true）
    if (config_.server.run_mode == 1) {
        for (auto& [aid, agent] : session->agents) {
            auto& cache = session->agent_sample_caches[aid];
            if (!cache.empty()) {
                LOG_INFO("MazeService", "S:%d Episode %d Agent[%d] 结束，发送剩余 %zu 个样本",
                         session_id, ep_id, aid, cache.size());
            }
            // 即使缓存为空也发送 is_episode_end=true 信号，确保 Learner 端 trajectory 完成
            FlushAgentSamples(*session, aid, true);
        }
    }

    // 重置 Agent 状态
    if (config_.server.run_mode == 3) {
        for (auto& [id, agent] : session->agents) {
            agent.path_valid = agent.solver.PlanPath(session->start_x, session->start_y,
                                                     session->end_x, session->end_y);
        }
    }

    // 重置训练状态
    for (auto& [id, agent] : session->agents) {
        agent.prev_grid_x = -1;
        agent.prev_grid_y = -1;
        agent.reached_goal = false;
        agent.done_collected = false;
        agent.visited.clear();
        agent.recent_positions.clear();
    }

    // 重置竞争排名状态
    session->first_done_frame = -1;
    session->ranking_order.clear();

    rsp->set_ret_code(0);
    LOG_FILE("RPC:EndEpisode", "S:%d EpisodeEndRsp: ret_code=0", session_id);

    return grpc::Status::OK;
}
