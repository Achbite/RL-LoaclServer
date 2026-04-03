#include "grpc/grpc_client.h"
#include "env/maze_env.h"
#include "maze.pb.h"
#include "config/config_loader.h"
#include "pool/thread_pool.h"
#include "pool/episode_pool.h"
#include "viz/viz_recorder.h"
#include "log/logger.h"

#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

// --- 默认配置文件路径 ---
static const char* kDefaultConfigPath = "configs/client_config.yaml";

// --- 全局统计（线程安全）---
static std::atomic<int> g_completed_episodes{0};   // 已完成 Episode 数
static std::atomic<int> g_passed_episodes{0};       // 通关 Episode 数
static std::atomic<int> g_total_frames{0};          // 总帧数
static std::mutex       g_log_mutex;                // 日志互斥锁

// ---- 构建可视化 JSON 数据 ----
// 帧数据不再存储任何地图信息，只存 map_id 引用，地图文件由播放器独立加载
static std::string BuildVizJson(const MazeEnv& env,
                                int frame_id, int episode_id,
                                const std::vector<int>& actions) {
    std::ostringstream ss;
    ss << "{";

    // 帧信息
    ss << "\"type\":\"frame_update\",";
    ss << "\"frame_id\":" << frame_id << ",";
    ss << "\"episode_id\":" << episode_id << ",";

    // 地图引用（仅存 map_id，播放器通过 /api/map 接口加载完整地图）
    ss << "\"map_id\":\"" << env.GetMapId() << "\",";

    // Agent 列表（网格坐标转换为连续坐标用于可视化）
    ss << "\"agents\":[";
    for (int i = 0; i < env.GetAgentNum(); ++i) {
        const AgentInfo& a = env.GetAgent(i);
        float world_x = env.GetWorldX(a.grid_x);
        float world_y = env.GetWorldY(a.grid_y);
        if (i > 0) ss << ",";
        ss << "{\"id\":" << a.id
           << ",\"x\":" << world_x
           << ",\"y\":" << world_y
           << ",\"done\":" << (a.done ? "true" : "false")
           << ",\"action_id\":" << (i < static_cast<int>(actions.size()) ? actions[i] : 0)
           << "}";
    }
    ss << "]";

    ss << "}";
    return ss.str();
}

// ---- 运行单个 Episode（在线程池中执行）----
static void RunEpisode(EpisodeWorker* worker, int episode_id,
                       const ClientConfig& cfg) {
    MazeEnv& env = worker->env;
    GrpcClient& client = worker->client;
    int session_id = worker->session_id;

    env.Reset();

    LOG_FILE("Train", "W:%d S:%d EP:%d 开始", worker->worker_id, session_id, episode_id);

    // ---- 初始化帧数据记录器（每个 Episode 独立实例，线程安全）----
    VizRecorder viz_recorder;
    if (cfg.viz.enabled) {
        viz_recorder.Begin(cfg.viz.output_dir, episode_id,
                           env.GetMapId(), env.GetMapFilePath());
    }

    // 每帧记录各 Agent 的动作（用于可视化）
    std::vector<int> last_actions(cfg.run.agent_num, 0);

    // ---- 发送 InitReq（每个 Episode 重新初始化会话）----
    {
        maze::InitReq req;
        req.set_agent_num(cfg.run.agent_num);
        req.set_session_id(session_id);

        auto* map_size = req.mutable_map_size();
        map_size->set_x(env.GetMapWidth());
        map_size->set_y(env.GetMapHeight());

        auto* start_pos = req.mutable_start_pos();
        start_pos->set_x(env.GetStartX());
        start_pos->set_y(env.GetStartY());

        auto* end_pos = req.mutable_end_pos();
        end_pos->set_x(env.GetEndX());
        end_pos->set_y(env.GetEndY());

        maze::InitRsp rsp;
        if (!client.Init(req, rsp) || rsp.ret_code() != 0) {
            LOG_ERROR("Train", "W:%d S:%d EP:%d Init 失败", worker->worker_id, session_id, episode_id);
            return;
        }
    }

    // ---- 帧循环 ----
    while (!env.AllDone()) {
        // 构建 UpdateReq
        maze::UpdateReq update_req;
        update_req.set_frame_id(env.GetFrameId());
        update_req.set_session_id(session_id);

        for (int i = 0; i < env.GetAgentNum(); ++i) {
            const AgentInfo& info = env.GetAgent(i);
            auto* agent_state = update_req.add_agents();
            agent_state->set_agent_id(info.id);
            auto* pos = agent_state->mutable_pos();
            pos->set_x(static_cast<float>(info.grid_x));
            pos->set_y(static_cast<float>(info.grid_y));
            agent_state->set_is_done(info.done);

            // 填充 8 方向射线观测特征（Client 端计算，传给 AIServer）
            if (!info.done) {
                RayResult rays = env.CastRays(info.grid_x, info.grid_y);
                for (int d = 0; d < 8; ++d) {
                    agent_state->add_obs(rays.distances[d]);
                }
            }
        }

        // 发送 UpdateReq
        maze::UpdateRsp update_rsp;
        if (!client.Update(update_req, update_rsp)) {
            LOG_ERROR("Train", "W:%d S:%d EP:%d Update RPC 失败", worker->worker_id, session_id, episode_id);
            return;
        }

        // 执行动作
        for (int i = 0; i < update_rsp.actions_size(); ++i) {
            const auto& action = update_rsp.actions(i);
            int aid = action.agent_id();
            int act = action.action_id();
            env.Step(aid, act);

            // 记录动作用于可视化
            if (aid >= 0 && aid < static_cast<int>(last_actions.size())) {
                last_actions[aid] = act;
            }
        }

        env.AdvanceFrame();

        // ---- 记录帧数据（切片记录，用于离线回放）----
        if (cfg.viz.enabled && env.GetFrameId() % cfg.viz.interval == 0) {
            std::string json = BuildVizJson(env, env.GetFrameId(), episode_id, last_actions);
            viz_recorder.RecordFrame(json);
        }
    }

    // ---- Episode 结束：多 Agent 通关统计 ----
    viz_recorder.End();

    int frames = env.GetFrameId();
    int agent_passed_count = 0;
    for (int i = 0; i < env.GetAgentNum(); ++i) {
        const AgentInfo& a = env.GetAgent(i);
        if (a.grid_x == env.ToGridX(env.GetEndX()) &&
            a.grid_y == env.ToGridY(env.GetEndY())) {
            agent_passed_count++;
        }
    }
    bool passed = (agent_passed_count > 0);  // 任一 Agent 通关即计为通关

    // 发送 EpisodeEndReq
    {
        maze::EpisodeEndReq ep_end_req;
        ep_end_req.set_episode_id(episode_id);
        ep_end_req.set_session_id(session_id);

        maze::EpisodeEndRsp ep_end_rsp;
        if (!client.EndEpisode(ep_end_req, ep_end_rsp)) {
            LOG_WARN("Train", "W:%d S:%d EP:%d EndEpisode RPC 失败", worker->worker_id, session_id, episode_id);
        }
    }

    // 更新全局统计
    g_completed_episodes.fetch_add(1);
    g_total_frames.fetch_add(frames);
    if (passed) {
        g_passed_episodes.fetch_add(1);
    }

    LOG_FILE("Train", "W:%d S:%d EP:%d 结束 | 帧数:%d | 通关Agent:%d/%d | %s",
             worker->worker_id, session_id, episode_id, frames,
             agent_passed_count, env.GetAgentNum(),
             passed ? "通关" : "超时");
}

int main(int argc, char* argv[]) {
    std::printf("============================================\n");
    std::printf("  迷宫训练框架 - TrainClient 并行训练\n");
    std::printf("============================================\n\n");

    // ---- 0. 加载配置 ----
    const char* config_path = (argc > 1) ? argv[1] : kDefaultConfigPath;
    ClientConfig cfg;
    LoadClientConfig(config_path, cfg);

    // ---- 0a. 初始化日志系统 ----
    Logger::Instance().Init("log");
    Logger::Instance().SetConsoleLevel(LogLevel::INFO);
    Logger::Instance().SetFileLevel(LogLevel::DEBUG);

    int thread_count = cfg.train.thread_count;
    int pool_size    = cfg.train.episode_pool_size;
    int max_episodes = cfg.train.max_episodes;
    int summary_interval = cfg.train.log_summary_interval;

    LOG_INFO("Train", "并行训练配置: threads=%d, pool=%d, max_episodes=%d",
             thread_count, pool_size, max_episodes);

    // ---- 1. 初始化 Episode 对象池 ----
    EpisodePool episode_pool;
    int ready_count = episode_pool.Init(pool_size, cfg);
    if (ready_count == 0) {
        LOG_ERROR("Train", "没有可用的 Worker，退出");
        Logger::Instance().Close();
        return 1;
    }

    LOG_INFO("Train", "对象池就绪: %d/%d 个 Worker", ready_count, pool_size);

    // ---- 2. 创建线程池 + 提交任务（作用域块控制线程池生命周期）----
    auto start_time = std::chrono::steady_clock::now();
    {
        ThreadPool thread_pool(thread_count);
        LOG_INFO("Train", "线程池就绪: %d 个工作线程", thread_count);

        // ---- 3. 提交所有 Episode 任务 ----
        std::vector<std::future<void>> futures;
        futures.reserve(max_episodes);

        for (int ep = 0; ep < max_episodes; ++ep) {
            futures.push_back(thread_pool.Enqueue([&episode_pool, ep, &cfg, summary_interval]() {
                // 借出 Worker
                EpisodeWorker* worker = episode_pool.Acquire();

                // 运行 Episode
                RunEpisode(worker, ep, cfg);

                // 归还 Worker
                episode_pool.Release(worker->worker_id);

                // 定期打印汇总
                int completed = g_completed_episodes.load();
                if (completed % summary_interval == 0) {
                    int passed = g_passed_episodes.load();
                    int total_frames = g_total_frames.load();
                    float pass_rate = (completed > 0) ? (100.0f * passed / completed) : 0.0f;
                    float avg_frames = (completed > 0) ? (static_cast<float>(total_frames) / completed) : 0.0f;
                    LOG_INFO("Train", "进度: %d/%d | 通关率: %.1f%% | 平均帧数: %.0f",
                             completed, cfg.train.max_episodes, pass_rate, avg_frames);
                }
            }));
        }

        // ---- 4. 等待所有任务完成 ----
        LOG_INFO("Train", "等待 %d 个 Episode 完成...", max_episodes);
        for (auto& f : futures) {
            f.get();
        }
    }
    // 作用域结束：线程池析构，所有工作线程 join 完毕

    // ---- 4a. 关闭对象池（断开所有 gRPC 连接）----
    episode_pool.Shutdown();

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // ---- 5. 打印最终汇总 ----
    int completed = g_completed_episodes.load();
    int passed = g_passed_episodes.load();
    int total_frames = g_total_frames.load();
    float pass_rate = (completed > 0) ? (100.0f * passed / completed) : 0.0f;
    float avg_frames = (completed > 0) ? (static_cast<float>(total_frames) / completed) : 0.0f;

    std::printf("\n============================================\n");
    std::printf("  训练完成\n");
    std::printf("============================================\n");
    LOG_INFO("Train", "总 Episode: %d | 通关: %d | 通关率: %.1f%%",
             completed, passed, pass_rate);
    LOG_INFO("Train", "总帧数: %d | 平均帧数: %.0f", total_frames, avg_frames);
    LOG_INFO("Train", "耗时: %.1f 秒 | 吞吐: %.1f Episode/秒",
             elapsed, completed / elapsed);

    Logger::Instance().Close();
    return 0;
}
