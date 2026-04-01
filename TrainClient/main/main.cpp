#include "grpc_client.h"
#include "maze_env.h"
#include "maze.pb.h"
#include "config_loader.h"
#include "viz_recorder.h"
#include "logger.h"

#include <cmath>
#include <string>
#include <sstream>
#include <vector>

// --- 默认配置文件路径 ---
static const char* kDefaultConfigPath = "configs/client_config.yaml";

// ---- 动作名称表（用于日志输出）----
static const char* kActionNames[9] = {
    "不动", "上", "右上", "右", "右下", "下", "左下", "左", "左上"
};

// ---- 构建可视化 JSON 数据 ----
static std::string BuildVizJson(const MazeEnv& env, const ClientConfig& cfg,
                                int frame_id, int episode_id,
                                const std::vector<int>& actions) {
    std::ostringstream ss;
    ss << "{";

    // 帧信息
    ss << "\"type\":\"frame_update\",";
    ss << "\"frame_id\":" << frame_id << ",";
    ss << "\"episode_id\":" << episode_id << ",";

    // 地图信息
    ss << "\"map\":{";
    ss << "\"width\":" << cfg.env.map_width << ",";
    ss << "\"height\":" << cfg.env.map_height << ",";

    // 起点
    ss << "\"start_pos\":{\"x\":" << cfg.env.start_x << ",\"y\":" << cfg.env.start_y << "},";

    // 终点
    ss << "\"end_pos\":{\"x\":" << cfg.env.end_x << ",\"y\":" << cfg.env.end_y << "},";

    // 墙壁（内部隔墙，不含外围边界）
    ss << "\"walls\":[";
    ss << "{\"x1\":5000,\"y1\":0,\"x2\":5000,\"y2\":14000,\"thickness\":100},";
    ss << "{\"x1\":10000,\"y1\":6000,\"x2\":10000,\"y2\":20000,\"thickness\":100},";
    ss << "{\"x1\":15000,\"y1\":0,\"x2\":15000,\"y2\":14000,\"thickness\":100}";
    ss << "]},";

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

int main(int argc, char* argv[]) {
    std::printf("============================================\n");
    std::printf("  迷宫训练框架 - TrainClient (Demo)\n");
    std::printf("============================================\n\n");

    // ---- 0. 加载配置 ----
    const char* config_path = (argc > 1) ? argv[1] : kDefaultConfigPath;
    ClientConfig cfg;
    LoadClientConfig(config_path, cfg);

    // ---- 0a. 初始化日志系统 ----
    Logger::Instance().Init("log");
    Logger::Instance().SetConsoleLevel(LogLevel::INFO);
    Logger::Instance().SetFileLevel(LogLevel::DEBUG);

    // ---- 1. 初始化环境 ----
    MazeEnv env;
    env.Init(cfg);

    // ---- 2. 初始化帧数据记录器 ----
    VizRecorder viz_recorder;

    // ---- 3. 建立 gRPC 连接 ----
    GrpcClient client;
    if (!client.Connect(cfg.network.server_host, cfg.network.server_port)) {
        LOG_ERROR("Main", "无法连接 AIServer %s:%d，退出",
                  cfg.network.server_host.c_str(), cfg.network.server_port);
        Logger::Instance().Close();
        return 1;
    }

    // ---- 4. 发送 InitReq ----
    {
        maze::InitReq req;
        req.set_agent_num(cfg.run.agent_num);

        auto* map_size = req.mutable_map_size();
        map_size->set_x(cfg.env.map_width);
        map_size->set_y(cfg.env.map_height);

        auto* start_pos = req.mutable_start_pos();
        start_pos->set_x(cfg.env.start_x);
        start_pos->set_y(cfg.env.start_y);

        auto* end_pos = req.mutable_end_pos();
        end_pos->set_x(cfg.env.end_x);
        end_pos->set_y(cfg.env.end_y);

        maze::InitRsp rsp;
        if (!client.Init(req, rsp)) {
            LOG_ERROR("Main", "Init RPC 失败");
            Logger::Instance().Close();
            return 1;
        }
        LOG_INFO("Main", "初始化成功: ret_code=%d", rsp.ret_code());

        if (rsp.ret_code() != 0) {
            LOG_ERROR("Main", "初始化失败，退出");
            Logger::Instance().Close();
            return 1;
        }
    }

    // ---- 5. Episode 循环 ----
    for (int ep = 0; ep < cfg.run.max_episodes; ++ep) {
        env.Reset();
        LOG_INFO("Main", "===== Episode %d 开始 =====", ep);

        // 开始帧数据记录
        if (cfg.viz.enabled) {
            viz_recorder.Begin(cfg.viz.output_dir, ep);
        }

        // 每帧记录各 Agent 的动作（用于可视化）
        std::vector<int> last_actions(cfg.run.agent_num, 0);

        // ---- 帧循环 ----
        while (!env.AllDone()) {
            // ---- 5a. 构建 UpdateReq（上报网格坐标，通过 Vec2 float 传输）----
            maze::UpdateReq update_req;
            update_req.set_frame_id(env.GetFrameId());

            for (int i = 0; i < env.GetAgentNum(); ++i) {
                const AgentInfo& info = env.GetAgent(i);
                auto* agent_state = update_req.add_agents();
                agent_state->set_agent_id(info.id);
                auto* pos = agent_state->mutable_pos();
                pos->set_x(static_cast<float>(info.grid_x));
                pos->set_y(static_cast<float>(info.grid_y));
                agent_state->set_is_done(info.done);
            }

            // ---- 5b. 发送 UpdateReq，接收 UpdateRsp ----
            maze::UpdateRsp update_rsp;
            if (!client.Update(update_req, update_rsp)) {
                LOG_ERROR("Main", "Update RPC 失败，退出");
                viz_recorder.End();
                Logger::Instance().Close();
                return 1;
            }

            // ---- 5c. 执行动作 ----
            for (int i = 0; i < update_rsp.actions_size(); ++i) {
                const auto& action = update_rsp.actions(i);
                int aid = action.agent_id();
                int act = action.action_id();

                // 记录执行前位置
                const AgentInfo& before = env.GetAgent(aid);
                int prev_gx = before.grid_x;
                int prev_gy = before.grid_y;

                env.Step(aid, act);

                // 记录执行后位置
                const AgentInfo& after = env.GetAgent(aid);
                const char* act_name = (act >= 0 && act < 9) ? kActionNames[act] : "未知";

                // 帧级详细日志（仅写文件）
                LOG_FILE("Frame", "EP:%d F:%d A:%d | 指令:%d(%s) | (%d,%d)->(%d,%d) %s",
                         ep, env.GetFrameId(), aid, act, act_name,
                         prev_gx, prev_gy, after.grid_x, after.grid_y,
                         (prev_gx == after.grid_x && prev_gy == after.grid_y) ? "[未移动]" : "[已移动]");

                // 记录动作用于可视化
                if (aid >= 0 && aid < static_cast<int>(last_actions.size())) {
                    last_actions[aid] = act;
                }
            }

            // ---- 5d. 帧号递增 ----
            env.AdvanceFrame();

            // ---- 5e. 记录帧数据（切片记录，用于离线回放）----
            if (cfg.viz.enabled) {
                if (env.GetFrameId() % cfg.viz.interval == 0) {
                    std::string json = BuildVizJson(env, cfg, env.GetFrameId(), ep, last_actions);
                    viz_recorder.RecordFrame(json);
                }
            }

            // ---- 5f. 定期日志：打印接收到的指令和执行效果 ----
            if (env.GetFrameId() % cfg.run.log_interval == 0) {
                const AgentInfo& a = env.GetAgent(0);
                int act = last_actions.empty() ? 0 : last_actions[0];
                const char* act_name = (act >= 0 && act < 9) ? kActionNames[act] : "未知";
                LOG_INFO("Exec", "EP:%d F:%d | 指令:action=%d(%s) | 网格:(%d,%d) 终点:(%d,%d)",
                         ep, env.GetFrameId(), act, act_name,
                         a.grid_x, a.grid_y, env.ToGridX(cfg.env.end_x), env.ToGridY(cfg.env.end_y));
            }
        }

        // ---- 6. Episode 结束 ----
        {
            // 结束帧数据记录
            viz_recorder.End();

            // 统计结果
            const AgentInfo& a = env.GetAgent(0);
            bool passed = (a.grid_x == env.ToGridX(cfg.env.end_x) &&
                           a.grid_y == env.ToGridY(cfg.env.end_y));
            int grid_dist = std::abs(a.grid_x - env.ToGridX(cfg.env.end_x)) +
                            std::abs(a.grid_y - env.ToGridY(cfg.env.end_y));

            LOG_INFO("Result", "EP:%d 结束 | 帧数:%d | %s | 网格距离:%d",
                     ep, env.GetFrameId(), passed ? "通关" : "超时", grid_dist);

            // 发送 EpisodeEndReq
            maze::EpisodeEndReq ep_end_req;
            ep_end_req.set_episode_id(ep);

            maze::EpisodeEndRsp ep_end_rsp;
            if (!client.EndEpisode(ep_end_req, ep_end_rsp)) {
                LOG_ERROR("Main", "EndEpisode RPC 失败");
                break;
            }
        }
    }

    // ---- 7. 结束 ----
    LOG_INFO("Main", "训练结束");
    Logger::Instance().Close();

    return 0;
}
