#pragma once

#include "ai/astar_solver.h"
#include "maze.pb.h"

#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>

// ---- 会话管理器（并行 Episode 隔离）----
// 每个 session 维护独立的 Agent 运行时状态和样本缓存，
// 不同 session 之间互不干扰，支持多 Episode 并行采集。

class SessionManager {
public:
    // ---- Agent 运行时状态（每个 session 内独立）----
    struct AgentRuntime {
        AStarSolver solver;             // 独立寻路器
        int         last_action = 0;    // 上一帧动作
        bool        path_valid  = false;

        // 训练模式：帧样本缓存
        int   prev_grid_x  = -1;       // 上一帧网格坐标（用于计算距离变化）
        int   prev_grid_y  = -1;
        bool  reached_goal = false;     // 本 Episode 是否到达终点
        bool  done_collected = false;   // 终止帧样本是否已收集（防止重复收集）
    };

    // ---- 单个会话 ----
    struct Session {
        int session_id = 0;
        std::unordered_map<int, AgentRuntime> agents;   // agent_id → 运行时状态
        int current_episode_id = 0;                     // 当前 Episode ID
        std::unordered_map<int, std::vector<maze::Sample>> agent_sample_caches;  // agent_id → 样本缓存（多 Agent 隔离）

        // 地图参数（每个 session 独立，支持不同地图配置）
        float map_width  = 0.0f;
        float map_height = 0.0f;
        float start_x    = 0.0f;
        float start_y    = 0.0f;
        float end_x      = 0.0f;
        float end_y      = 0.0f;

        // 网格参数（Init 时计算）
        int end_gx    = 0;              // 终点网格坐标
        int end_gy    = 0;
        int grid_cols = 0;              // 网格列数
        int grid_rows = 0;              // 网格行数

        bool initialized = false;       // 是否已初始化

        // 竞争排名机制
        int first_done_frame = -1;                  // 首个 Agent 完成时的帧号（-1 表示尚无 Agent 完成）
        std::vector<int> ranking_order;              // Agent 完成排名顺序（先完成的在前）
    };

    SessionManager() = default;
    ~SessionManager() = default;

    // 创建新会话，返回分配的 session_id
    int CreateSession();

    // 获取指定会话（不存在则返回 nullptr）
    Session* GetSession(int session_id);

    // 获取或创建会话（session_id=0 时使用默认会话）
    Session* GetOrCreateSession(int session_id);

    // 销毁指定会话
    void DestroySession(int session_id);

    // 获取当前活跃会话数
    int GetActiveSessionCount() const;

private:
    std::unordered_map<int, Session> sessions_;
    mutable std::mutex mutex_;
    int next_session_id_ = 1;           // 从 1 开始分配，0 保留给默认会话
};
