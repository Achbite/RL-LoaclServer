#pragma once

#include "ai/astar_solver.h"
#include "maze.pb.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>
#include <cmath>
#include <algorithm>

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

        // --- 奖励辅助状态 ---
        std::unordered_set<int> visited;        // 已访问网格集合（key = gy * cols + gx），用于探索奖励
        std::deque<int>         recent_positions; // 最近 N 步位置历史（滑动窗口），用于徘徊惩罚
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

        // --- 网格障碍物（true=不可通行，用于射线检测）---
        std::vector<bool> blocked;

        // 初始化网格障碍物（Init RPC 时调用）
        void InitBlocked(int grid_size) {
            blocked.assign(grid_cols * grid_rows, false);
            // 内部隔墙（与 TrainClient 端 LoadWalls 一致）
            AddWallToGrid(5000, 0, 5000, 14000, 100, grid_size);
            AddWallToGrid(10000, 6000, 10000, 20000, 100, grid_size);
            AddWallToGrid(15000, 0, 15000, 14000, 100, grid_size);
            // 确保起点和终点可通行
            int start_gx = static_cast<int>(start_x / grid_size);
            int start_gy = static_cast<int>(start_y / grid_size);
            blocked[start_gy * grid_cols + start_gx] = false;
            blocked[end_gy * grid_cols + end_gx] = false;
        }

        // 网格是否可通行（越界视为不可通行）
        bool IsWalkable(int gx, int gy) const {
            if (gx < 0 || gx >= grid_cols || gy < 0 || gy >= grid_rows) return false;
            return !blocked[gy * grid_cols + gx];
        }

        // 竞争排名机制
        int first_done_frame = -1;                  // 首个 Agent 完成时的帧号（-1 表示尚无 Agent 完成）
        std::vector<int> ranking_order;              // Agent 完成排名顺序（先完成的在前）

    private:
        // 添加单面墙壁到网格（AABB 包围盒映射）
        void AddWallToGrid(float x1, float y1, float x2, float y2,
                           float thickness, int grid_size) {
            float half_t = thickness * 0.5f;
            float min_x = std::min(x1, x2) - half_t;
            float max_x = std::max(x1, x2) + half_t;
            float min_y = std::min(y1, y2) - half_t;
            float max_y = std::max(y1, y2) + half_t;

            int gx_min = std::max(0, static_cast<int>(std::floor(min_x / grid_size)));
            int gx_max = std::min(grid_cols - 1, static_cast<int>(std::floor(max_x / grid_size)));
            int gy_min = std::max(0, static_cast<int>(std::floor(min_y / grid_size)));
            int gy_max = std::min(grid_rows - 1, static_cast<int>(std::floor(max_y / grid_size)));

            for (int gy = gy_min; gy <= gy_max; ++gy) {
                for (int gx = gx_min; gx <= gx_max; ++gx) {
                    blocked[gy * grid_cols + gx] = true;
                }
            }
        }
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
