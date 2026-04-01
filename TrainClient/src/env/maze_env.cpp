#include "env/maze_env.h"
#include "config/config_loader.h"
#include "log/logger.h"

#include <cmath>
#include <algorithm>

// ---- 从完整配置初始化环境 ----
void MazeEnv::Init(const ClientConfig& config) {
    const EnvConfig& env = config.env;

    // 加载地图参数
    map_width_     = env.map_width;
    map_height_    = env.map_height;
    max_steps_     = env.max_steps;
    start_x_       = env.start_x;
    start_y_       = env.start_y;
    end_x_         = env.end_x;
    end_y_         = env.end_y;
    grid_size_     = env.grid_size;

    // 计算网格尺寸
    grid_cols_ = static_cast<int>(std::ceil(map_width_ / grid_size_));
    grid_rows_ = static_cast<int>(std::ceil(map_height_ / grid_size_));

    // 计算起点/终点的网格坐标
    start_gx_ = ToGridX(start_x_);
    start_gy_ = ToGridY(start_y_);
    end_gx_   = ToGridX(end_x_);
    end_gy_   = ToGridY(end_y_);

    // 初始化网格障碍物
    blocked_.assign(grid_cols_ * grid_rows_, false);
    LoadWalls();

    // 初始化 Agent
    int agent_num = config.run.agent_num;
    agents_.resize(agent_num);
    for (int i = 0; i < agent_num; ++i) {
        agents_[i].id     = i;
        agents_[i].grid_x = start_gx_;
        agents_[i].grid_y = start_gy_;
        agents_[i].done   = false;
    }

    frame_id_ = 0;

    LOG_INFO("MazeEnv", "初始化完成: agent_num=%d, grid=%dx%d (格子=%dcm), "
                "start_grid=(%d,%d), end_grid=(%d,%d), max_steps=%d",
                agent_num, grid_cols_, grid_rows_, grid_size_,
                start_gx_, start_gy_, end_gx_, end_gy_, max_steps_);
}

// ---- 重置所有 Agent 到起点 ----
void MazeEnv::Reset() {
    for (auto& agent : agents_) {
        agent.grid_x = start_gx_;
        agent.grid_y = start_gy_;
        agent.done   = false;
    }
    frame_id_ = 0;
    first_done_frame_ = -1;
}

// ---- 执行网格级移动 ----
void MazeEnv::Step(int agent_id, int action_id) {
    if (agent_id < 0 || agent_id >= static_cast<int>(agents_.size())) {
        return;
    }

    AgentInfo& agent = agents_[agent_id];

    // 已结束的 Agent 不再移动
    if (agent.done) {
        return;
    }

    // 动作范围校验
    if (action_id < 0 || action_id > 8) {
        action_id = 0;
    }

    // ---- 1. 计算目标网格 ----
    int dx = kGridActionDirs[action_id][0];
    int dy = kGridActionDirs[action_id][1];
    int new_gx = agent.grid_x + dx;
    int new_gy = agent.grid_y + dy;

    // ---- 2. 可达性检查 ----
    bool can_move = true;

    // 越界检查
    if (new_gx < 0 || new_gx >= grid_cols_ || new_gy < 0 || new_gy >= grid_rows_) {
        can_move = false;
    }

    // 目标格子是否有墙壁
    if (can_move && !IsWalkable(new_gx, new_gy)) {
        can_move = false;
    }

    // 对角线移动时，检查两个相邻格子是否可通行（防止穿墙角）
    if (can_move && dx != 0 && dy != 0) {
        if (!IsWalkable(agent.grid_x + dx, agent.grid_y) ||
            !IsWalkable(agent.grid_x, agent.grid_y + dy)) {
            can_move = false;
        }
    }

    // ---- 3. 执行移动 ----
    if (can_move) {
        agent.grid_x = new_gx;
        agent.grid_y = new_gy;
    }
    // 不可达则保持原位，AIServer 下一帧会根据新状态重新决策

    // ---- 4. 终止判定 ----
    if (CheckGoalReached(agent)) {
        agent.done = true;
        // 记录首个 Agent 完成的帧号（启动倒计时）
        if (first_done_frame_ < 0) {
            first_done_frame_ = frame_id_;
            LOG_INFO("MazeEnv", "Agent %d 首个通关! frame=%d 启动%d帧倒计时",
                        agent_id, frame_id_, kCountdownFrames);
        }
        LOG_INFO("MazeEnv", "Agent %d 到达终点! frame=%d grid=(%d,%d)",
                    agent_id, frame_id_, agent.grid_x, agent.grid_y);
    } else if (CheckTimeout()) {
        agent.done = true;
        LOG_INFO("MazeEnv", "Agent %d 超时! frame=%d grid=(%d,%d)",
                    agent_id, frame_id_, agent.grid_x, agent.grid_y);
    } else if (CheckCountdownExpired()) {
        // 倒计时到期，强制结束未完成的 Agent
        agent.done = true;
        LOG_INFO("MazeEnv", "Agent %d 倒计时结束! frame=%d grid=(%d,%d)",
                    agent_id, frame_id_, agent.grid_x, agent.grid_y);
    }
}

// ---- 帧号递增 ----
void MazeEnv::AdvanceFrame() {
    ++frame_id_;
}

// ---- 获取 Agent 状态 ----
const AgentInfo& MazeEnv::GetAgent(int agent_id) const {
    return agents_[agent_id];
}

// ---- 当前帧号 ----
int MazeEnv::GetFrameId() const {
    return frame_id_;
}

// ---- 所有 Agent 是否都已结束 ----
bool MazeEnv::AllDone() const {
    for (const auto& agent : agents_) {
        if (!agent.done) return false;
    }
    return true;
}

// ---- 是否有任一 Agent 已结束 ----
bool MazeEnv::HasAnyDone() const {
    for (const auto& agent : agents_) {
        if (agent.done) return true;
    }
    return false;
}

// ---- 获取首个 Agent 完成时的帧号 ----
int MazeEnv::GetFirstDoneFrame() const {
    return first_done_frame_;
}

// ---- Agent 数量 ----
int MazeEnv::GetAgentNum() const {
    return static_cast<int>(agents_.size());
}

// ---- 连续坐标 → 网格 X ----
int MazeEnv::ToGridX(float x) const {
    int gx = static_cast<int>(std::floor(x / grid_size_));
    return std::max(0, std::min(grid_cols_ - 1, gx));
}

// ---- 连续坐标 → 网格 Y ----
int MazeEnv::ToGridY(float y) const {
    int gy = static_cast<int>(std::floor(y / grid_size_));
    return std::max(0, std::min(grid_rows_ - 1, gy));
}

// ---- 网格是否可通行 ----
bool MazeEnv::IsWalkable(int gx, int gy) const {
    if (gx < 0 || gx >= grid_cols_ || gy < 0 || gy >= grid_rows_) {
        return false;
    }
    return !blocked_[gy * grid_cols_ + gx];
}

// ---- 是否到达终点网格 ----
bool MazeEnv::CheckGoalReached(const AgentInfo& agent) const {
    return agent.grid_x == end_gx_ && agent.grid_y == end_gy_;
}

// ---- 是否超时 ----
bool MazeEnv::CheckTimeout() const {
    return frame_id_ >= max_steps_;
}

// ---- 是否倒计时到期（首个 Agent 通关后 N 帧）----
bool MazeEnv::CheckCountdownExpired() const {
    if (first_done_frame_ < 0) return false;
    return (frame_id_ - first_done_frame_) >= kCountdownFrames;
}

// ---- 加载墙壁到网格 ----
void MazeEnv::LoadWalls() {
    // 内部隔墙（与 test_maze.json 一致，不添加外围边界墙壁，网格越界检查天然阻止越界）
    AddWallToGrid(5000, 0, 5000, 14000, 100);
    AddWallToGrid(10000, 6000, 10000, 20000, 100);
    AddWallToGrid(15000, 0, 15000, 14000, 100);

    // 确保起点和终点可通行
    blocked_[start_gy_ * grid_cols_ + start_gx_] = false;
    blocked_[end_gy_ * grid_cols_ + end_gx_]     = false;

    LOG_INFO("MazeEnv", "墙壁加载完成: 3 面内部隔墙");
}

// ---- 添加单面墙壁到网格 ----
void MazeEnv::AddWallToGrid(float x1, float y1, float x2, float y2, float thickness) {
    float half_t = thickness * 0.5f;
    float min_x = std::min(x1, x2) - half_t;
    float max_x = std::max(x1, x2) + half_t;
    float min_y = std::min(y1, y2) - half_t;
    float max_y = std::max(y1, y2) + half_t;

    int gx_min = std::max(0, static_cast<int>(std::floor(min_x / grid_size_)));
    int gx_max = std::min(grid_cols_ - 1, static_cast<int>(std::floor(max_x / grid_size_)));
    int gy_min = std::max(0, static_cast<int>(std::floor(min_y / grid_size_)));
    int gy_max = std::min(grid_rows_ - 1, static_cast<int>(std::floor(max_y / grid_size_)));

    for (int gy = gy_min; gy <= gy_max; ++gy) {
        for (int gx = gx_min; gx <= gx_max; ++gx) {
            blocked_[gy * grid_cols_ + gx] = true;
        }
    }
}


