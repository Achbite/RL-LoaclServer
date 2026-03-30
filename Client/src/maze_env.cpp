#include "maze_env.h"
#include "config_loader.h"
#include "logger.h"

#include <cmath>

// ---- 从完整配置初始化环境 ----
void MazeEnv::Init(const ClientConfig& config) {
    const EnvConfig& env = config.env;

    // 加载环境参数
    map_width_     = env.map_width;
    map_height_    = env.map_height;
    move_distance_ = env.move_speed * env.frame_interval;
    goal_radius_   = env.goal_radius;
    max_steps_     = env.max_steps;
    start_x_       = env.start_x;
    start_y_       = env.start_y;
    end_x_         = env.end_x;
    end_y_         = env.end_y;

    // 初始化 Agent
    int agent_num = config.run.agent_num;
    agents_.resize(agent_num);
    for (int i = 0; i < agent_num; ++i) {
        agents_[i].id    = i;
        agents_[i].pos_x = start_x_;
        agents_[i].pos_y = start_y_;
        agents_[i].done  = false;
    }

    frame_id_ = 0;

    LOG_INFO("MazeEnv", "初始化完成: agent_num=%d, start=(%.0f,%.0f), end=(%.0f,%.0f), "
                "move_dist=%.1f, goal_r=%.1f, max_steps=%d",
                agent_num, start_x_, start_y_, end_x_, end_y_,
                move_distance_, goal_radius_, max_steps_);
}

// ---- 重置所有 Agent 到起点 ----
void MazeEnv::Reset() {
    for (auto& agent : agents_) {
        agent.pos_x = start_x_;
        agent.pos_y = start_y_;
        agent.done  = false;
    }
    frame_id_ = 0;
}

// ---- 执行动作 ----
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

    // ---- 1. 计算新位置 ----
    float dx = kActionDirs[action_id][0] * move_distance_;
    float dy = kActionDirs[action_id][1] * move_distance_;
    agent.pos_x += dx;
    agent.pos_y += dy;

    // ---- 2. 边界裁剪 ----
    ClampPosition(agent);

    // ---- 3. 终止判定 ----
    if (CheckGoalReached(agent)) {
        agent.done = true;
        LOG_INFO("MazeEnv", "Agent %d 到达终点! frame=%d pos=(%.1f,%.1f)",
                    agent_id, frame_id_, agent.pos_x, agent.pos_y);
    } else if (CheckTimeout()) {
        agent.done = true;
        LOG_INFO("MazeEnv", "Agent %d 超时! frame=%d pos=(%.1f,%.1f)",
                    agent_id, frame_id_, agent.pos_x, agent.pos_y);
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

// ---- Agent 数量 ----
int MazeEnv::GetAgentNum() const {
    return static_cast<int>(agents_.size());
}

// ---- 是否到达终点 ----
bool MazeEnv::CheckGoalReached(const AgentInfo& agent) const {
    float dx = agent.pos_x - end_x_;
    float dy = agent.pos_y - end_y_;
    float dist = std::sqrt(dx * dx + dy * dy);
    return dist <= goal_radius_;
}

// ---- 是否超时 ----
bool MazeEnv::CheckTimeout() const {
    return frame_id_ >= max_steps_;
}

// ---- 位置裁剪 ----
void MazeEnv::ClampPosition(AgentInfo& agent) {
    if (agent.pos_x < 0.0f)        agent.pos_x = 0.0f;
    if (agent.pos_x > map_width_)   agent.pos_x = map_width_;
    if (agent.pos_y < 0.0f)        agent.pos_y = 0.0f;
    if (agent.pos_y > map_height_)  agent.pos_y = map_height_;
}
