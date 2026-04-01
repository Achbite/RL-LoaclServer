#include "ai/maze_reward.h"

#include <cmath>
#include <algorithm>

// ---- 通关奖励 ----
static float GoalReward(const SessionManager::Session& session,
                        int gx, int gy, bool is_done) {
    if (is_done && gx == session.end_gx && gy == session.end_gy) {
        return 10.0f;
    }
    return 0.0f;
}

// ---- 有效位移奖励：朝目标方向移动给正奖励，远离给负奖励 ----
static float DisplacementReward(const SessionManager::Session& session,
                                int agent_id, int gx, int gy) {
    auto it = session.agents.find(agent_id);
    if (it == session.agents.end()) {
        return 0.0f;
    }

    const auto& agent = it->second;
    if (agent.prev_grid_x < 0 || agent.prev_grid_y < 0) {
        return 0.0f;
    }

    // 计算前后到终点的欧氏距离
    float prev_dx = static_cast<float>(session.end_gx - agent.prev_grid_x);
    float prev_dy = static_cast<float>(session.end_gy - agent.prev_grid_y);
    float prev_dist = std::sqrt(prev_dx * prev_dx + prev_dy * prev_dy);

    float curr_dx = static_cast<float>(session.end_gx - gx);
    float curr_dy = static_cast<float>(session.end_gy - gy);
    float curr_dist = std::sqrt(curr_dx * curr_dx + curr_dy * curr_dy);

    // 双向引导：靠近正奖励，远离负奖励
    float delta = prev_dist - curr_dist;
    return 0.1f * delta;
}

// ---- 超时惩罚（未通关且超时）----
static float TimeoutPenalty(const SessionManager::Session& session,
                            int gx, int gy, bool is_done) {
    if (is_done && !(gx == session.end_gx && gy == session.end_gy)) {
        return -2.0f;
    }
    return 0.0f;
}

// ---- 排名奖励计算 ----
float MazeReward::CalculateRankReward(const std::vector<int>& ranking_order,
                                       int agent_num, int agent_id, bool reached_goal) {
    if (ranking_order.empty() || agent_num <= 1) {
        return 0.0f;
    }

    // 查找该 Agent 在排名中的位置（0-based）
    int rank = -1;
    for (int i = 0; i < static_cast<int>(ranking_order.size()); ++i) {
        if (ranking_order[i] == agent_id) {
            rank = i;
            break;
        }
    }

    float reward = 0.0f;

    if (rank >= 0) {
        // 在排名中（已完成的 Agent）
        float rank_ratio = static_cast<float>(rank) / agent_num;
        float half = 0.5f;

        if (rank == 0) {
            // 第 1 名：额外 +5.0 通关奖励
            reward += 5.0f;
        }

        if (rank_ratio < half) {
            // 前 50%：正奖励，按排名比例递减
            reward += 3.0f * (1.0f - rank_ratio);
        } else {
            // 后 50%：负奖励，按落后程度递增
            reward -= 3.0f * (rank_ratio - half) / half;
        }
    } else {
        // 不在排名中（未完成的 Agent，被倒计时强制结束）
        // 视为最后一名，给最大惩罚
        reward -= 3.0f;
    }

    return reward;
}

// ---- 汇总计算所有奖励分项 ----
RewardDetail MazeReward::Calculate(const SessionManager::Session& session,
                                   int agent_id, int gx, int gy, bool is_done,
                                   int agent_num) {
    RewardDetail detail;

    // 通关奖励
    float goal = GoalReward(session, gx, gy, is_done);
    detail.items.emplace_back("goal_reward", goal);

    // 有效位移奖励（双向引导）
    float disp = DisplacementReward(session, agent_id, gx, gy);
    detail.items.emplace_back("displacement_reward", disp);

    // 超时惩罚
    float timeout = TimeoutPenalty(session, gx, gy, is_done);
    detail.items.emplace_back("timeout_penalty", timeout);

    // 排名奖励（仅在终止帧且有 Agent 完成排名时计算）
    float rank_reward = 0.0f;
    if (is_done && !session.ranking_order.empty()) {
        bool reached = (gx == session.end_gx && gy == session.end_gy);
        rank_reward = CalculateRankReward(session.ranking_order, agent_num, agent_id, reached);
    }
    detail.items.emplace_back("rank_reward", rank_reward);

    // 汇总
    detail.total = goal + disp + timeout + rank_reward;

    return detail;
}
