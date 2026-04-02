#include "ai/maze_reward.h"

#include <cmath>
#include <algorithm>

// --- 奖励常量 ---
static constexpr float kGoalReward       = 10.0f;   // 通关奖励
static constexpr float kTimeoutPenalty    = -2.0f;   // 超时惩罚
static constexpr float kPotentialScale    = 1.0f;    // 势能引导缩放系数
static constexpr float kExplorationBonus  = 0.05f;   // 首次访问新格子的探索奖励
static constexpr float kLoiterPenalty     = -0.02f;  // 短期重复访问同一格子的徘徊惩罚
static constexpr int   kLoiterWindow      = 8;       // 徘徊检测滑动窗口大小（步数）
static constexpr float kGamma             = 0.99f;   // 折扣因子（与 PPO 训练配置一致）

// ---- 通关奖励 ----
static float GoalReward(const SessionManager::Session& session,
                        int gx, int gy, bool is_done) {
    if (is_done && gx == session.end_gx && gy == session.end_gy) {
        return kGoalReward;
    }
    return 0.0f;
}

// ---- 势能引导奖励（Potential-Based Reward Shaping, Ng 1999）----
// F(s, s') = γ × Φ(s') - Φ(s)，其中 Φ(s) = -dist(s, goal) / max_dist
// 数学保证：来回震荡时净奖励为负（因 γ < 1），从根本上杜绝刷奖励
static float PotentialReward(const SessionManager::Session& session,
                             int agent_id, int gx, int gy) {
    auto it = session.agents.find(agent_id);
    if (it == session.agents.end()) return 0.0f;

    const auto& agent = it->second;
    if (agent.prev_grid_x < 0 || agent.prev_grid_y < 0) return 0.0f;

    // 计算网格对角线长度（归一化基准）
    float max_dist = std::sqrt(static_cast<float>(
        session.grid_cols * session.grid_cols + session.grid_rows * session.grid_rows));
    if (max_dist <= 0.0f) return 0.0f;

    // 前一状态势能 Φ(s)
    float prev_dx = static_cast<float>(session.end_gx - agent.prev_grid_x);
    float prev_dy = static_cast<float>(session.end_gy - agent.prev_grid_y);
    float phi_prev = -std::sqrt(prev_dx * prev_dx + prev_dy * prev_dy) / max_dist;

    // 当前状态势能 Φ(s')
    float curr_dx = static_cast<float>(session.end_gx - gx);
    float curr_dy = static_cast<float>(session.end_gy - gy);
    float phi_curr = -std::sqrt(curr_dx * curr_dx + curr_dy * curr_dy) / max_dist;

    // F(s, s') = γ × Φ(s') - Φ(s)
    return kPotentialScale * (kGamma * phi_curr - phi_prev);
}

// ---- 探索奖励：首次访问新网格给奖励 ----
static float ExplorationReward(const SessionManager::AgentRuntime& agent,
                               int gx, int gy, int grid_cols) {
    int key = gy * grid_cols + gx;
    if (agent.visited.find(key) == agent.visited.end()) {
        return kExplorationBonus;
    }
    return 0.0f;
}

// ---- 徘徊惩罚：短时间内重复访问同一网格 ----
static float LoiterPenalty(const SessionManager::AgentRuntime& agent,
                           int gx, int gy, int grid_cols) {
    int key = gy * grid_cols + gx;
    for (int pos : agent.recent_positions) {
        if (pos == key) {
            return kLoiterPenalty;
        }
    }
    return 0.0f;
}

// ---- 超时惩罚（未通关且超时）----
static float TimeoutPenalty(const SessionManager::Session& session,
                            int gx, int gy, bool is_done) {
    if (is_done && !(gx == session.end_gx && gy == session.end_gy)) {
        return kTimeoutPenalty;
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

    // 势能引导（替代旧的 displacement_reward，防刷奖励）
    float potential = PotentialReward(session, agent_id, gx, gy);
    detail.items.emplace_back("potential_reward", potential);

    // 探索奖励（首次访问新网格）
    auto it = session.agents.find(agent_id);
    float explore = 0.0f;
    float loiter = 0.0f;
    if (it != session.agents.end()) {
        explore = ExplorationReward(it->second, gx, gy, session.grid_cols);
        loiter = LoiterPenalty(it->second, gx, gy, session.grid_cols);
    }
    detail.items.emplace_back("exploration_reward", explore);

    // 徘徊惩罚（短期重复访问）
    detail.items.emplace_back("loiter_penalty", loiter);

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
    detail.total = goal + potential + explore + loiter + timeout + rank_reward;

    return detail;
}
