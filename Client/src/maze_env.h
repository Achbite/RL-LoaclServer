#pragma once

#include <vector>
#include <cstdint>

// 前向声明
struct EnvConfig;

// --- 动作方向表 ---
// action_id 0-8 对应的方向向量（dx, dy），索引即动作 ID
constexpr float kActionDirs[9][2] = {
    { 0.0f,  0.0f},         // 0: 不动
    { 0.0f,  1.0f},         // 1: 上 (Y+)
    { 0.707f, 0.707f},      // 2: 右上
    { 1.0f,  0.0f},         // 3: 右 (X+)
    { 0.707f,-0.707f},      // 4: 右下
    { 0.0f, -1.0f},         // 5: 下 (Y-)
    {-0.707f,-0.707f},      // 6: 左下
    {-1.0f,  0.0f},         // 7: 左 (X-)
    {-0.707f, 0.707f},      // 8: 左上
};

// ---- 单个 Agent 的运行时状态 ----
struct AgentInfo {
    int   id     = 0;
    float pos_x  = 0.0f;       // 当前 X 坐标
    float pos_y  = 0.0f;       // 当前 Y 坐标
    bool  done   = false;       // 是否已结束（到达终点或超时）
};

// ---- 客户端完整配置（前向声明引用）----
struct ClientConfig;

// ---- 迷宫环境模拟器 ----
class MazeEnv {
public:
    // 初始化（从配置加载所有参数）
    void Init(const ClientConfig& config);               // 从完整配置初始化
    void Reset();                                         // 重置所有 Agent 到起点

    // 帧更新
    void Step(int agent_id, int action_id);               // 执行动作，更新位置和终止状态
    void AdvanceFrame();                                  // 帧号递增（所有 Agent Step 完后调用）

    // 查询
    const AgentInfo& GetAgent(int agent_id) const;        // 获取 Agent 状态
    int   GetFrameId() const;                             // 当前帧号
    bool  AllDone() const;                                // 所有 Agent 是否都已结束
    int   GetAgentNum() const;                            // Agent 数量

    // 地图参数
    float GetStartX() const { return start_x_; }
    float GetStartY() const { return start_y_; }
    float GetEndX()   const { return end_x_; }
    float GetEndY()   const { return end_y_; }

private:
    std::vector<AgentInfo> agents_;

    // --- 环境参数（从配置加载）---
    float map_width_      = 20000.0f;   // 地图宽度 (cm)
    float map_height_     = 20000.0f;   // 地图高度 (cm)
    float move_distance_  = 60.0f;      // 每帧移动距离 (cm) = speed * interval
    float goal_radius_    = 300.0f;     // 通关判定距离 (cm)
    int   max_steps_      = 2000;       // 最大步数
    float start_x_        = 500.0f;     // 起点 X
    float start_y_        = 500.0f;     // 起点 Y
    float end_x_          = 19500.0f;   // 终点 X
    float end_y_          = 19500.0f;   // 终点 Y
    int   frame_id_       = 0;          // 当前帧号

    // 终止判定
    bool CheckGoalReached(const AgentInfo& agent) const;  // 是否到达终点
    bool CheckTimeout() const;                             // 是否超时

    // 位置裁剪（限制在地图边界内）
    void ClampPosition(AgentInfo& agent);
};
