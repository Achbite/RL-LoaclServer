#pragma once

#include <vector>
#include <cstdint>

// 前向声明
struct EnvConfig;

// --- 网格级动作方向表 ---
// action_id 0-8 对应的网格偏移 (dx, dy)，一次移动即 +1/-1
constexpr int kGridActionDirs[9][2] = {
    { 0,  0},         // 0: 不动
    { 0,  1},         // 1: 上 (Y+)
    { 1,  1},         // 2: 右上
    { 1,  0},         // 3: 右 (X+)
    { 1, -1},         // 4: 右下
    { 0, -1},         // 5: 下 (Y-)
    {-1, -1},         // 6: 左下
    {-1,  0},         // 7: 左 (X-)
    {-1,  1},         // 8: 左上
};

// ---- 单个 Agent 的运行时状态 ----
struct AgentInfo {
    int   id      = 0;
    int   grid_x  = 0;         // 网格 X 坐标
    int   grid_y  = 0;         // 网格 Y 坐标
    bool  done    = false;     // 是否已结束（到达终点或超时）
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
    void Step(int agent_id, int action_id);               // 执行网格级移动，检查可达性
    void AdvanceFrame();                                  // 帧号递增（所有 Agent Step 完后调用）

    // 查询
    const AgentInfo& GetAgent(int agent_id) const;        // 获取 Agent 状态
    int   GetFrameId() const;                             // 当前帧号
    bool  AllDone() const;                                // 所有 Agent 是否都已结束
    bool  HasAnyDone() const;                              // 是否有任一 Agent 已结束
    int   GetFirstDoneFrame() const;                       // 获取首个 Agent 完成时的帧号（-1 表示无）
    int   GetAgentNum() const;                             // Agent 数量

    // 地图参数
    float GetMapWidth()  const { return map_width_; }
    float GetMapHeight() const { return map_height_; }
    float GetStartX() const { return start_x_; }
    float GetStartY() const { return start_y_; }
    float GetEndX()   const { return end_x_; }
    float GetEndY()   const { return end_y_; }
    int   GetGridSize() const { return grid_size_; }
    int   GetGridCols() const { return grid_cols_; }
    int   GetGridRows() const { return grid_rows_; }

    // 网格坐标 → 连续坐标（网格中心，用于可视化和通信）
    float GetWorldX(int gx) const { return (gx + 0.5f) * grid_size_; }
    float GetWorldY(int gy) const { return (gy + 0.5f) * grid_size_; }

    // 连续坐标 → 网格坐标
    int ToGridX(float x) const;
    int ToGridY(float y) const;

    // 网格是否可通行
    bool IsWalkable(int gx, int gy) const;

private:
    std::vector<AgentInfo> agents_;

    // --- 地图参数（从配置加载）---
    float map_width_      = 20000.0f;   // 地图宽度 (cm)
    float map_height_     = 20000.0f;   // 地图高度 (cm)
    int   max_steps_      = 10000;      // 最大步数
    float start_x_        = 500.0f;     // 起点 X（连续坐标）
    float start_y_        = 500.0f;     // 起点 Y（连续坐标）
    float end_x_          = 19500.0f;   // 终点 X（连续坐标）
    float end_y_          = 19500.0f;   // 终点 Y（连续坐标）
    int   frame_id_       = 0;          // 当前帧号

    // --- 网格参数 ---
    int   grid_size_      = 500;        // 网格大小 (cm)
    int   grid_cols_      = 0;          // 网格列数
    int   grid_rows_      = 0;          // 网格行数
    int   start_gx_       = 0;          // 起点网格 X
    int   start_gy_       = 0;          // 起点网格 Y
    int   end_gx_         = 0;          // 终点网格 X
    int   end_gy_         = 0;          // 终点网格 Y

    // 竞争倒计时机制
    int   first_done_frame_ = -1;       // 首个 Agent 完成时的帧号（-1 表示尚无）
    static constexpr int kCountdownFrames = 100;  // 首通关后倒计时帧数

    // --- 网格障碍物（true=不可通行）---
    std::vector<bool> blocked_;

    // 加载墙壁到网格
    void LoadWalls();

    // 添加单面墙壁到网格
    void AddWallToGrid(float x1, float y1, float x2, float y2, float thickness);

    // 终止判定
    bool CheckGoalReached(const AgentInfo& agent) const;  // 是否到达终点网格
    bool CheckTimeout() const;                             // 是否超时
    bool CheckCountdownExpired() const;                    // 首通关后倒计时是否到期
};
