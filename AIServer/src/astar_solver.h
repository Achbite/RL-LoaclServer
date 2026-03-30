#pragma once

#include <vector>
#include <cstdint>

// ---- A* 网格坐标 ----
struct GridPos {
    int gx = 0;     // 网格 X 索引
    int gy = 0;     // 网格 Y 索引
};

// ---- A* 寻路器 ----
class AStarSolver {
public:
    // 初始化网格（地图尺寸 + 网格大小）
    void Init(float map_width, float map_height, int grid_size);

    // 添加墙壁障碍（线段墙壁，带厚度）
    void AddWall(float x1, float y1, float x2, float y2, float thickness);

    // 规划路径（连续坐标 → 路径点序列）
    bool PlanPath(float start_x, float start_y, float end_x, float end_y);

    // 根据当前位置获取推荐动作 ID（0-8）
    int GetAction(float cur_x, float cur_y) const;

    // 获取路径点数量
    int GetPathLength() const { return static_cast<int>(path_.size()); }

    // 调试：获取路径点（连续坐标）
    const std::vector<GridPos>& GetPath() const { return path_; }

private:
    // 连续坐标 → 网格坐标
    GridPos ToGrid(float x, float y) const;

    // 网格坐标 → 连续坐标（网格中心）
    void ToWorld(const GridPos& gp, float& x, float& y) const;

    // 网格是否可通行
    bool IsWalkable(int gx, int gy) const;

    // 网格参数
    int   grid_size_  = 500;        // 网格大小 (cm)
    int   grid_cols_  = 0;          // 网格列数
    int   grid_rows_  = 0;          // 网格行数
    float map_width_  = 0.0f;       // 地图宽度
    float map_height_ = 0.0f;       // 地图高度

    // 障碍物网格（true=不可通行）
    std::vector<bool> blocked_;

    // 规划结果（网格坐标序列，从起点到终点）
    std::vector<GridPos> path_;
};
