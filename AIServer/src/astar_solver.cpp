#include "astar_solver.h"
#include "logger.h"

#include <cmath>
#include <queue>
#include <unordered_map>
#include <algorithm>

// ---- 初始化网格 ----
void AStarSolver::Init(float map_width, float map_height, int grid_size) {
    map_width_  = map_width;
    map_height_ = map_height;
    grid_size_  = grid_size;

    grid_cols_ = static_cast<int>(std::ceil(map_width / grid_size));
    grid_rows_ = static_cast<int>(std::ceil(map_height / grid_size));

    blocked_.assign(grid_cols_ * grid_rows_, false);
    path_.clear();

    LOG_DEBUG("AStar", "网格初始化: %dx%d (格子大小=%dcm)",
                grid_cols_, grid_rows_, grid_size_);
}

// ---- 添加墙壁障碍 ----
void AStarSolver::AddWall(float x1, float y1, float x2, float y2, float thickness) {
    // 计算墙壁的 AABB 包围盒（含厚度扩展）
    float half_t = thickness * 0.5f;
    float min_x = std::min(x1, x2) - half_t;
    float max_x = std::max(x1, x2) + half_t;
    float min_y = std::min(y1, y2) - half_t;
    float max_y = std::max(y1, y2) + half_t;

    // 将 AABB 映射到网格范围
    int gx_min = std::max(0, static_cast<int>(std::floor(min_x / grid_size_)));
    int gx_max = std::min(grid_cols_ - 1, static_cast<int>(std::floor(max_x / grid_size_)));
    int gy_min = std::max(0, static_cast<int>(std::floor(min_y / grid_size_)));
    int gy_max = std::min(grid_rows_ - 1, static_cast<int>(std::floor(max_y / grid_size_)));

    // 标记覆盖的网格为不可通行
    for (int gy = gy_min; gy <= gy_max; ++gy) {
        for (int gx = gx_min; gx <= gx_max; ++gx) {
            blocked_[gy * grid_cols_ + gx] = true;
        }
    }
}

// ---- 连续坐标 → 网格坐标 ----
GridPos AStarSolver::ToGrid(float x, float y) const {
    GridPos gp;
    gp.gx = std::max(0, std::min(grid_cols_ - 1, static_cast<int>(std::floor(x / grid_size_))));
    gp.gy = std::max(0, std::min(grid_rows_ - 1, static_cast<int>(std::floor(y / grid_size_))));
    return gp;
}

// ---- 网格坐标 → 连续坐标（网格中心）----
void AStarSolver::ToWorld(const GridPos& gp, float& x, float& y) const {
    x = (gp.gx + 0.5f) * grid_size_;
    y = (gp.gy + 0.5f) * grid_size_;
}

// ---- 网格是否可通行 ----
bool AStarSolver::IsWalkable(int gx, int gy) const {
    if (gx < 0 || gx >= grid_cols_ || gy < 0 || gy >= grid_rows_) {
        return false;
    }
    return !blocked_[gy * grid_cols_ + gx];
}

// ---- A* 路径规划 ----
bool AStarSolver::PlanPath(float start_x, float start_y, float end_x, float end_y) {
    path_.clear();

    GridPos start = ToGrid(start_x, start_y);
    GridPos goal  = ToGrid(end_x, end_y);

    // 确保起点和终点可通行（如果被墙壁覆盖则强制清除）
    blocked_[start.gy * grid_cols_ + start.gx] = false;
    blocked_[goal.gy * grid_cols_ + goal.gx]   = false;

    // A* 节点
    struct Node {
        int gx, gy;
        float g_cost;       // 从起点到当前节点的实际代价
        float f_cost;       // g_cost + 启发式估计
        int parent_idx;     // 父节点在 closed 列表中的索引（-1=无）
    };

    // 编码网格坐标为唯一整数
    auto encode = [this](int gx, int gy) -> int {
        return gy * grid_cols_ + gx;
    };

    // 启发式函数（八方向切比雪夫距离）
    auto heuristic = [](int gx1, int gy1, int gx2, int gy2) -> float {
        int dx = std::abs(gx1 - gx2);
        int dy = std::abs(gy1 - gy2);
        return static_cast<float>(std::max(dx, dy)) + 0.414f * static_cast<float>(std::min(dx, dy));
    };

    // 优先队列（最小 f_cost 优先）
    struct PQItem {
        float f_cost;
        int   encoded;
        bool operator>(const PQItem& o) const { return f_cost > o.f_cost; }
    };

    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> open_set;
    std::unordered_map<int, Node> all_nodes;
    std::unordered_map<int, bool> closed;

    // 初始化起点
    int start_enc = encode(start.gx, start.gy);
    int goal_enc  = encode(goal.gx, goal.gy);

    Node start_node;
    start_node.gx = start.gx;
    start_node.gy = start.gy;
    start_node.g_cost = 0.0f;
    start_node.f_cost = heuristic(start.gx, start.gy, goal.gx, goal.gy);
    start_node.parent_idx = -1;

    all_nodes[start_enc] = start_node;
    open_set.push({start_node.f_cost, start_enc});

    // 八方向邻居偏移
    static const int dx8[] = { 0, 1, 1, 1, 0,-1,-1,-1};
    static const int dy8[] = { 1, 1, 0,-1,-1,-1, 0, 1};
    static const float cost8[] = {1.0f, 1.414f, 1.0f, 1.414f, 1.0f, 1.414f, 1.0f, 1.414f};

    bool found = false;

    while (!open_set.empty()) {
        PQItem cur = open_set.top();
        open_set.pop();

        if (closed.count(cur.encoded)) continue;
        closed[cur.encoded] = true;

        const Node& cur_node = all_nodes[cur.encoded];

        // 到达终点
        if (cur.encoded == goal_enc) {
            found = true;
            break;
        }

        // 展开邻居
        for (int d = 0; d < 8; ++d) {
            int nx = cur_node.gx + dx8[d];
            int ny = cur_node.gy + dy8[d];

            if (!IsWalkable(nx, ny)) continue;

            int nenc = encode(nx, ny);
            if (closed.count(nenc)) continue;

            // 对角线移动时检查两个相邻格子是否可通行（防止穿墙角）
            if (dx8[d] != 0 && dy8[d] != 0) {
                if (!IsWalkable(cur_node.gx + dx8[d], cur_node.gy) ||
                    !IsWalkable(cur_node.gx, cur_node.gy + dy8[d])) {
                    continue;
                }
            }

            float new_g = cur_node.g_cost + cost8[d];
            float new_f = new_g + heuristic(nx, ny, goal.gx, goal.gy);

            auto it = all_nodes.find(nenc);
            if (it == all_nodes.end() || new_g < it->second.g_cost) {
                Node neighbor;
                neighbor.gx = nx;
                neighbor.gy = ny;
                neighbor.g_cost = new_g;
                neighbor.f_cost = new_f;
                neighbor.parent_idx = cur.encoded;
                all_nodes[nenc] = neighbor;
                open_set.push({new_f, nenc});
            }
        }
    }

    if (!found) {
        LOG_WARN("AStar", "未找到路径! start=(%d,%d) goal=(%d,%d)",
                    start.gx, start.gy, goal.gx, goal.gy);
        return false;
    }

    // 回溯路径
    std::vector<GridPos> reversed_path;
    int trace = goal_enc;
    while (trace != -1) {
        const Node& n = all_nodes[trace];
        reversed_path.push_back({n.gx, n.gy});
        trace = n.parent_idx;
    }

    // 反转得到正序路径
    path_.resize(reversed_path.size());
    for (size_t i = 0; i < reversed_path.size(); ++i) {
        path_[i] = reversed_path[reversed_path.size() - 1 - i];
    }

    LOG_DEBUG("AStar", "路径规划成功: %d 个路径点", static_cast<int>(path_.size()));
    return true;
}

// ---- 根据当前网格坐标获取推荐动作 ID ----
int AStarSolver::GetAction(int cur_gx, int cur_gy) const {
    if (path_.empty()) return 0;

    // 找到路径上距离当前网格最近的点
    int best_idx = 0;
    int best_dist = std::abs(cur_gx - path_[0].gx) + std::abs(cur_gy - path_[0].gy);

    for (int i = 1; i < static_cast<int>(path_.size()); ++i) {
        int dist = std::abs(cur_gx - path_[i].gx) + std::abs(cur_gy - path_[i].gy);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    // 选择路径上的下一个目标点
    int target_idx = std::min(best_idx + 1, static_cast<int>(path_.size()) - 1);

    // 计算网格级方向偏移
    int dx = path_[target_idx].gx - cur_gx;
    int dy = path_[target_idx].gy - cur_gy;

    // 已在目标点上则不动
    if (dx == 0 && dy == 0) return 0;

    // 将偏移 clamp 到 [-1, 1]（每步只移动一格）
    if (dx > 0) dx = 1;
    else if (dx < 0) dx = -1;
    if (dy > 0) dy = 1;
    else if (dy < 0) dy = -1;

    // 网格偏移 → action_id 映射表
    // (dx, dy) → action_id
    static const int grid_action_dirs[9][2] = {
        { 0,  0},   // 0: 不动
        { 0,  1},   // 1: 上 (Y+)
        { 1,  1},   // 2: 右上
        { 1,  0},   // 3: 右 (X+)
        { 1, -1},   // 4: 右下
        { 0, -1},   // 5: 下 (Y-)
        {-1, -1},   // 6: 左下
        {-1,  0},   // 7: 左 (X-)
        {-1,  1},   // 8: 左上
    };

    // 查找匹配的 action_id
    for (int a = 1; a <= 8; ++a) {
        if (grid_action_dirs[a][0] == dx && grid_action_dirs[a][1] == dy) {
            return a;
        }
    }

    return 0;
}
