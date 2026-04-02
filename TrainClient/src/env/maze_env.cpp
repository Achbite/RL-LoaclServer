#include "env/maze_env.h"
#include "config/config_loader.h"
#include "log/logger.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <dirent.h>

// ---- 从完整配置初始化环境 ----
void MazeEnv::Init(const ClientConfig& config) {
    const EnvConfig& env = config.env;

    // 保存地图文件路径和目录
    map_file_ = env.map_file;
    map_dir_  = env.map_dir;

    // 加载地图参数（先设默认值，地图文件可能覆盖）
    map_width_     = env.map_width;
    map_height_    = env.map_height;
    max_steps_     = env.max_steps;
    start_x_       = env.start_x;
    start_y_       = env.start_y;
    end_x_         = env.end_x;
    end_y_         = env.end_y;
    grid_size_     = env.grid_size;

    // 计算网格尺寸（默认值，地图文件可能覆盖）
    grid_cols_ = static_cast<int>(std::ceil(map_width_ / grid_size_));
    grid_rows_ = static_cast<int>(std::ceil(map_height_ / grid_size_));

    // 初始化网格障碍物
    blocked_.assign(grid_cols_ * grid_rows_, false);
    walls_.clear();

    // ---- 地图加载优先级：map_file > map_dir 随机选取 > 默认墙壁 ----
    bool map_loaded = false;

    // 1. 优先使用指定的地图文件
    if (!map_file_.empty()) {
        if (LoadMapFromFile(map_file_)) {
            LOG_INFO("MazeEnv", "从指定地图文件加载成功: %s", map_file_.c_str());
            map_loaded = true;
        } else {
            LOG_WARN("MazeEnv", "指定地图文件加载失败: %s", map_file_.c_str());
        }
    }

    // 2. 从 map_dir 目录随机选取一个地图文件
    if (!map_loaded && !map_dir_.empty()) {
        std::string picked = ScanAndPickMap(map_dir_);
        if (!picked.empty()) {
            if (LoadMapFromFile(picked)) {
                LOG_INFO("MazeEnv", "从目录随机选取地图加载成功: %s", picked.c_str());
                map_loaded = true;
            } else {
                LOG_WARN("MazeEnv", "随机选取的地图文件加载失败: %s", picked.c_str());
            }
        } else {
            LOG_INFO("MazeEnv", "地图目录无可用 .json 文件: %s", map_dir_.c_str());
        }
    }

    // 3. 兜底：使用默认墙壁
    if (!map_loaded) {
        LOG_INFO("MazeEnv", "使用默认墙壁");
        LoadWalls();
    }

    // 计算起点/终点的网格坐标（地图文件可能已覆盖起终点坐标）
    start_gx_ = ToGridX(start_x_);
    start_gy_ = ToGridY(start_y_);
    end_gx_   = ToGridX(end_x_);
    end_gy_   = ToGridY(end_y_);

    // 确保起点和终点可通行
    blocked_[start_gy_ * grid_cols_ + start_gx_] = false;
    blocked_[end_gy_ * grid_cols_ + end_gx_]     = false;

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

    LOG_INFO("MazeEnv", "初始化完成: agent_num=%d, grid=%dx%d (格子=%.2fcm), "
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

// ---- 加载默认墙壁到网格（map_file 为空时的兜底方案）----
void MazeEnv::LoadWalls() {
    // 内部隔墙（硬编码默认 3 面墙壁，不添加外围边界墙壁，网格越界检查天然阻止越界）
    struct { float x1, y1, x2, y2, t; } default_walls[] = {
        {5000, 0, 5000, 14000, 100},
        {10000, 6000, 10000, 20000, 100},
        {15000, 0, 15000, 14000, 100}
    };

    for (const auto& w : default_walls) {
        AddWallToGrid(w.x1, w.y1, w.x2, w.y2, w.t);
        walls_.push_back({w.x1, w.y1, w.x2, w.y2, w.t});
    }

    LOG_INFO("MazeEnv", "默认墙壁加载完成: %d 面内部隔墙", static_cast<int>(walls_.size()));
}

// ---- 从 JSON 文件加载地图数据 ----
bool MazeEnv::LoadMapFromFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        LOG_WARN("MazeEnv", "无法打开地图文件: %s", filepath.c_str());
        return false;
    }

    // 读取整个文件内容
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    ifs.close();

    // 轻量 JSON 解析：提取 start_pos、end_pos、bounds、walls
    // 辅助 lambda：查找 "key": value 中的数值
    auto findNumber = [&](const std::string& text, const std::string& key) -> float {
        std::string pattern = "\"" + key + "\"";
        size_t pos = text.find(pattern);
        if (pos == std::string::npos) return -1.0f;
        pos = text.find(':', pos + pattern.size());
        if (pos == std::string::npos) return -1.0f;
        pos++;
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) pos++;
        size_t end = pos;
        while (end < text.size() && (std::isdigit(text[end]) || text[end] == '.' || text[end] == '-')) end++;
        if (end == pos) return -1.0f;
        try { return std::stof(text.substr(pos, end - pos)); } catch (...) { return -1.0f; }
    };

    // 解析 start_pos
    size_t sp_pos = content.find("\"start_pos\"");
    if (sp_pos != std::string::npos) {
        size_t brace = content.find('{', sp_pos);
        size_t brace_end = content.find('}', brace);
        if (brace != std::string::npos && brace_end != std::string::npos) {
            std::string block = content.substr(brace, brace_end - brace + 1);
            float sx = findNumber(block, "x");
            float sy = findNumber(block, "y");
            if (sx >= 0 && sy >= 0) {
                start_x_ = sx;
                start_y_ = sy;
                LOG_INFO("MazeEnv", "地图起点: (%.0f, %.0f)", start_x_, start_y_);
            }
        }
    }

    // 解析 end_pos
    size_t ep_pos = content.find("\"end_pos\"");
    if (ep_pos != std::string::npos) {
        size_t brace = content.find('{', ep_pos);
        size_t brace_end = content.find('}', brace);
        if (brace != std::string::npos && brace_end != std::string::npos) {
            std::string block = content.substr(brace, brace_end - brace + 1);
            float ex = findNumber(block, "x");
            float ey = findNumber(block, "y");
            if (ex >= 0 && ey >= 0) {
                end_x_ = ex;
                end_y_ = ey;
                LOG_INFO("MazeEnv", "地图终点: (%.0f, %.0f)", end_x_, end_y_);
            }
        }
    }

    // 解析 grid_size（v2 格式，覆盖配置值）
    float json_grid_size = findNumber(content, "grid_size");
    if (json_grid_size > 0) {
        grid_size_ = json_grid_size;
        LOG_INFO("MazeEnv", "地图 grid_size: %.2f", grid_size_);
    }

    // 解析 grid_count（v2 格式，直接使用，不再 ceil 计算）
    float json_grid_count = findNumber(content, "grid_count");
    bool has_grid_count = (json_grid_count > 0);

    // 解析 bounds（可选，覆盖地图尺寸）
    size_t bounds_pos = content.find("\"bounds\"");
    if (bounds_pos != std::string::npos) {
        size_t brace = content.find('{', bounds_pos);
        size_t brace_end = content.find('}', brace);
        if (brace != std::string::npos && brace_end != std::string::npos) {
            std::string block = content.substr(brace, brace_end - brace + 1);
            float x_max = findNumber(block, "x_max");
            float y_max = findNumber(block, "y_max");
            if (x_max > 0) map_width_ = x_max;
            if (y_max > 0) map_height_ = y_max;
        }
    }

    // 重建网格：优先使用地图提供的 grid_count，否则 ceil 计算
    if (has_grid_count) {
        int gc = static_cast<int>(json_grid_count);
        grid_cols_ = gc;
        grid_rows_ = gc;
        LOG_INFO("MazeEnv", "地图 grid_count: %d（直接使用）", gc);
    } else {
        grid_cols_ = static_cast<int>(std::ceil(map_width_ / grid_size_));
        grid_rows_ = static_cast<int>(std::ceil(map_height_ / grid_size_));
        LOG_INFO("MazeEnv", "grid_count 由 ceil 计算: %dx%d", grid_cols_, grid_rows_);
    }
    blocked_.assign(grid_cols_ * grid_rows_, false);

    // 解析 walls 数组
    size_t walls_pos = content.find("\"walls\"");
    if (walls_pos == std::string::npos) {
        LOG_WARN("MazeEnv", "地图文件缺少 walls 字段");
        return false;
    }

    // 找到 walls 数组的起始 '['
    size_t arr_start = content.find('[', walls_pos);
    if (arr_start == std::string::npos) return false;

    // 逐个解析墙壁对象 {...}
    int wall_count = 0;
    size_t search_pos = arr_start;
    while (true) {
        size_t obj_start = content.find('{', search_pos);
        if (obj_start == std::string::npos) break;

        // 检查是否已超出 walls 数组（遇到 ']'）
        size_t arr_end_check = content.find(']', search_pos);
        if (arr_end_check != std::string::npos && arr_end_check < obj_start) break;

        size_t obj_end = content.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string wall_str = content.substr(obj_start, obj_end - obj_start + 1);

        float x1 = findNumber(wall_str, "x1");
        float y1 = findNumber(wall_str, "y1");
        float x2 = findNumber(wall_str, "x2");
        float y2 = findNumber(wall_str, "y2");
        float t  = findNumber(wall_str, "thickness");
        if (t < 0) t = 10.0f;  // 默认厚度（与地图生成器一致）

        if (x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0) {
            AddWallToGrid(x1, y1, x2, y2, t);
            walls_.push_back({x1, y1, x2, y2, t});
            wall_count++;
        }

        search_pos = obj_end + 1;
    }

    LOG_INFO("MazeEnv", "地图文件加载完成: %d 面墙壁, 地图尺寸=%.0fx%.0f",
             wall_count, map_width_, map_height_);
    return wall_count > 0;
}

// ---- 8 方向射线检测 ----
RayResult MazeEnv::CastRays(int gx, int gy, int max_range) const {
    // 方向顺序：上、右上、右、右下、下、左下、左、左上（与 AIServer BuildObs 一致）
    static const int ray_dx[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    static const int ray_dy[8] = { 1,  1,  0, -1, -1, -1,  0,  1};

    RayResult result;
    for (int d = 0; d < 8; ++d) {
        int ray_dist = 0;
        for (int step = 1; step <= max_range; ++step) {
            int nx = gx + ray_dx[d] * step;
            int ny = gy + ray_dy[d] * step;
            if (!IsWalkable(nx, ny)) break;
            ray_dist = step;
        }
        result.distances[d] = static_cast<float>(ray_dist) / max_range;
    }
    return result;
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

// ---- 扫描目录并随机选取一个 .json 地图文件 ----
std::string MazeEnv::ScanAndPickMap(const std::string& dir_path) {
    std::vector<std::string> json_files;

    DIR* dir = opendir(dir_path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.size() > 5 && name.substr(name.size() - 5) == ".json") {
                json_files.push_back(dir_path + "/" + name);
            }
        }
        closedir(dir);
    }

    if (json_files.empty()) {
        return "";
    }

    // 随机选取一个
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(json_files.size()) - 1);
    int idx = dist(rng);

    LOG_INFO("MazeEnv", "目录 %s 下发现 %zu 个地图文件，随机选取: %s",
             dir_path.c_str(), json_files.size(), json_files[idx].c_str());
    return json_files[idx];
}


