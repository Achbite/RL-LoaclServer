#include "viz/viz_recorder.h"
#include "log/logger.h"

#include <cerrno>
#include <cstring>
#include <sys/stat.h>

// ---- 递归创建目录（等效 mkdir -p）----
static bool MkdirRecursive(const std::string& path) {
    if (path.empty()) return false;

    // 逐级检查并创建
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            struct stat st;
            if (stat(current.c_str(), &st) != 0) {
                if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
    }
    return true;
}

// ---- 析构函数 ----
VizRecorder::~VizRecorder() {
    End();
}

// ---- 开始新 Episode 的记录 ----
bool VizRecorder::Begin(const std::string& output_dir, int episode_id,
                         const std::string& map_id,
                         const std::string& map_source_path) {
    // 关闭上一个 Episode 的文件（如果有）
    End();

    output_dir_ = output_dir;
    frame_count_ = 0;

    // 递归创建输出目录（等效 mkdir -p，支持多级路径如 log/viz）
    if (!MkdirRecursive(output_dir)) {
        LOG_ERROR("VizRecorder", "无法创建输出目录: %s (errno=%d: %s)",
                  output_dir.c_str(), errno, std::strerror(errno));
        return false;
    }

    // 生成文件名：ep_NNN_YYYYMMDD_HHMMSS.jsonl
    std::time_t now = std::time(nullptr);
    std::tm* tm = std::localtime(&now);
    char filename[512];
    std::snprintf(filename, sizeof(filename),
                  "%s/ep_%03d_%04d%02d%02d_%02d%02d%02d.jsonl",
                  output_dir.c_str(), episode_id,
                  tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                  tm->tm_hour, tm->tm_min, tm->tm_sec);

    file_ = std::fopen(filename, "w");
    if (!file_) {
        LOG_ERROR("VizRecorder", "无法创建记录文件: %s", filename);
        return false;
    }

    LOG_DEBUG("VizRecorder", "开始记录: %s", filename);

    // 复制地图文件到 output_dir/maps/ 目录（供播放器独立加载）
    if (!map_id.empty() && !map_source_path.empty()) {
        CopyMapFile(map_id, map_source_path);
    }

    return true;
}

// ---- 记录一帧数据 ----
void VizRecorder::RecordFrame(const std::string& json_line) {
    if (!file_) return;

    std::fprintf(file_, "%s\n", json_line.c_str());
    ++frame_count_;

    // 每 10 帧刷新一次缓冲区（配合 Live 播放模式，降低可视化延迟）
    if (frame_count_ % 10 == 0) {
        std::fflush(file_);
    }
}

// ---- 结束当前 Episode 的记录 ----
void VizRecorder::End() {
    if (file_) {
        std::fflush(file_);
        std::fclose(file_);
        file_ = nullptr;
        LOG_DEBUG("VizRecorder", "记录结束: %d 帧", frame_count_);
    }
    frame_count_ = 0;
}

// ---- 复制地图文件到 output_dir/maps/ 目录 ----
void VizRecorder::CopyMapFile(const std::string& map_id, const std::string& source_path) {
    // 目标目录：output_dir_/maps/
    std::string maps_dir = output_dir_ + "/maps";
    MkdirRecursive(maps_dir);

    // 目标文件：maps/{map_id}.json
    std::string dest_path = maps_dir + "/" + map_id + ".json";

    // 已存在则跳过（同一地图不重复复制）
    struct stat st;
    if (stat(dest_path.c_str(), &st) == 0) {
        return;
    }

    // 复制文件
    FILE* src = std::fopen(source_path.c_str(), "rb");
    if (!src) {
        LOG_WARN("VizRecorder", "无法打开地图源文件: %s", source_path.c_str());
        return;
    }

    FILE* dst = std::fopen(dest_path.c_str(), "wb");
    if (!dst) {
        std::fclose(src);
        LOG_WARN("VizRecorder", "无法创建地图目标文件: %s", dest_path.c_str());
        return;
    }

    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), src)) > 0) {
        std::fwrite(buf, 1, n, dst);
    }

    std::fclose(src);
    std::fclose(dst);
    LOG_INFO("VizRecorder", "地图文件已复制: %s -> %s", source_path.c_str(), dest_path.c_str());
}
