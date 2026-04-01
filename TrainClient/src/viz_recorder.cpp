#include "viz_recorder.h"
#include "logger.h"

// ---- 析构函数 ----
VizRecorder::~VizRecorder() {
    End();
}

// ---- 开始新 Episode 的记录 ----
bool VizRecorder::Begin(const std::string& output_dir, int episode_id) {
    // 关闭上一个 Episode 的文件（如果有）
    End();

    output_dir_ = output_dir;
    frame_count_ = 0;

    // 创建输出目录
    mkdir(output_dir.c_str(), 0755);

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
    return true;
}

// ---- 记录一帧数据 ----
void VizRecorder::RecordFrame(const std::string& json_line) {
    if (!file_) return;

    std::fprintf(file_, "%s\n", json_line.c_str());
    ++frame_count_;

    // 每 100 帧刷新一次缓冲区
    if (frame_count_ % 100 == 0) {
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
