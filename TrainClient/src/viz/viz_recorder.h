#pragma once

#include <string>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>

// ---- 帧数据记录器（切片记录，用于离线可视化回放）----
// 每个 Episode 生成一个 .jsonl 文件，每行一个 JSON 帧数据
class VizRecorder {
public:
    VizRecorder() = default;
    ~VizRecorder();

    // 开始新 Episode 的记录（创建输出目录 + 打开 .jsonl 文件）
    bool Begin(const std::string& output_dir, int episode_id);

    // 记录一帧数据（JSON 格式，写入一行）
    void RecordFrame(const std::string& json_line);

    // 结束当前 Episode 的记录（关闭文件）
    void End();

private:
    FILE* file_ = nullptr;
    std::string output_dir_;
    int frame_count_ = 0;
};
