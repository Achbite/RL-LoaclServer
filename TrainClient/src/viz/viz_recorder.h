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
    // map_id: 当前地图 ID（用于地图文件复制命名）
    // map_source_path: 地图源文件路径（非空时复制到 output_dir/maps/{map_id}.json）
    bool Begin(const std::string& output_dir, int episode_id,
               const std::string& map_id = "",
               const std::string& map_source_path = "");

    // 记录一帧数据（JSON 格式，写入一行）
    void RecordFrame(const std::string& json_line);

    // 结束当前 Episode 的记录（关闭文件）
    void End();

    // 获取当前已记录的帧数（用于判断是否为首帧）
    int GetFrameCount() const { return frame_count_; }

private:
    // 复制地图文件到 output_dir/maps/ 目录（已存在则跳过）
    void CopyMapFile(const std::string& map_id, const std::string& source_path);
    FILE* file_ = nullptr;
    std::string output_dir_;
    int frame_count_ = 0;
};
