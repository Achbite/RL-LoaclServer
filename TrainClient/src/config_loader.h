#pragma once

#include <string>
#include <unordered_map>

// ---- 运行参数（main 使用）----
struct RunConfig {
    int   agent_num     = 1;            // Agent 数量
    int   max_episodes  = 100;          // 最大 Episode 数
    int   log_interval  = 100;          // 日志打印间隔（帧）
};

// ---- 环境参数（MazeEnv 使用）----
struct EnvConfig {
    float map_width      = 20000.0f;    // 地图宽度 (cm)
    float map_height     = 20000.0f;    // 地图高度 (cm)
    int   grid_size      = 500;         // 网格大小 (cm)，将连续坐标离散化为网格
    int   max_steps      = 2000;        // 最大步数
    float start_x        = 500.0f;      // 起点 X
    float start_y        = 500.0f;      // 起点 Y
    float end_x          = 19500.0f;    // 终点 X
    float end_y          = 19500.0f;    // 终点 Y
};

// ---- 网络参数 ----
struct NetworkConfig {
    std::string server_host = "127.0.0.1";  // AIServer 地址
int         server_port = 9002;          // AIServer gRPC 端口
};

// ---- 可视化参数（帧数据记录 + HTTP 回放服务）----
struct VizConfig {
    bool        enabled     = true;              // 是否启用帧数据记录
    std::string output_dir  = "log/viz";         // 记录文件输出目录
    int         interval    = 1;                 // 记录间隔（帧），每隔多少帧记录一次
    int         server_port = 9004;              // 可视化 HTTP 服务端口，浏览器通过此端口访问回放
};

// ---- 并行训练参数（train_main 使用）----
struct TrainConfig {
    int thread_count       = 4;         // 工作线程数，即同时并行运行的 Episode 数
    int episode_pool_size  = 0;         // 对象池大小，0=自动等于 thread_count
    int max_episodes       = 1000;      // 总训练 Episode 数
    int log_summary_interval = 10;      // 摘要日志间隔（每完成 N 个 Episode 打印一次汇总）
};

// ---- 客户端完整配置 ----
struct ClientConfig {
    RunConfig     run;
    EnvConfig     env;
    NetworkConfig network;
    VizConfig     viz;
    TrainConfig   train;
};

// ---- 配置加载器 ----
// 从 YAML 文件加载配置，失败时使用默认值
bool LoadClientConfig(const std::string& yaml_path, ClientConfig& out_config);
