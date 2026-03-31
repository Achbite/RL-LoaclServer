#pragma once

#include <string>

// ---- 服务参数 ----
struct ServerConfig {
    int listen_port = 9002;             // gRPC 监听端口
    int max_agents  = 10;               // 最大 Agent 数量
    int run_mode    = 3;                // 运行模式：1=训练, 2=推理, 3=A*测试
};

// ---- 策略参数 ----
struct StrategyConfig {
    int         grid_size  = 500;       // 网格大小 (cm)
    int         replan_interval = 10;   // A* 模式下重新规划间隔（帧）
};

// ---- 模型参数 ----
struct ModelConfig {
    std::string local_dir     = "models/local";   // 本地模型目录（推理优先）
    std::string p2p_dir       = "models/p2p";     // P2P 模型目录（Learner 共享卷）
    int         poll_interval = 10;               // 模型轮询间隔（秒）
    std::string save_name     = "SaveModel";      // 本地保存的模型文件名（不含扩展名）
};

// ---- Learner 连接参数 ----
struct LearnerConfig {
    std::string host              = "127.0.0.1";    // Learner 地址
    int         port              = 9003;            // Learner 端口
    int         send_interval     = 32;              // 样本发送间隔（帧）
    int         sample_batch_size = 128;             // 样本批量发送大小
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    ServerConfig   server;
    StrategyConfig strategy;
    ModelConfig    model;
    LearnerConfig  learner;
};

// ---- 配置加载器 ----
// 从 YAML 文件加载配置，失败时使用默认值
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
