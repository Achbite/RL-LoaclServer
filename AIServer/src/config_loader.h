#pragma once

#include <string>

// ---- 服务参数 ----
struct ServerConfig {
int listen_port = 9002;             // gRPC 监听端口
    int max_agents  = 10;               // 最大 Agent 数量
};

// ---- 策略参数 ----
struct StrategyConfig {
    std::string mode       = "astar";   // 策略模式：astar / random
    int         grid_size  = 500;       // A* 网格大小 (cm)
    int         replan_interval = 10;   // 重新规划间隔（帧）
};

// ---- Learner 连接参数 ----
struct LearnerConfig {
    std::string host          = "127.0.0.1";    // Learner 地址
int         port          = 9003;            // Learner 端口
    int         send_interval = 32;              // 样本发送间隔（帧）
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    ServerConfig   server;
    StrategyConfig strategy;
    LearnerConfig  learner;
};

// ---- 配置加载器 ----
// 从 YAML 文件加载配置，失败时使用默认值
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
