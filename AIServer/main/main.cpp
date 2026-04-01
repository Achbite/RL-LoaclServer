#include "grpc/maze_service.h"
#include "config/config_loader.h"
#include "log/logger.h"

#include <grpcpp/grpcpp.h>
#include <cstdio>
#include <string>
#include <csignal>
#include <atomic>

// --- 全局信号标志 ---
static std::atomic<bool> g_running{true};

// ---- 信号处理（优雅退出）----
static void SignalHandler(int sig) {
    LOG_INFO("Main", "收到信号 %d，准备退出...", sig);
    g_running.store(false);
}

// --- 默认配置文件路径 ---
static const char* kDefaultConfigPath = "configs/server_config.yaml";

// ---- 解析命令行参数 ----
static bool HasFlag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    std::printf("============================================\n");
    std::printf("  迷宫训练框架 - AIServer (Demo)\n");
    std::printf("============================================\n\n");

    // ---- 0. 初始化日志系统 ----
    Logger::Instance().Init("log");
    Logger::Instance().SetConsoleLevel(LogLevel::INFO);
    Logger::Instance().SetFileLevel(LogLevel::DEBUG);

    // ---- 1. 注册信号处理 ----
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // ---- 2. 加载配置 ----
    // 查找第一个非 -- 开头的参数作为配置文件路径
    const char* config_path = kDefaultConfigPath;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).substr(0, 2) != "--") {
            config_path = argv[i];
            break;
        }
    }
    AIServerConfig cfg;
    LoadServerConfig(config_path, cfg);

    // ---- 2a. 处理 --train 参数：强制训练模式 ----
    if (HasFlag(argc, argv, "--train")) {
        cfg.server.run_mode = 1;
        LOG_INFO("Main", "--train 参数生效: run_mode=1(训练)");
    }

    // ---- 3. 创建 gRPC 服务 ----
    MazeServiceImpl service(cfg);

    std::string listen_addr = "0.0.0.0:" + std::to_string(cfg.server.listen_port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

    if (!server) {
        LOG_ERROR("Main", "gRPC 服务启动失败，端口: %s", listen_addr.c_str());
        return 1;
    }

    LOG_INFO("Main", "AIServer 已启动，监听: %s", listen_addr.c_str());
    LOG_INFO("Main", "运行模式: %d (%s)", cfg.server.run_mode,
             cfg.server.run_mode == 1 ? "训练" :
             cfg.server.run_mode == 2 ? "推理" : "A*测试");
    LOG_INFO("Main", "等待 Client 连接...");

    // ---- 4. 等待退出信号 ----
    server->Wait();

    LOG_INFO("Main", "AIServer 已停止");
    Logger::Instance().Close();
    return 0;
}
