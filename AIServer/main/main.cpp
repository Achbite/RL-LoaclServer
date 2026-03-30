#include "maze_service.h"
#include "config_loader.h"
#include "logger.h"

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
    const char* config_path = (argc > 1) ? argv[1] : kDefaultConfigPath;
    AIServerConfig cfg;
    LoadServerConfig(config_path, cfg);

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
    LOG_INFO("Main", "策略模式: %s", cfg.strategy.mode.c_str());
    LOG_INFO("Main", "等待 Client 连接...");

    // ---- 4. 等待退出信号 ----
    server->Wait();

    LOG_INFO("Main", "AIServer 已停止");
    Logger::Instance().Close();
    return 0;
}
