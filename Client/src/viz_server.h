#pragma once

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

// ---- 极简 WebSocket 可视化服务器 ----
// 内嵌于 Client 进程，每帧推送地图和 Agent 状态到浏览器
// 协议：HTTP Upgrade → WebSocket（RFC 6455 基础帧）
class VizServer {
public:
    VizServer() = default;
    ~VizServer();

    // 启动服务（监听指定端口，非阻塞，内部启动线程）
    bool Start(int port);

    // 停止服务
    void Stop();

    // 推送 JSON 数据到所有已连接的客户端
    void Broadcast(const std::string& json_data);

    // 查询连接数
    int GetClientCount() const;

private:
    // 接受连接的主循环（在独立线程中运行）
    void AcceptLoop();

    // 处理单个客户端连接（在独立线程中运行）
    void HandleClient(int client_fd);

    // WebSocket 握手（HTTP Upgrade）
    bool DoHandshake(int client_fd);

    // 发送 WebSocket 文本帧
    bool SendFrame(int client_fd, const std::string& data);

    // 服务端 socket
    int server_fd_ = -1;
    int port_      = 0;

    // 线程管理
    std::thread accept_thread_;
    std::atomic<bool> running_{false};

    // 已连接的客户端 fd 列表
    std::vector<int> clients_;
    mutable std::mutex clients_mutex_;
};
