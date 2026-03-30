#include "tcp_client.h"
#include "logger.h"

#include <google/protobuf/message.h>
#include <cstring>

// --- Linux POSIX Socket ---
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

TcpClient::~TcpClient() {
    Disconnect();
}

// ---- 建立 TCP 连接 ----
bool TcpClient::Connect(const std::string& host, int port) {
    if (connected_) {
        return true;
    }

    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        LOG_ERROR("TcpClient", "socket 创建失败");
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    int ret = connect(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));

    if (ret != 0) {
        LOG_ERROR("TcpClient", "连接 %s:%d 失败", host.c_str(), port);
        close(socket_fd_);
        return false;
    }

    connected_ = true;
    LOG_INFO("TcpClient", "已连接 %s:%d", host.c_str(), port);
    return true;
}

// ---- 断开连接 ----
void TcpClient::Disconnect() {
    if (connected_) {
        close(socket_fd_);
        connected_ = false;
        LOG_INFO("TcpClient", "已断开连接");
    }
}

// ---- 连接状态查询 ----
bool TcpClient::IsConnected() const {
    return connected_;
}

// ---- 序列化并发送 Protobuf 消息 ----
bool TcpClient::SendMessage(int32_t msg_type, const google::protobuf::Message& msg) {
    // ---- 1. 序列化 payload ----
    std::string payload;
    if (!msg.SerializeToString(&payload)) {
        LOG_ERROR("TcpClient", "序列化失败: msg_type=%d", msg_type);
        return false;
    }

    // ---- 2. 写入帧头：[4B msg_type][4B payload_len] ----
    int32_t payload_len = static_cast<int32_t>(payload.size());

    if (!SendRaw(&msg_type, sizeof(msg_type))) return false;
    if (!SendRaw(&payload_len, sizeof(payload_len))) return false;

    // ---- 3. 写入 payload ----
    if (payload_len > 0) {
        if (!SendRaw(payload.data(), payload_len)) return false;
    }

    return true;
}

// ---- 接收消息（返回 msg_type 和原始 payload）----
bool TcpClient::RecvMessage(int32_t& out_msg_type, std::string& out_payload) {
    // ---- 1. 读取帧头 ----
    if (!RecvRaw(&out_msg_type, sizeof(out_msg_type))) return false;

    int32_t payload_len = 0;
    if (!RecvRaw(&payload_len, sizeof(payload_len))) return false;

    // ---- 2. 读取 payload ----
    out_payload.resize(payload_len);
    if (payload_len > 0) {
        if (!RecvRaw(&out_payload[0], payload_len)) return false;
    }

    return true;
}

// ---- 发送原始字节（循环发送直到全部写完）----
bool TcpClient::SendRaw(const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;

    while (sent < len) {
        ssize_t n = send(socket_fd_, ptr + sent, len - sent, 0);
        if (n <= 0) {
            LOG_ERROR("TcpClient", "发送失败");
            connected_ = false;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

// ---- 接收指定长度字节（循环接收直到全部读完）----
bool TcpClient::RecvRaw(void* buf, size_t len) {
    char* ptr = static_cast<char*>(buf);
    size_t received = 0;

    while (received < len) {
        ssize_t n = recv(socket_fd_, ptr + received, len - received, 0);
        if (n <= 0) {
            LOG_ERROR("TcpClient", "接收失败");
            connected_ = false;
            return false;
        }
        received += static_cast<size_t>(n);
    }
    return true;
}
