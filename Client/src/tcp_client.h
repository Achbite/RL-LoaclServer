#pragma once

#include <string>
#include <cstdint>

namespace google { namespace protobuf { class Message; } }

// ---- TCP 长连接客户端（阻塞式，用于 Client↔AIServer 通信）----
class TcpClient {
public:
    TcpClient() = default;
    ~TcpClient();

    // 连接管理
    bool Connect(const std::string& host, int port);    // 建立 TCP 连接
    void Disconnect();                                   // 断开连接
    bool IsConnected() const;                            // 连接状态查询

    // 消息收发（帧格式：[4B msg_type][4B payload_len][payload]）
    bool SendMessage(int32_t msg_type, const google::protobuf::Message& msg);   // 序列化并发送
    bool RecvMessage(int32_t& out_msg_type, std::string& out_payload);          // 接收并返回原始 payload

private:
    // 底层 IO
    bool SendRaw(const void* data, size_t len);     // 发送原始字节
    bool RecvRaw(void* buf, size_t len);             // 接收指定长度字节

    // --- socket 句柄 ---
    int socket_fd_ = -1;
    bool connected_ = false;
};
