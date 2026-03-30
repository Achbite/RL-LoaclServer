#include "viz_server.h"
#include "logger.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <functional>

// --- Linux socket 头文件 ---
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

// --- SHA-1 + Base64（WebSocket 握手用，极简实现）---

// ---- SHA-1 计算（RFC 3174 精简实现）----
static void SHA1(const unsigned char* data, size_t len, unsigned char out[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476, h4 = 0xC3D2E1F0;

    // 填充消息
    size_t new_len = len + 1;
    while (new_len % 64 != 56) ++new_len;
    new_len += 8;

    std::vector<unsigned char> msg(new_len, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;

    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i) {
        msg[new_len - 1 - i] = static_cast<unsigned char>(bit_len >> (i * 8));
    }

    // 处理每个 512-bit 块
    for (size_t offset = 0; offset < new_len; offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[offset + i * 4]) << 24) |
                   (static_cast<uint32_t>(msg[offset + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[offset + i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(msg[offset + i * 4 + 3]));
        }
        for (int i = 16; i < 80; ++i) {
            uint32_t val = w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16];
            w[i] = (val << 1) | (val >> 31);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;

        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d);       k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;                   k = 0xCA62C1D6; }

            uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
            e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = temp;
        }

        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    // 输出 20 字节哈希
    for (int i = 0; i < 4; ++i) { out[i]      = static_cast<unsigned char>(h0 >> (24 - i * 8)); }
    for (int i = 0; i < 4; ++i) { out[4 + i]  = static_cast<unsigned char>(h1 >> (24 - i * 8)); }
    for (int i = 0; i < 4; ++i) { out[8 + i]  = static_cast<unsigned char>(h2 >> (24 - i * 8)); }
    for (int i = 0; i < 4; ++i) { out[12 + i] = static_cast<unsigned char>(h3 >> (24 - i * 8)); }
    for (int i = 0; i < 4; ++i) { out[16 + i] = static_cast<unsigned char>(h4 >> (24 - i * 8)); }
}

// ---- Base64 编码 ----
static std::string Base64Encode(const unsigned char* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<uint32_t>(data[i + 2]);

        result += table[(n >> 18) & 0x3F];
        result += table[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? table[n & 0x3F] : '=';
    }

    return result;
}

// ============================================================================
// VizServer 实现
// ============================================================================

// ---- 析构函数 ----
VizServer::~VizServer() {
    Stop();
}

// ---- 启动服务 ----
bool VizServer::Start(int port) {
    port_ = port;

    // 创建 TCP socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        LOG_ERROR("VizServer", "创建 socket 失败");
        return false;
    }

    // 允许端口复用
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 绑定地址
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        LOG_ERROR("VizServer", "绑定端口 %d 失败", port);
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 5) < 0) {
        LOG_ERROR("VizServer", "监听失败");
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    accept_thread_ = std::thread(&VizServer::AcceptLoop, this);

    LOG_INFO("VizServer", "已启动，监听端口: %d", port);
    return true;
}

// ---- 停止服务 ----
void VizServer::Stop() {
    running_.store(false);

    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    // 关闭所有客户端连接
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (int fd : clients_) {
        close(fd);
    }
    clients_.clear();

    LOG_INFO("VizServer", "已停止");
}

// ---- 接受连接的主循环 ----
void VizServer::AcceptLoop() {
    while (running_.load()) {
        // 使用 poll 等待连接（带超时，避免阻塞退出）
        struct pollfd pfd{};
        pfd.fd = server_fd_;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 500);  // 500ms 超时
        if (ret <= 0) continue;

        struct sockaddr_in client_addr{};
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &addr_len);

        if (client_fd < 0) continue;

        LOG_DEBUG("VizServer", "新连接: fd=%d", client_fd);

        // 在独立线程中处理握手
        std::thread(&VizServer::HandleClient, this, client_fd).detach();
    }
}

// ---- 处理单个客户端连接 ----
void VizServer::HandleClient(int client_fd) {
    // 执行 WebSocket 握手
    if (!DoHandshake(client_fd)) {
        close(client_fd);
        return;
    }

    // 握手成功，加入客户端列表
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients_.push_back(client_fd);
    }

    LOG_DEBUG("VizServer", "WebSocket 握手成功: fd=%d", client_fd);

    // 保持连接，读取客户端消息（主要处理 close/ping 帧）
    unsigned char buf[1024];
    while (running_.load()) {
        struct pollfd pfd{};
        pfd.fd = client_fd;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 1000);
        if (ret <= 0) continue;

        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) break;  // 连接断开

        // 解析 WebSocket 帧 opcode
        if (n >= 2) {
            uint8_t opcode = buf[0] & 0x0F;
            if (opcode == 0x08) break;  // Close 帧
            if (opcode == 0x09) {
                // Ping 帧 → 回复 Pong
                buf[0] = (buf[0] & 0xF0) | 0x0A;
                send(client_fd, buf, static_cast<size_t>(n), 0);
            }
        }
    }

    // 移除客户端
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients_.erase(std::remove(clients_.begin(), clients_.end(), client_fd), clients_.end());
    }

    close(client_fd);
    LOG_DEBUG("VizServer", "客户端断开: fd=%d", client_fd);
}

// ---- WebSocket 握手 ----
bool VizServer::DoHandshake(int client_fd) {
    char buf[4096];
    ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return false;
    buf[n] = '\0';

    std::string request(buf);

    // 提取 Sec-WebSocket-Key
    std::string key_header = "Sec-WebSocket-Key: ";
    size_t key_pos = request.find(key_header);
    if (key_pos == std::string::npos) return false;

    size_t key_start = key_pos + key_header.size();
    size_t key_end = request.find("\r\n", key_start);
    if (key_end == std::string::npos) return false;

    std::string ws_key = request.substr(key_start, key_end - key_start);

    // 计算 Sec-WebSocket-Accept
    std::string magic = ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char sha1_hash[20];
    SHA1(reinterpret_cast<const unsigned char*>(magic.c_str()), magic.size(), sha1_hash);
    std::string accept_key = Base64Encode(sha1_hash, 20);

    // 构建响应
    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept_key + "\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";

    ssize_t sent = send(client_fd, response.c_str(), response.size(), 0);
    return sent == static_cast<ssize_t>(response.size());
}

// ---- 发送 WebSocket 文本帧 ----
bool VizServer::SendFrame(int client_fd, const std::string& data) {
    size_t len = data.size();
    std::vector<unsigned char> frame;

    // 帧头：FIN=1, opcode=0x01 (text)
    frame.push_back(0x81);

    // 负载长度编码
    if (len <= 125) {
        frame.push_back(static_cast<unsigned char>(len));
    } else if (len <= 65535) {
        frame.push_back(126);
        frame.push_back(static_cast<unsigned char>((len >> 8) & 0xFF));
        frame.push_back(static_cast<unsigned char>(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; --i) {
            frame.push_back(static_cast<unsigned char>((len >> (i * 8)) & 0xFF));
        }
    }

    // 负载数据
    frame.insert(frame.end(), data.begin(), data.end());

    ssize_t sent = send(client_fd, frame.data(), frame.size(), MSG_NOSIGNAL);
    return sent == static_cast<ssize_t>(frame.size());
}

// ---- 广播 JSON 数据到所有客户端 ----
void VizServer::Broadcast(const std::string& json_data) {
    std::lock_guard<std::mutex> lock(clients_mutex_);

    // 移除发送失败的客户端
    std::vector<int> failed;

    for (int fd : clients_) {
        if (!SendFrame(fd, json_data)) {
            failed.push_back(fd);
        }
    }

    for (int fd : failed) {
        close(fd);
        clients_.erase(std::remove(clients_.begin(), clients_.end(), fd), clients_.end());
    }
}

// ---- 查询连接数 ----
int VizServer::GetClientCount() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return static_cast<int>(clients_.size());
}
