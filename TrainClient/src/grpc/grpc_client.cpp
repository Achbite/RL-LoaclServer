#include "grpc/grpc_client.h"
#include "log/logger.h"

// ---- 创建 gRPC Channel 连接 AIServer ----
bool GrpcClient::Connect(const std::string& host, int port) {
    std::string target = host + ":" + std::to_string(port);

    // 创建不安全通道（内网通信，无需 TLS）
    channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = maze::MazeService::NewStub(channel_);

    // 等待通道就绪（最多 5 秒）
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    bool ok = channel_->WaitForConnected(deadline);

    if (ok) {
        connected_ = true;
        LOG_INFO("GrpcClient", "已连接 %s", target.c_str());
    } else {
        LOG_ERROR("GrpcClient", "连接 %s 超时", target.c_str());
    }

    return ok;
}

// ---- 连接状态查询 ----
bool GrpcClient::IsConnected() const {
    return connected_;
}

// ---- 初始化 RPC ----
bool GrpcClient::Init(const maze::InitReq& req, maze::InitRsp& rsp) {
    grpc::ClientContext context;
    grpc::Status status = stub_->Init(&context, req, &rsp);

    if (!status.ok()) {
        LOG_ERROR("GrpcClient", "Init RPC 失败: %s", status.error_message().c_str());
        return false;
    }
    return true;
}

// ---- 帧同步 RPC ----
bool GrpcClient::Update(const maze::UpdateReq& req, maze::UpdateRsp& rsp) {
    grpc::ClientContext context;
    grpc::Status status = stub_->Update(&context, req, &rsp);

    if (!status.ok()) {
        LOG_ERROR("GrpcClient", "Update RPC 失败: %s", status.error_message().c_str());
        return false;
    }
    return true;
}

// ---- Episode 结束 RPC ----
bool GrpcClient::EndEpisode(const maze::EpisodeEndReq& req, maze::EpisodeEndRsp& rsp) {
    grpc::ClientContext context;
    grpc::Status status = stub_->EndEpisode(&context, req, &rsp);

    if (!status.ok()) {
        LOG_ERROR("GrpcClient", "EndEpisode RPC 失败: %s", status.error_message().c_str());
        return false;
    }
    return true;
}

// ---- 断开连接，释放 Channel 和 Stub ----
void GrpcClient::Disconnect() {
    if (!connected_) return;

    stub_.reset();
    channel_.reset();
    connected_ = false;
}
