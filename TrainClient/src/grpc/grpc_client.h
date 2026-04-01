#pragma once

#include "maze.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>

// ---- gRPC 客户端（Client↔AIServer 通信）----
class GrpcClient {
public:
    GrpcClient() = default;
    ~GrpcClient() = default;

    // 连接管理
    bool Connect(const std::string& host, int port);    // 创建 gRPC Channel 连接 AIServer
    void Disconnect();                                   // 断开连接，释放 Channel 和 Stub
    bool IsConnected() const;                            // 连接状态查询

    // RPC 调用（对应 MazeService 的三个方法）
    bool Init(const maze::InitReq& req, maze::InitRsp& rsp);                       // 初始化
    bool Update(const maze::UpdateReq& req, maze::UpdateRsp& rsp);                 // 帧同步
    bool EndEpisode(const maze::EpisodeEndReq& req, maze::EpisodeEndRsp& rsp);     // Episode 结束

private:
    std::shared_ptr<grpc::Channel> channel_;                // gRPC 通道
    std::unique_ptr<maze::MazeService::Stub> stub_;         // MazeService 存根
    bool connected_ = false;
};
