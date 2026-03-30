"""
RL-Learner 训练主入口
启动 gRPC 服务接收 AIServer 样本，后台运行训练循环（Phase 3B 实现）
"""

import argparse
import signal
import sys
import os
import time
from concurrent import futures

import grpc
import yaml

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import maze_pb2_grpc
from src.learner_service import LearnerServiceImpl
from src.sample_buffer import SampleBuffer
from src.logger import setup_logger


# --- 全局退出标志 ---
_running = True


# ---- 信号处理（优雅退出）----
def signal_handler(sig, frame):
    global _running
    logger = setup_logger("Main")
    logger.info("收到信号 %d，准备退出...", sig)
    _running = False


# ---- 加载配置文件 ----
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- 主入口 ----
def main():
    global _running

    parser = argparse.ArgumentParser(description="RL-Learner 训练服务")
    parser.add_argument("--config", type=str, default="configs/learner_config.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    # ---- 0. 加载配置 ----
    config = load_config(args.config)
    server_cfg = config.get("server", {})
    buffer_cfg = config.get("buffer", {})
    log_cfg = config.get("log", {})

    # ---- 1. 初始化日志 ----
    log_dir = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        "Main",
        console_level=log_cfg.get("console_level", "INFO"),
        file_level=log_cfg.get("file_level", "DEBUG"),
        log_dir=log_dir,
    )

    logger.info("============================================")
    logger.info("  迷宫训练框架 - RL-Learner")
    logger.info("============================================")

    # ---- 2. 注册信号处理 ----
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ---- 3. 创建样本缓存 ----
    max_size = buffer_cfg.get("max_size", 4096)
    sample_buffer = SampleBuffer(max_size=max_size)
    logger.info("样本缓存已创建，最大容量: %d", max_size)

    # ---- 4. 启动 gRPC 服务 ----
    listen_port = server_cfg.get("listen_port", 9003)
    max_workers = server_cfg.get("max_workers", 4)
    listen_addr = f"0.0.0.0:{listen_port}"

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    service = LearnerServiceImpl(sample_buffer, config)
    maze_pb2_grpc.add_LearnerServiceServicer_to_server(service, grpc_server)
    grpc_server.add_insecure_port(listen_addr)
    grpc_server.start()

    logger.info("Learner gRPC 服务已启动，监听: %s", listen_addr)
    logger.info("等待 AIServer 发送样本...")

    # ---- 5. 主循环（Phase 3A：仅接收样本并统计，Phase 3B 加入训练逻辑）----
    try:
        while _running:
            time.sleep(5.0)

            # 定期打印样本缓存状态
            buf_size = sample_buffer.size()
            total_received = sample_buffer.total_received()
            if total_received > 0:
                logger.info("样本缓存: %d / %d | 累计接收: %d",
                            buf_size, max_size, total_received)
    except KeyboardInterrupt:
        pass

    # ---- 6. 优雅退出 ----
    logger.info("正在关闭 gRPC 服务...")
    grpc_server.stop(grace=5)
    logger.info("RL-Learner 已停止")


if __name__ == "__main__":
    main()
