"""
RL-Learner 训练主入口
启动 gRPC 服务接收 AIServer 样本，后台运行训练循环（Phase 3B 实现）
支持 --model_path 指定本地模型加载路径，无参数则自动生成空模型
"""

import argparse
import signal
import sys
import os
import time
import shutil
import glob
from concurrent import futures

import grpc
import yaml
import torch
import numpy as np

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

# ---- 清理模型缓存（训练启动时清理 P2P 目录下的旧模型）----
def clean_model_cache(config: dict, logger):
    """清理 P2P 模型目录下的旧模型文件"""
    p2p_dir = config.get("model", {}).get("p2p_dir", "models/p2p")

    if not os.path.exists(p2p_dir):
        os.makedirs(p2p_dir, exist_ok=True)
        logger.info("P2P 模型目录已创建: %s", p2p_dir)
        return

    # 清理 P2P 目录下的所有 .onnx 文件
    onnx_files = glob.glob(os.path.join(p2p_dir, "*.onnx"))
    if onnx_files:
        for f in onnx_files:
            os.remove(f)
            logger.info("清理旧模型: %s", f)
        logger.info("P2P 模型缓存已清理，共删除 %d 个文件", len(onnx_files))
    else:
        logger.info("P2P 模型目录无缓存文件: %s", p2p_dir)

# ---- 构建 Actor-Critic 网络（空模型，随机权重）----
def build_actor_critic(obs_dim: int, action_dim: int, hidden_dim: int):
    """构建简单的 Actor-Critic 网络"""
    class SimpleActorCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Policy 分支
            self.policy_encoder = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.policy_head = torch.nn.Linear(hidden_dim, action_dim)

            # Value 分支
            self.value_encoder = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.value_head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, obs):
            # Policy
            p = self.policy_encoder(obs)
            action_probs = torch.softmax(self.policy_head(p), dim=-1)
            # Value
            v = self.value_encoder(obs)
            value = self.value_head(v)
            return action_probs, value

    return SimpleActorCritic()

# ---- 导出 ONNX 模型 ----
def export_onnx_model(model, obs_dim: int, export_path: str, logger):
    """将 PyTorch 模型导出为 ONNX 格式（兼容 PyTorch 2.6+ 新版导出器）"""
    model.eval()
    dummy_input = torch.randn(1, obs_dim)

    export_kwargs = dict(
        input_names=["obs"],
        output_names=["action_probs", "value"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action_probs": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=11,
    )

    # PyTorch 2.6+ 默认走 dynamo 导出路径（依赖 onnxscript），
    # 简单网络使用 TorchScript 导出即可，通过 dynamo=False 回退
    try:
        torch.onnx.export(model, dummy_input, export_path, dynamo=False, **export_kwargs)
    except TypeError:
        # PyTorch < 2.6 不支持 dynamo 参数，直接调用旧版 API
        torch.onnx.export(model, dummy_input, export_path, **export_kwargs)

    logger.info("ONNX 模型已导出: %s", export_path)

# ---- 初始化模型（加载或生成空模型）----
def init_model(config: dict, model_path: str, logger):
    """
    初始化模型：
    - 若指定 model_path 且文件存在，复制到 local 目录并加载
    - 否则生成空模型（随机权重）并导出到 local 目录和 P2P 目录
    """
    model_cfg = config.get("model", {})
    export_dir = model_cfg.get("export_dir", "models/local")
    p2p_dir = model_cfg.get("p2p_dir", "models/p2p")
    save_name = model_cfg.get("save_name", "SaveModel")
    obs_dim = model_cfg.get("obs_dim", 5)
    action_dim = model_cfg.get("action_dim", 9)
    hidden_dim = model_cfg.get("hidden_dim", 64)

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(p2p_dir, exist_ok=True)

    local_onnx_path = os.path.join(export_dir, f"{save_name}.onnx")
    p2p_onnx_path = os.path.join(p2p_dir, f"{save_name}.onnx")

    if model_path and os.path.isfile(model_path):
        # 从指定路径加载模型
        logger.info("加载本地模型: %s", model_path)
        shutil.copy2(model_path, local_onnx_path)
        shutil.copy2(model_path, p2p_onnx_path)
        logger.info("模型已复制到: %s, %s", local_onnx_path, p2p_onnx_path)
        return local_onnx_path
    else:
        if model_path:
            logger.warning("指定的模型路径不存在: %s，将生成空模型", model_path)

        # 生成空模型
        logger.info("生成空模型 (obs_dim=%d, action_dim=%d, hidden=%d)", obs_dim, action_dim, hidden_dim)
        model = build_actor_critic(obs_dim, action_dim, hidden_dim)

        # 导出到 local 目录
        export_onnx_model(model, obs_dim, local_onnx_path, logger)

        # 同步到 P2P 目录（供 AIServer 拉取）
        shutil.copy2(local_onnx_path, p2p_onnx_path)
        logger.info("空模型已同步到 P2P 目录: %s", p2p_onnx_path)

        return local_onnx_path

# ---- 主入口 ----
def main():
    global _running

    parser = argparse.ArgumentParser(description="RL-Learner 训练服务")
    parser.add_argument("--config", type=str, default="configs/learner_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--model_path", type=str, default="",
                        help="本地模型加载路径（为空则自动生成空模型）")
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

    # ---- 3. 清理模型缓存（训练启动时清理 P2P 旧模型）----
    clean_model_cache(config, logger)

    # ---- 4. 初始化模型（加载或生成空模型）----
    try:
        model_onnx_path = init_model(config, args.model_path, logger)
        logger.info("模型初始化完成: %s", model_onnx_path)
    except Exception as e:
        logger.error("模型初始化失败: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        return

    # ---- 5. 创建样本缓存 ----
    max_size = buffer_cfg.get("max_size", 4096)
    sample_buffer = SampleBuffer(max_size=max_size)
    logger.info("样本缓存已创建，最大容量: %d", max_size)

    # ---- 6. 启动 gRPC 服务 ----
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

    # ---- 7. 主循环（Phase 3A：接收样本并统计，训练逻辑为空）----
    last_total = 0
    try:
        while _running:
            time.sleep(5.0)

            # 定期打印样本缓存状态
            buf_size = sample_buffer.size()
            total_received = sample_buffer.total_received()

            if total_received > last_total:
                logger.info("样本缓存: %d / %d | 累计接收: %d (+%d)",
                            buf_size, max_size, total_received, total_received - last_total)
                last_total = total_received

                # 训练逻辑占位：消费样本但不训练（验证数据流）
                consume_size = buffer_cfg.get("consume_batch_size", 256)
                if buf_size >= consume_size:
                    consumed = sample_buffer.consume(consume_size)
                    logger.info("[训练占位] 消费 %d 个样本（未执行实际训练）", len(consumed))
            elif total_received > 0:
                logger.info("样本缓存: %d / %d | 累计接收: %d（等待新样本）",
                            buf_size, max_size, total_received)
    except KeyboardInterrupt:
        pass

    # ---- 8. 优雅退出 ----
    logger.info("正在关闭 gRPC 服务...")
    grpc_server.stop(grace=5)
    logger.info("RL-Learner 已停止")

if __name__ == "__main__":
    main()
