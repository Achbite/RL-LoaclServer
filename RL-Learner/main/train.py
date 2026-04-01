"""
RL-Learner 训练主入口
启动 gRPC 服务接收 AIServer 样本，运行 PPO 训练循环
支持 --model_path 指定本地 checkpoint 加载路径，无参数则自动生成空模型
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
from src.ppo_trainer import PPOTrainer
from src.logger import setup_logger
from src.metrics_backend import create_backend
from src.metrics_collector import MetricsCollector

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

# ---- 初始化 PPOTrainer（统一模型管理）----
def init_trainer(config: dict, model_path: str, logger) -> PPOTrainer:
    """
    初始化 PPOTrainer 并导出初始 ONNX 模型

    Args:
        config: 完整配置字典
        model_path: 本地 checkpoint 路径（为空则使用随机权重）
        logger: 日志实例
    Returns:
        PPOTrainer 实例
    """
    model_cfg = config.get("model", {})
    export_dir = model_cfg.get("export_dir", "models/local")
    p2p_dir = model_cfg.get("p2p_dir", "models/p2p")
    save_name = model_cfg.get("save_name", "SaveModel")

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(p2p_dir, exist_ok=True)

    # 构建 PPOTrainer（内部创建 ActorCritic 网络 + Adam 优化器）
    trainer = PPOTrainer(config)

    # 加载 checkpoint（如果指定了路径）
    if model_path and os.path.isfile(model_path):
        logger.info("加载 checkpoint: %s", model_path)
        trainer.load_checkpoint(model_path)
    elif model_path:
        logger.warning("指定的 checkpoint 路径不存在: %s，使用随机权重", model_path)

    # 导出初始 ONNX 模型到 local 和 P2P 目录
    local_onnx_path = os.path.join(export_dir, f"{save_name}.onnx")
    p2p_onnx_path = os.path.join(p2p_dir, f"{save_name}.onnx")

    trainer.export_onnx(local_onnx_path)
    shutil.copy2(local_onnx_path, p2p_onnx_path)
    logger.info("初始模型已同步到 P2P 目录: %s", p2p_onnx_path)

    return trainer

# ---- 主入口 ----
def main():
    global _running

    parser = argparse.ArgumentParser(description="RL-Learner 训练服务")
    parser.add_argument("--config", type=str, default="configs/learner_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--model_path", type=str, default="",
                        help="本地 checkpoint 加载路径（为空则自动生成空模型）")
    args = parser.parse_args()

    # ---- 0. 加载配置 ----
    config = load_config(args.config)
    server_cfg = config.get("server", {})
    buffer_cfg = config.get("buffer", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    log_cfg = config.get("log", {})
    dashboard_cfg = config.get("dashboard", {})

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

    # ---- 4. 初始化 PPOTrainer（统一管理模型 + 优化器）----
    try:
        trainer = init_trainer(config, args.model_path, logger)
        logger.info("PPOTrainer 初始化完成，模型版本: %d", trainer.model_version)
    except Exception as e:
        logger.error("PPOTrainer 初始化失败: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        return

    # ---- 5. 创建样本缓存 ----
    warn_threshold = buffer_cfg.get("warn_threshold", 8192)
    sample_buffer = SampleBuffer(warn_threshold=warn_threshold)
    logger.info("样本缓存已创建，无界队列（零丢弃），拥塞告警水位线: %d", warn_threshold)

    # ---- 6. 初始化训练指标采集器 ----
    metrics_collector = None
    if dashboard_cfg.get("enabled", True):
        metrics_dir = dashboard_cfg.get("metrics_dir", "logs/metrics")
        backend_type = dashboard_cfg.get("backend", "jsonl")
        window_size = dashboard_cfg.get("window_size", 100)
        backend = create_backend(backend_type, metrics_dir)
        metrics_collector = MetricsCollector(backend, window_size=window_size)
        logger.info("训练指标采集器已启动，存储后端: %s，指标目录: %s", backend_type, metrics_dir)
    else:
        logger.info("训练指标采集器已禁用")

    # ---- 7. 启动 gRPC 服务 ----
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

    # ---- 8. PPO 训练主循环 ----
    consume_size = buffer_cfg.get("consume_batch_size", 256)
    consume_timeout = buffer_cfg.get("consume_timeout", 1.0)
    export_interval = model_cfg.get("export_interval", 10)
    save_name = model_cfg.get("save_name", "SaveModel")
    export_dir = model_cfg.get("export_dir", "models/local")
    p2p_dir = model_cfg.get("p2p_dir", "models/p2p")
    pass_reward_threshold = train_cfg.get("pass_reward_threshold", 10.0)

    local_onnx_path = os.path.join(export_dir, f"{save_name}.onnx")
    p2p_onnx_path = os.path.join(p2p_dir, f"{save_name}.onnx")

    last_log_time = time.time()
    train_step = 0

    try:
        while _running:
            # ---- 8.1 等待 trajectory 积累 ----
            ready = sample_buffer.get_ready_trajectories(
                min_samples=consume_size, timeout=consume_timeout
            )

            if not ready:
                # 超时无数据，打印状态后继续等待
                now = time.time()
                if now - last_log_time >= 5.0:
                    total_received = sample_buffer.total_received()
                    if total_received > 0:
                        traj_stats = sample_buffer.trajectory_stats()
                        logger.info(
                            "等待 trajectory | 累计接收: %d | 活跃 traj: %d | 待消费: %d",
                            total_received, traj_stats["active_trajectories"],
                            traj_stats["pending_completed"],
                        )
                    last_log_time = now
                continue

            # ---- 8.2 GAE 计算 + Episode 级多 Agent 平均指标 ----
            all_samples = []
            # 当前批次 Episode 统计（用于即时反映当前模型效果）
            batch_episode_rewards = []          # 本批次各 Episode 的平均总奖励
            batch_reward_breakdowns = []        # 本批次各 Episode 的平均奖励分项 dict

            # 按 episode_id 聚合同一 Episode 的多个 Agent trajectory
            episode_agent_data = {}             # episode_id → [{"total_reward", "passed", "length", "breakdown"}, ...]

            for traj_key, samples, is_ep_end in ready:
                ep_id, agent_id = traj_key

                # 跳过空 trajectory（Episode 结束信号但无样本数据）
                if not samples:
                    continue

                # 对每条 trajectory 计算 GAE
                trainer.compute_gae(samples, is_ep_end)
                all_samples.extend(samples)

                # 收集 Agent 级指标（仅对已完成的 Episode）
                if is_ep_end:
                    total_reward = sum(s["reward"] for s in samples)
                    passed = samples[-1]["reward"] >= pass_reward_threshold if samples else False

                    # 聚合奖励分项（按 Agent 累计每个奖励组件的值）
                    reward_breakdown = {}
                    for s in samples:
                        for rk, val in s.get("reward_details", {}).items():
                            reward_breakdown[rk] = reward_breakdown.get(rk, 0.0) + val

                    agent_data = {
                        "total_reward": total_reward,
                        "passed": passed,
                        "length": len(samples),
                        "breakdown": reward_breakdown,
                    }
                    episode_agent_data.setdefault(ep_id, []).append(agent_data)

            # 对每个 Episode 计算 Agent 平均指标
            for ep_id, agents_data in episode_agent_data.items():
                n = len(agents_data)
                avg_reward = sum(a["total_reward"] for a in agents_data) / n
                avg_length = sum(a["length"] for a in agents_data) / n
                # 通关率：通关 Agent 数 / 总 Agent 数
                avg_passed = sum(1.0 if a["passed"] else 0.0 for a in agents_data) / n

                # 各奖励分项取 Agent 平均
                all_keys = set()
                for a in agents_data:
                    all_keys.update(a["breakdown"].keys())
                avg_breakdown = {}
                for key in all_keys:
                    vals = [a["breakdown"].get(key, 0.0) for a in agents_data]
                    avg_breakdown[key] = sum(vals) / n

                if metrics_collector is not None:
                    metrics_collector.on_episode_end(
                        avg_reward, int(avg_length), avg_passed > 0.5, avg_breakdown
                    )

                # 收集当前批次 Episode 统计
                batch_episode_rewards.append(avg_reward)
                batch_reward_breakdowns.append(avg_breakdown)

            # ---- 8.3 PPO 训练 ----
            train_step += 1
            stats = trainer.train_on_batch(all_samples)

            logger.info(
                "[训练] 步骤 %d | 样本 %d | policy=%.4f | value=%.4f | entropy=%.4f | clip=%.3f | v%d",
                train_step, len(all_samples),
                stats["policy_loss"], stats["value_loss"],
                stats["entropy"], stats["clip_fraction"],
                stats["model_version"],
            )

            # ---- 8.4 记录训练指标 ----
            if metrics_collector is not None:
                # 构建当前批次 Episode 统计（即时反映当前模型版本的训练效果）
                batch_stats = {}
                if batch_episode_rewards:
                    batch_stats["batch_episode_count"] = len(batch_episode_rewards)
                    batch_stats["batch_mean_reward"] = round(
                        sum(batch_episode_rewards) / len(batch_episode_rewards), 4
                    )
                    # 计算各奖励分项的批次平均值
                    merged_breakdown = {}
                    for bd in batch_reward_breakdowns:
                        for rk, val in bd.items():
                            merged_breakdown.setdefault(rk, []).append(val)
                    batch_stats["batch_reward_breakdown"] = {
                        rk: round(sum(vals) / len(vals), 4)
                        for rk, vals in merged_breakdown.items()
                    }
                metrics_collector.set_batch_episode_stats(batch_stats)
                metrics_collector.on_train_step(train_step, stats, sample_buffer.get_stats())

            # ---- 8.5 定期导出 ONNX 模型 ----
            if train_step % export_interval == 0:
                trainer.export_onnx(local_onnx_path)
                shutil.copy2(local_onnx_path, p2p_onnx_path)
                service.update_model_version(trainer.model_version)
                logger.info("模型已导出并同步到 P2P: v%d → %s", trainer.model_version, p2p_onnx_path)

            # ---- 8.6 定期打印缓存状态 ----
            now = time.time()
            if now - last_log_time >= 5.0:
                traj_stats = sample_buffer.trajectory_stats()
                logger.info(
                    "缓存状态 | 活跃 traj: %d | 待消费: %d | 累计完成: %d | 累计接收: %d",
                    traj_stats["active_trajectories"], traj_stats["pending_completed"],
                    traj_stats["total_completed"], sample_buffer.total_received(),
                )
                last_log_time = now

    except KeyboardInterrupt:
        pass

    # ---- 9. 关闭指标采集器 ----
    if metrics_collector is not None:
        metrics_collector.close()
        logger.info("训练指标采集器已关闭")

    # ---- 10. 保存最终 checkpoint + 导出最终模型 ----
    try:
        checkpoint_path = os.path.join(export_dir, f"{save_name}_final.pt")
        trainer.save_checkpoint(checkpoint_path)
        trainer.export_onnx(local_onnx_path)
        shutil.copy2(local_onnx_path, p2p_onnx_path)
        logger.info("最终模型已保存: checkpoint=%s, onnx=%s", checkpoint_path, p2p_onnx_path)
    except Exception as e:
        logger.error("最终模型保存失败: %s", str(e))

    # ---- 11. 训练结束汇总 ----
    stats = sample_buffer.get_stats()
    logger.info("============================================")
    logger.info("  训练结束汇总")
    logger.info("============================================")
    logger.info("  训练步数: %d", train_step)
    logger.info("  模型版本: %d", trainer.model_version)
    logger.info("  累计接收样本: %d", stats["total_received"])
    logger.info("  累计消费样本: %d", stats["total_consumed"])
    logger.info("  峰值缓冲区大小: %d", stats["peak_size"])
    logger.info("  累计完成 trajectory: %d", stats.get("total_completed", 0))
    logger.info("  拥塞告警次数: %d", stats["warn_count"])
    logger.info("  样本丢失: %d", stats["dropped"])
    if stats["warn_count"] > 0:
        logger.warning("⚠ 检测到 %d 次缓冲区拥塞（告警水位线 %d），"
                       "AIServer 样本产生速度超过 Learner 消费速度",
                       stats["warn_count"], stats["warn_threshold"])
        logger.warning("  建议：增加 Learner 训练并行度 或 减少 AIServer 并行 Episode 数")

    if metrics_collector is not None:
        ep_count = metrics_collector.get_episode_count()
        logger.info("  累计完成 Episode: %d", ep_count)

    # ---- 12. 优雅退出 ----
    logger.info("正在关闭 gRPC 服务...")
    grpc_server.stop(grace=5)
    logger.info("RL-Learner 已停止")

if __name__ == "__main__":
    main()
