"""
训练指标采集器

接收训练循环的原始指标，维护 Episode 滑动窗口统计（奖励、通关率等），
通过 MetricsBackend 持久化到存储层。

调用时机：
    - on_episode_end()  → 每个 Episode 结束时调用
    - on_train_step()   → 每个训练 batch 结束后调用
"""

import time
from collections import deque
from typing import Optional

from src.metrics_backend import MetricsBackend


class MetricsCollector:
    """训练指标采集器"""

    def __init__(self, backend: MetricsBackend, window_size: int = 100):
        """
        Args:
            backend: 存储后端实例
            window_size: Episode 指标滑动窗口大小（计算平均奖励/通关率的窗口）
        """
        self._backend = backend
        self._window_size = window_size

        # Episode 滑动窗口
        self._episode_rewards = deque(maxlen=window_size)   # Episode 总奖励
        self._episode_lengths = deque(maxlen=window_size)   # Episode 帧数
        self._episode_passed = deque(maxlen=window_size)    # 是否通关（0/1）
        self._episode_count = 0                              # 累计完成 Episode 数

        # 时间统计
        self._start_time = time.time()
        self._last_consumed = 0
        self._last_throughput_time = time.time()

    def on_episode_end(self, total_reward: float, length: int, passed: bool):
        """
        Episode 结束时调用，更新滑动窗口

        Args:
            total_reward: 该 Episode 的总奖励
            length: 该 Episode 的帧数
            passed: 是否通关
        """
        self._episode_rewards.append(total_reward)
        self._episode_lengths.append(length)
        self._episode_passed.append(1.0 if passed else 0.0)
        self._episode_count += 1

    def on_train_step(self, step: int, train_stats: dict, buffer_stats: dict):
        """
        每个训练 batch 结束后调用，记录完整指标

        Args:
            step: 当前训练步数
            train_stats: 训练核心指标（policy_loss, value_loss, entropy 等）
            buffer_stats: 缓冲区统计（来自 SampleBuffer.get_stats()）
        """
        now = time.time()

        # 计算样本吞吐率（每秒消费样本数）
        total_consumed = buffer_stats.get("total_consumed", 0)
        dt = now - self._last_throughput_time
        if dt > 0 and self._last_consumed > 0:
            samples_per_sec = (total_consumed - self._last_consumed) / dt
        else:
            samples_per_sec = 0.0
        self._last_consumed = total_consumed
        self._last_throughput_time = now

        record = {
            # 时间信息
            "timestamp": now,
            "elapsed": now - self._start_time,
            "train_step": step,

            # 训练核心指标
            "policy_loss": train_stats.get("policy_loss", 0.0),
            "value_loss": train_stats.get("value_loss", 0.0),
            "total_loss": train_stats.get("total_loss", 0.0),
            "entropy": train_stats.get("entropy", 0.0),
            "clip_fraction": train_stats.get("clip_fraction", 0.0),
            "mean_advantage": train_stats.get("mean_advantage", 0.0),
            "learning_rate": train_stats.get("learning_rate", 0.0),

            # Episode 效果指标（滑动窗口聚合）
            "episode_count": self._episode_count,
            "mean_episode_reward": self._safe_mean(self._episode_rewards),
            "mean_episode_length": self._safe_mean(self._episode_lengths),
            "pass_rate": self._safe_mean(self._episode_passed),

            # 分布式系统指标
            "buffer_size": buffer_stats.get("current_size", 0),
            "buffer_peak": buffer_stats.get("peak_size", 0),
            "total_received": buffer_stats.get("total_received", 0),
            "total_consumed": buffer_stats.get("total_consumed", 0),
            "congestion_warns": buffer_stats.get("warn_count", 0),
            "samples_per_sec": round(samples_per_sec, 1),
            "model_version": train_stats.get("model_version", 0),
        }

        self._backend.write(record)

    def get_episode_count(self) -> int:
        """返回累计完成的 Episode 数"""
        return self._episode_count

    def close(self):
        """关闭采集器，释放后端资源"""
        self._backend.close()

    @staticmethod
    def _safe_mean(data: deque) -> float:
        """安全计算平均值，空序列返回 0.0"""
        if len(data) == 0:
            return 0.0
        return round(sum(data) / len(data), 4)
