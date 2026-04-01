"""
线程安全样本缓存（生产者-消费者模型）
接收 gRPC 线程推入的样本，供训练线程消费
支持两种模式：
  - 扁平模式：push_batch() / consume()（Phase 3A 兼容）
  - Trajectory 模式：push_trajectory() / get_ready_trajectories()（Phase 3B PPO 训练）
    方案 A：TMax 截断即消费 —— 每收到一个 TMax 片段立即标记可消费，
    不再等待 Episode 完整结束，大幅降低样本延迟和缓冲区堆积。
"""

import threading
import time
from collections import deque
from typing import List, Tuple, Optional


class SampleBuffer:
    """线程安全的无界样本缓存（零丢弃保证）"""

    def __init__(self, warn_threshold: int = 8192):
        """
        Args:
            warn_threshold: 拥塞告警水位线，缓冲区大小超过此值时触发告警
        """
        # ---- 扁平模式（Phase 3A 兼容）----
        self._buffer = deque()                      # 无界队列，零丢弃保证
        self._cond = threading.Condition()           # 条件变量：替代 Lock，支持 wait/notify
        self._warn_threshold = warn_threshold

        # ---- Trajectory 模式（Phase 3B PPO 训练 — 方案 A：TMax 即消费）----
        # 就绪片段队列：每个元素 = (traj_key, samples, is_episode_end)
        # traj_key = (episode_id, agent_id)
        self._ready_fragments = deque()
        self._traj_cond = threading.Condition()      # trajectory 专用条件变量
        self._total_traj_completed = 0               # 累计完成的 Episode trajectory 数
        self._total_fragments = 0                    # 累计接收的片段数（含 TMax 截断和 Episode 结束）
        self._active_fragment_samples = 0            # 就绪队列中待消费的样本总数

        # ---- 统计计数器 ----
        self._total_received = 0                     # 累计接收样本数
        self._total_consumed = 0                     # 累计消费样本数
        self._peak_size = 0                          # 历史峰值缓冲区大小
        self._warn_count = 0                         # 拥塞告警触发次数
        self._last_warn_time = 0.0                   # 上次告警时间（避免日志刷屏）

    # ========================================================================
    #  Trajectory 模式接口（Phase 3B PPO 训练 — 方案 A：TMax 即消费）
    # ========================================================================

    # ---- 推入一条 trajectory 片段 ----
    def push_trajectory(self, episode_id: int, agent_id: int, samples: list, is_episode_end: bool):
        """
        推入一条 trajectory 片段，立即标记为可消费（方案 A：TMax 即消费）

        每次收到 TMax=32 帧的片段就直接放入就绪队列，训练线程可以立即取走。
        不再等待 Episode 完整结束，大幅降低样本延迟。

        Args:
            episode_id: Episode 编号
            agent_id: Agent 编号
            samples: 样本 dict 列表
            is_episode_end: True=Episode 结束（GAE 用 V=0），False=TMax 截断（GAE 用 Bootstrap）
        """
        key = (episode_id, agent_id)

        with self._traj_cond:
            # 直接放入就绪队列（不再缓存拼接）
            self._ready_fragments.append((key, samples, is_episode_end))
            self._total_fragments += 1

            # 更新统计
            self._total_received += len(samples)
            self._active_fragment_samples += len(samples)

            if is_episode_end:
                self._total_traj_completed += 1

            # 峰值追踪
            if self._active_fragment_samples > self._peak_size:
                self._peak_size = self._active_fragment_samples

            # 唤醒训练线程
            self._traj_cond.notify()

    # ---- 获取就绪的 trajectory 片段列表 ----
    def get_ready_trajectories(self, min_samples: int = 256, timeout: float = 1.0) -> List[Tuple]:
        """
        获取就绪的 trajectory 片段列表，直到累计样本数 >= min_samples

        方案 A 下，每个 TMax 片段都是独立可消费的，不再等待 Episode 完整结束。

        Args:
            min_samples: 最少需要的样本数（不强制凑满，有多少取多少）
            timeout: 等待超时（秒），无数据时阻塞等待
        Returns:
            [(traj_key, samples, is_episode_end), ...] 列表，空列表表示超时无数据
        """
        with self._traj_cond:
            # 无就绪片段时等待
            while len(self._ready_fragments) == 0:
                if not self._traj_cond.wait(timeout):
                    return []                        # 超时，返回空让主循环检查退出标志

            # 取出就绪片段，直到累计样本数 >= min_samples
            result = []
            total_count = 0

            while self._ready_fragments and total_count < min_samples:
                key, samples, is_ep_end = self._ready_fragments.popleft()

                total_count += len(samples)
                self._total_consumed += len(samples)
                self._active_fragment_samples -= len(samples)
                result.append((key, samples, is_ep_end))

            return result

    # ---- Trajectory 统计 ----
    def trajectory_stats(self) -> dict:
        """返回 trajectory 组织层的统计信息"""
        with self._traj_cond:
            return {
                "pending_fragments": len(self._ready_fragments),    # 待消费的片段数
                "active_samples": self._active_fragment_samples,    # 待消费的样本总数
                "total_completed": self._total_traj_completed,      # 累计完成的 Episode trajectory 数
                "total_fragments": self._total_fragments,           # 累计接收的片段数
            }

    # ========================================================================
    #  扁平模式接口（Phase 3A 兼容，保留但 Phase 3B 不再使用）
    # ========================================================================

    # ---- 推入单个样本 ----
    def push(self, sample: dict):
        with self._cond:
            self._buffer.append(sample)
            self._total_received += 1
            self._update_peak()
            self._cond.notify()                      # 唤醒消费者

    # ---- 推入一批样本 ----
    def push_batch(self, samples: list):
        with self._cond:
            self._buffer.extend(samples)
            self._total_received += len(samples)
            self._update_peak()
            self._cond.notify()                      # 唤醒消费者

    # ---- 消费指定数量的样本（FIFO，阻塞等待）----
    def consume(self, batch_size: int, timeout: float = 1.0) -> list:
        """
        消费最多 batch_size 个样本。
        - 有数据时立即返回（不要求凑满 batch_size）
        - 无数据时阻塞等待，超时返回空列表（让主循环检查退出标志）

        Args:
            batch_size: 最大消费数量
            timeout: 等待超时（秒），超时返回空列表
        Returns:
            样本列表（可能少于 batch_size）
        """
        with self._cond:
            # 无数据时等待生产者唤醒
            while len(self._buffer) == 0:
                if not self._cond.wait(timeout):
                    return []                        # 超时，返回空让主循环检查退出标志

            actual = min(batch_size, len(self._buffer))
            batch = [self._buffer.popleft() for _ in range(actual)]
            self._total_consumed += actual
            return batch

    # ========================================================================
    #  公共接口
    # ========================================================================

    # ---- 当前缓存大小 ----
    def size(self) -> int:
        with self._cond:
            return len(self._buffer)

    # ---- 累计接收样本数 ----
    def total_received(self) -> int:
        # trajectory 模式下统计在 _traj_cond 锁内更新，这里直接返回
        return self._total_received

    # ---- 累计消费样本数 ----
    def total_consumed(self) -> int:
        return self._total_consumed

    # ---- 检查是否处于拥塞状态（供外部日志使用）----
    def check_congestion(self) -> bool:
        """检查缓冲区是否超过告警水位线，超过时递增告警计数"""
        with self._traj_cond:
            if self._active_fragment_samples > self._warn_threshold:
                now = time.time()
                # 限流：同一告警至少间隔 10 秒
                if now - self._last_warn_time >= 10.0:
                    self._warn_count += 1
                    self._last_warn_time = now
                    return True
            return False

    # ---- 获取统计汇总（训练结束时调用）----
    def get_stats(self) -> dict:
        """返回缓冲区运行统计，用于训练结束汇总报告"""
        traj_stats = self.trajectory_stats()
        return {
            "total_received": self._total_received,
            "total_consumed": self._total_consumed,
            "peak_size": self._peak_size,
            "warn_count": self._warn_count,
            "warn_threshold": self._warn_threshold,
            "current_size": traj_stats["active_samples"],
            "dropped": 0,                            # 无界队列，永远为 0
            # trajectory 统计
            "pending_fragments": traj_stats["pending_fragments"],
            "total_completed": traj_stats["total_completed"],
            "total_fragments": traj_stats["total_fragments"],
        }

    # ---- 清空缓存 ----
    def clear(self):
        with self._cond:
            self._buffer.clear()
        with self._traj_cond:
            self._ready_fragments.clear()
            self._active_fragment_samples = 0

    # ---- 内部：更新峰值并检测拥塞 ----
    def _update_peak(self):
        """更新峰值记录（调用方已持有锁）"""
        cur = len(self._buffer)
        if cur > self._peak_size:
            self._peak_size = cur
