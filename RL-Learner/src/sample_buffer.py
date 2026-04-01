"""
线程安全样本缓存（生产者-消费者模型）
接收 gRPC 线程推入的样本，供训练线程消费
支持两种模式：
  - 扁平模式：push_batch() / consume()（Phase 3A 兼容）
  - Trajectory 模式：push_trajectory() / get_ready_trajectories()（Phase 3B PPO 训练）
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

        # ---- Trajectory 模式（Phase 3B PPO 训练）----
        # key=(episode_id, agent_id), value={"samples": list, "complete": bool}
        self._trajectories = {}
        self._completed_keys = deque()               # 已完成的 trajectory key 队列（FIFO 消费）
        self._traj_cond = threading.Condition()      # trajectory 专用条件变量
        self._total_traj_completed = 0               # 累计完成的 trajectory 数

        # ---- 统计计数器 ----
        self._total_received = 0                     # 累计接收样本数
        self._total_consumed = 0                     # 累计消费样本数
        self._peak_size = 0                          # 历史峰值缓冲区大小
        self._warn_count = 0                         # 拥塞告警触发次数
        self._last_warn_time = 0.0                   # 上次告警时间（避免日志刷屏）

    # ========================================================================
    #  Trajectory 模式接口（Phase 3B PPO 训练使用）
    # ========================================================================

    # ---- 推入一条 trajectory 片段 ----
    def push_trajectory(self, episode_id: int, agent_id: int, samples: list, is_episode_end: bool):
        """
        推入一条 trajectory 片段，按 (episode_id, agent_id) 组织

        同一 (episode_id, agent_id) 可能收到多个 SampleBatch（每 TMax=32 帧一批），
        本方法将它们拼接为完整 trajectory。当 is_episode_end=True 时标记完成。

        Args:
            episode_id: Episode 编号
            agent_id: Agent 编号
            samples: 样本 dict 列表
            is_episode_end: True=Episode 结束，False=TMax 截断（后续还有数据）
        """
        key = (episode_id, agent_id)

        with self._traj_cond:
            # 追加到对应 trajectory
            if key not in self._trajectories:
                self._trajectories[key] = {"samples": [], "complete": False}

            self._trajectories[key]["samples"].extend(samples)

            # 更新统计
            self._total_received += len(samples)

            # Episode 结束时标记完成
            if is_episode_end:
                self._trajectories[key]["complete"] = True
                self._completed_keys.append(key)
                self._total_traj_completed += 1
                self._traj_cond.notify()             # 唤醒训练线程

            # 拥塞检测（基于活跃 trajectory 中的总样本数）
            total_samples = sum(len(t["samples"]) for t in self._trajectories.values())
            if total_samples > self._peak_size:
                self._peak_size = total_samples

    # ---- 获取已完成的 trajectory 列表 ----
    def get_ready_trajectories(self, min_samples: int = 256, timeout: float = 1.0) -> List[Tuple]:
        """
        获取已完成的 trajectory 列表，直到累计样本数 >= min_samples

        Args:
            min_samples: 最少需要的样本数（不强制凑满，有多少取多少）
            timeout: 等待超时（秒），无数据时阻塞等待
        Returns:
            [(key, samples, is_episode_end), ...] 列表，空列表表示超时无数据
        """
        with self._traj_cond:
            # 无已完成 trajectory 时等待
            while len(self._completed_keys) == 0:
                if not self._traj_cond.wait(timeout):
                    return []                        # 超时，返回空让主循环检查退出标志

            # 取出已完成的 trajectory，直到累计样本数 >= min_samples
            result = []
            total_count = 0

            while self._completed_keys and total_count < min_samples:
                key = self._completed_keys.popleft()
                traj_data = self._trajectories.pop(key, None)
                if traj_data is None:
                    continue

                samples = traj_data["samples"]
                total_count += len(samples)
                self._total_consumed += len(samples)
                # trajectory 模式下取出的都是 complete=True 的
                result.append((key, samples, True))

            return result

    # ---- Trajectory 统计 ----
    def trajectory_stats(self) -> dict:
        """返回 trajectory 组织层的统计信息"""
        with self._traj_cond:
            active_count = sum(1 for t in self._trajectories.values() if not t["complete"])
            pending_count = len(self._completed_keys)
            active_samples = sum(len(t["samples"]) for t in self._trajectories.values())
            return {
                "active_trajectories": active_count,       # 正在接收中的 trajectory 数
                "pending_completed": pending_count,         # 已完成待消费的 trajectory 数
                "total_completed": self._total_traj_completed,  # 累计完成的 trajectory 数
                "active_samples": active_samples,           # 活跃 trajectory 中的总样本数
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
            total_samples = sum(len(t["samples"]) for t in self._trajectories.values())
            if total_samples > self._warn_threshold:
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
            "active_trajectories": traj_stats["active_trajectories"],
            "pending_completed": traj_stats["pending_completed"],
            "total_completed": traj_stats["total_completed"],
        }

    # ---- 清空缓存 ----
    def clear(self):
        with self._cond:
            self._buffer.clear()
        with self._traj_cond:
            self._trajectories.clear()
            self._completed_keys.clear()

    # ---- 内部：更新峰值并检测拥塞 ----
    def _update_peak(self):
        """更新峰值记录（调用方已持有锁）"""
        cur = len(self._buffer)
        if cur > self._peak_size:
            self._peak_size = cur
