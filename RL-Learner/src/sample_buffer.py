"""
线程安全样本缓存（生产者-消费者模型）
接收 gRPC 线程推入的样本，供训练线程消费
利用 Condition 变量实现零轮询唤醒，无界队列保证零丢弃
"""

import threading
import time
from collections import deque


class SampleBuffer:
    """线程安全的无界样本缓存（零丢弃保证）"""

    def __init__(self, warn_threshold: int = 8192):
        """
        Args:
            warn_threshold: 拥塞告警水位线，缓冲区大小超过此值时触发告警
        """
        self._buffer = deque()                      # 无界队列，零丢弃保证
        self._cond = threading.Condition()           # 条件变量：替代 Lock，支持 wait/notify
        self._warn_threshold = warn_threshold

        # ---- 统计计数器 ----
        self._total_received = 0                     # 累计接收样本数
        self._total_consumed = 0                     # 累计消费样本数
        self._peak_size = 0                          # 历史峰值缓冲区大小
        self._warn_count = 0                         # 拥塞告警触发次数
        self._last_warn_time = 0.0                   # 上次告警时间（避免日志刷屏）

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

    # ---- 当前缓存大小 ----
    def size(self) -> int:
        with self._cond:
            return len(self._buffer)

    # ---- 累计接收样本数 ----
    def total_received(self) -> int:
        with self._cond:
            return self._total_received

    # ---- 累计消费样本数 ----
    def total_consumed(self) -> int:
        with self._cond:
            return self._total_consumed

    # ---- 检查是否处于拥塞状态（供外部日志使用）----
    def check_congestion(self) -> bool:
        """检查缓冲区是否超过告警水位线，超过时递增告警计数"""
        with self._cond:
            if len(self._buffer) > self._warn_threshold:
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
        with self._cond:
            return {
                "total_received": self._total_received,
                "total_consumed": self._total_consumed,
                "peak_size": self._peak_size,
                "warn_count": self._warn_count,
                "warn_threshold": self._warn_threshold,
                "current_size": len(self._buffer),
                "dropped": 0,                        # 无界队列，永远为 0
            }

    # ---- 清空缓存 ----
    def clear(self):
        with self._cond:
            self._buffer.clear()

    # ---- 内部：更新峰值并检测拥塞 ----
    def _update_peak(self):
        """更新峰值记录（调用方已持有锁）"""
        cur = len(self._buffer)
        if cur > self._peak_size:
            self._peak_size = cur
