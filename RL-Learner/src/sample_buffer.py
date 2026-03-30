"""
线程安全样本缓存
接收 gRPC 线程推入的样本，供训练线程消费
"""

import threading
from collections import deque


class SampleBuffer:
    """线程安全的环形样本缓存"""

    def __init__(self, max_size: int = 4096):
        self._max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._total_received = 0          # 累计接收样本数（不受 maxlen 限制）

    # ---- 推入单个样本 ----
    def push(self, sample: dict):
        with self._lock:
            self._buffer.append(sample)
            self._total_received += 1

    # ---- 推入一批样本 ----
    def push_batch(self, samples: list):
        with self._lock:
            self._buffer.extend(samples)
            self._total_received += len(samples)

    # ---- 消费指定数量的样本（FIFO）----
    def consume(self, batch_size: int) -> list:
        with self._lock:
            actual_size = min(batch_size, len(self._buffer))
            if actual_size == 0:
                return []
            batch = [self._buffer.popleft() for _ in range(actual_size)]
            return batch

    # ---- 当前缓存大小 ----
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    # ---- 累计接收样本数 ----
    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    # ---- 清空缓存 ----
    def clear(self):
        with self._lock:
            self._buffer.clear()
