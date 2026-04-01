"""
LearnerService gRPC 服务实现
接收 AIServer 推送的 SampleBatch，按 trajectory 组织存入样本缓存
"""

import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import maze_pb2
from proto import maze_pb2_grpc
from src.training.sample_buffer import SampleBuffer
from src.log.logger import setup_logger


class LearnerServiceImpl(maze_pb2_grpc.LearnerServiceServicer):
    """Learner gRPC 服务端，实现 LearnerService.SendSamples"""

    def __init__(self, sample_buffer: SampleBuffer, config: dict):
        self._buffer = sample_buffer
        self._config = config
        self._model_version = 0           # 当前模型版本号（由 train.py 主循环更新）
        self._logger = setup_logger("LearnerService")

    # ---- 接收样本批次（按 trajectory 组织）----
    def SendSamples(self, request, context):
        episode_id = request.episode_id
        agent_id = request.agent_id
        is_episode_end = request.is_episode_end
        num_samples = len(request.samples)

        self._logger.debug(
            "收到样本: episode=%d, agent=%d, samples=%d, ep_end=%s",
            episode_id, agent_id, num_samples, is_episode_end,
        )

        # 将 protobuf Sample 转换为 dict 列表
        batch = []
        for sample in request.samples:
            batch.append({
                "obs": list(sample.obs),
                "action": sample.action,
                "reward": sample.reward,
                "old_log_prob": sample.old_log_prob,
                "old_vpred": sample.old_vpred,
                "advantage": 0.0,                    # Learner 端 GAE 计算后填充
                "td_return": 0.0,                     # Learner 端 GAE 计算后填充
                "mask": sample.mask,
                "reward_details": dict(sample.reward_details),  # 奖励分项明细（用于 Dashboard 追踪）
            })

        # 按 trajectory 组织推入缓存
        self._buffer.push_trajectory(episode_id, agent_id, batch, is_episode_end)

        # 拥塞检测：缓冲区超过告警水位线时打印警告
        if self._buffer.check_congestion():
            stats = self._buffer.trajectory_stats()
            self._logger.warning(
                "⚠ 缓冲区拥塞: 待消费样本 %d，待消费片段 %d，"
                "Learner 消费速度可能跟不上 AIServer 样本产生速度",
                stats["active_samples"], stats["pending_fragments"],
            )

        # 构造响应
        response = maze_pb2.SampleResponse()
        response.ret_code = 0
        response.model_version = self._model_version

        return response

    # ---- 更新模型版本号（由 train.py 主循环调用）----
    def update_model_version(self, version: int):
        """更新返回给 AIServer 的模型版本号"""
        self._model_version = version
