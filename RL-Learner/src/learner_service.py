"""
LearnerService gRPC 服务实现
接收 AIServer 推送的 SampleBatch，存入样本缓存
"""

import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import maze_pb2
from proto import maze_pb2_grpc
from src.sample_buffer import SampleBuffer
from src.logger import setup_logger


class LearnerServiceImpl(maze_pb2_grpc.LearnerServiceServicer):
    """Learner gRPC 服务端，实现 LearnerService.SendSamples"""

    def __init__(self, sample_buffer: SampleBuffer, config: dict):
        self._buffer = sample_buffer
        self._config = config
        self._model_version = 0           # 当前模型版本号（Phase 3B 更新）
        self._logger = setup_logger("LearnerService")

    # ---- 接收样本批次 ----
    def SendSamples(self, request, context):
        episode_id = request.episode_id
        agent_id = request.agent_id
        num_samples = len(request.samples)

        self._logger.debug(
            "收到样本: episode=%d, agent=%d, samples=%d",
            episode_id, agent_id, num_samples,
        )

        # 将 protobuf Sample 转换为 dict 列表存入缓存
        for sample in request.samples:
            sample_dict = {
                "episode_id": episode_id,
                "agent_id": agent_id,
                "obs": list(sample.obs),
                "action": sample.action,
                "reward": sample.reward,
                "old_log_prob": sample.old_log_prob,
                "old_vpred": sample.old_vpred,
                "advantage": sample.advantage,
                "td_return": sample.td_return,
                "mask": sample.mask,
            }
            self._buffer.push(sample_dict)

        # 构造响应
        response = maze_pb2.SampleResponse()
        response.ret_code = 0
        response.model_version = self._model_version

        return response
