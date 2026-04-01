"""
Learner 日志工具
简略控制台输出 + 详细文件日志，避免并行时淹没终端
"""

import logging
import os
from datetime import datetime

# --- 全局日志管理器缓存 ---
_loggers = {}


# ---- 创建/获取日志器 ----
def setup_logger(
    name: str,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: str = "logs",
) -> logging.Logger:
    """
    创建或获取指定名称的日志器
    首次调用时初始化 handler，后续调用返回已有实例
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"learner.{name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 避免重复添加 handler
    if logger.handlers:
        _loggers[name] = logger
        return logger

    # --- 控制台 handler（简略输出）---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # --- 文件 handler（详细输出）---
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("learner_%Y%m%d_%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
