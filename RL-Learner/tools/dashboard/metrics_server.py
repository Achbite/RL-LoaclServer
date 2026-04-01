#!/usr/bin/env python3
"""
迷宫训练框架 - 训练指标面板 HTTP 服务

在 Learner 容器内运行，提供 HTTP API 供浏览器加载训练指标数据。
纯 Python 标准库实现，无第三方依赖。

用法：
    python3 tools/dashboard/metrics_server.py [--dir logs/metrics] [--port 9005]

API 接口：
    GET /                         → 返回训练指标面板页面
    GET /api/metrics?since=N      → 增量查询 train_step > N 的记录
    GET /api/metrics/latest       → 最新一条指标
    GET /api/metrics/summary      → 统计摘要
    GET /api/status               → 服务状态
"""

import json
import os
import sys
import glob
import time
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs


class MetricsFileReader:
    """
    指标文件读取器

    监控指标目录下的 .jsonl 文件，支持增量读取（文件尾部追加感知）。
    自动发现最新的指标文件并持续跟踪。
    """

    def __init__(self, metrics_dir: str):
        self._metrics_dir = os.path.abspath(metrics_dir)
        self._lock = threading.Lock()
        self._records = []              # 全部已读取的记录
        self._current_file = None       # 当前跟踪的文件路径
        self._file_offset = 0           # 文件读取偏移量（字节）
        self._last_scan_time = 0        # 上次扫描目录时间

        os.makedirs(self._metrics_dir, exist_ok=True)
        print(f"[MetricsServer] 监控目录: {self._metrics_dir}")

    def refresh(self):
        """刷新数据：扫描新文件 + 增量读取"""
        with self._lock:
            self._scan_and_read()

    def query(self, since_step: int = 0) -> list:
        """查询 train_step > since_step 的记录"""
        self.refresh()
        with self._lock:
            return [r for r in self._records if r.get("train_step", 0) > since_step]

    def latest(self) -> dict:
        """返回最新一条记录"""
        self.refresh()
        with self._lock:
            return self._records[-1] if self._records else {}

    def summary(self) -> dict:
        """返回统计摘要"""
        self.refresh()
        with self._lock:
            if not self._records:
                return {
                    "total_steps": 0,
                    "best_pass_rate": 0.0,
                    "best_mean_reward": 0.0,
                    "latest_loss": 0.0,
                }
            return {
                "total_steps": len(self._records),
                "best_pass_rate": max(r.get("pass_rate", 0.0) for r in self._records),
                "best_mean_reward": max(r.get("mean_episode_reward", 0.0) for r in self._records),
                "latest_loss": self._records[-1].get("total_loss", 0.0),
            }

    def get_status(self) -> dict:
        """获取服务状态"""
        with self._lock:
            return {
                "metrics_dir": self._metrics_dir,
                "current_file": os.path.basename(self._current_file) if self._current_file else None,
                "record_count": len(self._records),
                "file_offset": self._file_offset,
            }

    def _scan_and_read(self):
        """扫描目录并增量读取（调用方已持有锁）"""
        now = time.time()

        # 每 1 秒扫描一次目录，发现新文件
        if now - self._last_scan_time >= 1.0:
            self._last_scan_time = now
            pattern = os.path.join(self._metrics_dir, "metrics_*.jsonl")
            files = sorted(glob.glob(pattern), key=lambda f: os.path.getmtime(f))
            if files:
                newest = files[-1]
                if newest != self._current_file:
                    # 切换到新文件，重置偏移
                    self._current_file = newest
                    self._file_offset = 0
                    self._records.clear()
                    print(f"[MetricsServer] 跟踪文件: {os.path.basename(newest)}")

        # 增量读取当前文件
        if self._current_file and os.path.exists(self._current_file):
            try:
                file_size = os.path.getsize(self._current_file)
                if file_size > self._file_offset:
                    with open(self._current_file, "r", encoding="utf-8") as f:
                        f.seek(self._file_offset)
                        new_data = f.read()
                        self._file_offset = f.tell()

                    for line in new_data.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self._records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except OSError as e:
                print(f"[MetricsServer] 读取文件失败: {e}")


# 全局读取器实例
metrics_reader = None


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()

        elif path == "/api/metrics":
            since = int(params.get("since", ["0"])[0])
            records = metrics_reader.query(since_step=since)
            self._json_response({
                "records": records,
                "total": len(records),
            })

        elif path == "/api/metrics/latest":
            self._json_response(metrics_reader.latest())

        elif path == "/api/metrics/summary":
            self._json_response(metrics_reader.summary())

        elif path == "/api/status":
            self._json_response(metrics_reader.get_status())

        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json_response(self, data, status=200):
        """发送 JSON 响应"""
        try:
            body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            body_bytes = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body_bytes))
            self.end_headers()
            self.wfile.write(body_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_html(self):
        """返回训练指标面板 HTML 页面"""
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_dashboard.html")
        if not os.path.exists(html_path):
            self.send_error(404, "training_dashboard.html 未找到")
            return
        try:
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        """过滤常规 200 请求日志，只打印异常请求"""
        if "200" not in str(args):
            super().log_message(format, *args)


def main():
    global metrics_reader

    parser = argparse.ArgumentParser(description="迷宫训练框架 - 训练指标面板 HTTP 服务")
    parser.add_argument("--dir", "-d", default="logs/metrics",
                        help="指标数据目录（默认: logs/metrics）")
    parser.add_argument("--port", "-p", type=int, default=9005,
                        help="HTTP 服务端口（默认: 9005）")
    args = parser.parse_args()

    # 初始化读取器
    metrics_reader = MetricsFileReader(args.dir)

    # 初始扫描
    metrics_reader.refresh()
    status = metrics_reader.get_status()
    print(f"[MetricsServer] 当前记录数: {status['record_count']}")

    # 启动 HTTP 服务（多线程处理并发请求）
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MetricsHTTPHandler)
    print(f"[MetricsServer] 训练指标面板服务已启动: http://0.0.0.0:{args.port}")
    print(f"[MetricsServer] 浏览器访问: http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MetricsServer] 服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
