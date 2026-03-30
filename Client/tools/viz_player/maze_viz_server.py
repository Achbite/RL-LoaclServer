#!/usr/bin/env python3
"""
迷宫训练框架 - 可视化回放 HTTP 服务

在 Client 容器内运行，提供 HTTP API 供浏览器加载回放数据。
纯 Python 标准库实现，无第三方依赖。

用法：
    python3 tools/viz_player/maze_viz_server.py [--dir log/viz] [--port 9004]

API 接口：
    GET /                    → 返回可视化播放器页面（maze_viz.html）
    GET /api/files           → 列出回放目录下所有 .jsonl 文件
    GET /api/frames?file=xxx → 加载指定文件的全部帧数据
    GET /api/status          → 服务状态信息
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


class VizReplayServer:
    """回放文件管理器，负责扫描目录、读取帧数据。"""

    def __init__(self, replay_dir):
        self.replay_dir = os.path.abspath(replay_dir)
        self.lock = threading.Lock()
        self._file_cache = {}           # 文件名 → 帧数据列表（缓存已加载的文件）
        self._file_list_cache = None    # 文件列表缓存
        self._file_list_time = 0        # 文件列表缓存时间

        # 确保目录存在
        os.makedirs(self.replay_dir, exist_ok=True)
        print(f"[VizServer] 监控目录: {self.replay_dir}")

    def list_files(self):
        """列出回放目录下所有 .jsonl 文件，按修改时间倒序。"""
        # 缓存 2 秒，避免频繁扫描磁盘
        now = time.time()
        if self._file_list_cache is not None and (now - self._file_list_time) < 2.0:
            return self._file_list_cache

        pattern = os.path.join(self.replay_dir, "*.jsonl")
        files = glob.glob(pattern)
        result = []

        for f in sorted(files, key=lambda x: os.path.getmtime(x), reverse=True):
            try:
                stat = os.stat(f)
                name = os.path.basename(f)
                # 尝试从缓存获取帧数
                frame_count = len(self._file_cache[name]) if name in self._file_cache else None
                result.append({
                    "name": name,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "is_growing": (now - stat.st_mtime) < 10,   # 10 秒内有更新视为正在写入
                    "frame_count": frame_count,
                })
            except OSError:
                pass

        self._file_list_cache = result
        self._file_list_time = now
        return result

    def load_frames(self, filename):
        """加载指定文件的全部帧数据。带缓存，重复请求不重新读取。"""
        # 安全检查：防止路径穿越
        if '..' in filename or '/' in filename or '\\' in filename:
            return None, "非法文件名"

        filepath = os.path.join(self.replay_dir, filename)
        if not os.path.exists(filepath):
            return None, f"文件不存在: {filename}"

        with self.lock:
            # 检查缓存（如果文件仍在写入则不使用缓存）
            try:
                current_mtime = os.path.getmtime(filepath)
                current_size = os.path.getsize(filepath)
            except OSError:
                return None, f"无法读取文件: {filename}"

            cache_key = filename
            if cache_key in self._file_cache:
                cached = self._file_cache[cache_key]
                # 如果文件大小和修改时间未变，使用缓存
                if cached.get("_mtime") == current_mtime and cached.get("_size") == current_size:
                    return cached["frames"], None

            # 读取文件
            frames = []
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            frames.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"[VizServer] 跳过无效行 {filename}:{line_num}: {e}")
            except Exception as e:
                return None, f"读取文件失败: {e}"

            # 写入缓存
            self._file_cache[cache_key] = {
                "frames": frames,
                "_mtime": current_mtime,
                "_size": current_size,
            }

            print(f"[VizServer] 加载完成: {filename} ({len(frames)} 帧, {current_size / 1024:.1f} KB)")
            return frames, None

    def get_status(self):
        """获取服务状态。"""
        file_list = self.list_files()
        return {
            "replay_dir": self.replay_dir,
            "file_count": len(file_list),
            "cached_files": len(self._file_cache),
        }


# 全局服务实例
viz_server = None


class VizHTTPHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器。"""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self._serve_html()

        elif path == '/api/files':
            file_list = viz_server.list_files()
            self._json_response(file_list)

        elif path == '/api/frames':
            filename = params.get('file', [None])[0]
            if not filename:
                self._json_response({"error": "缺少 file 参数"}, status=400)
                return
            frames, error = viz_server.load_frames(filename)
            if error:
                self._json_response({"error": error}, status=404)
            else:
                self._json_response({
                    "file": filename,
                    "total_frames": len(frames),
                    "frames": frames,
                })

        elif path == '/api/status':
            self._json_response(viz_server.get_status())

        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """处理 CORS 预检请求。"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _json_response(self, data, status=200):
        """发送 JSON 响应。"""
        try:
            body = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            body_bytes = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(body_bytes))
            self.end_headers()
            self.wfile.write(body_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_html(self):
        """返回可视化播放器 HTML 页面。"""
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze_viz.html')
        if not os.path.exists(html_path):
            self.send_error(404, "maze_viz.html 未找到")
            return
        try:
            with open(html_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        """过滤常规 200 请求日志，只打印异常请求。"""
        if '200' not in str(args):
            super().log_message(format, *args)


def main():
    global viz_server

    parser = argparse.ArgumentParser(description='迷宫训练框架 - 可视化回放 HTTP 服务')
    parser.add_argument('--dir', '-d', default='log/viz',
                        help='回放文件目录（默认: log/viz）')
    parser.add_argument('--port', '-p', type=int, default=9004,
                        help='HTTP 服务端口（默认: 9004）')
    args = parser.parse_args()

    # 初始化服务
    viz_server = VizReplayServer(args.dir)

    # 初始扫描
    file_list = viz_server.list_files()
    print(f"[VizServer] 发现 {len(file_list)} 个回放文件")

    # 启动 HTTP 服务（多线程处理并发请求）
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer(('0.0.0.0', args.port), VizHTTPHandler)
    print(f"[VizServer] 可视化回放服务已启动: http://0.0.0.0:{args.port}")
    print(f"[VizServer] 浏览器访问: http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[VizServer] 服务已停止")
        server.shutdown()


if __name__ == '__main__':
    main()
