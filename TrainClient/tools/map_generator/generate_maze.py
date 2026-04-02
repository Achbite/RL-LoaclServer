#!/usr/bin/env python3
"""
随机迷宫地图生成器（混合模型 Room & Corridor）

使用 DFS（深度优先搜索）算法在 N×N 网格上生成随机迷宫。
墙壁是格子之间的边界线段，不占据格子空间——所有格子都是可通行的"房间"，
DFS 打通 = 移除两个相邻格子之间的边界线段。

设计理念（Room & Corridor 混合模型）：
  - 每个 DFS 房间展开为 K×K 的可通行区域（K=room_size，可配置）
  - 墙壁宽度固定为 1 格，房间之间通过 K 格宽走廊连通
  - 房间内可随机放置障碍物（柱子/口袋），密度可配置
  - 八向移动在宽敞房间内有实际意义（对角线可有效缩短路径）
  - 合并连续的水平/垂直 blocked 格子为长线段，减少墙壁数量
  - 输出格式与 Client 端 AABB 碰撞检测完全兼容

自动行为：
  - 不指定 --seed 时自动随机生成种子
  - 自动保存到 maps/save/ 目录，命名为 maze_[种子].json
  - 自动检查可达性，不可达则换种子重试（最多 100 次）
  - 支持从 map_config.yaml 读取生成参数

用法：
  python generate_maze.py                          # 自动生成一张
  python generate_maze.py --seed 42                # 指定种子
  python generate_maze.py --count 5                # 批量生成 5 张
  python generate_maze.py --seed 42 --preview      # 生成并终端预览
  python generate_maze.py --count 3 --output-dir ../../maps/test/  # 输出到 test 目录
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path):
    """
    从 YAML 配置文件加载生成参数

    使用简易 YAML 解析（不依赖 PyYAML），支持单层嵌套的 key: value 格式。

    参数：
        config_path: 配置文件路径

    返回：
        config: 嵌套字典 {section: {key: value}}
    """
    config = {}
    if not os.path.exists(config_path):
        return config

    current_section = None
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行内注释
            comment_pos = -1
            in_quotes = False
            for i, ch in enumerate(line):
                if ch in ('"', "'"):
                    in_quotes = not in_quotes
                elif ch == '#' and not in_quotes:
                    comment_pos = i
                    break
            if comment_pos >= 0:
                line = line[:comment_pos]

            stripped = line.strip()
            if not stripped:
                continue

            # 查找冒号
            colon_pos = stripped.find(':')
            if colon_pos < 0:
                continue

            key = stripped[:colon_pos].strip()
            val = stripped[colon_pos + 1:].strip().strip('"').strip("'")

            # 判断是 section 还是 key-value
            has_indent = line[0] in (' ', '\t')

            if not has_indent and not val:
                current_section = key
                config[current_section] = {}
            elif has_indent and current_section and key:
                config[current_section][key] = val

    return config


def get_config_value(config, section, key, default, type_fn=str):
    """从配置字典中安全获取值"""
    try:
        val = config.get(section, {}).get(key, "")
        if val == "" or val is None:
            return default
        return type_fn(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# 难度映射
# ============================================================

# 难度 → 额外打通比例（难度越高，打通越少，路径越少）
DIFFICULTY_MAP = {
    1: 0.20,    # 简单：大量额外通路
    2: 0.12,    # 较易
    3: 0.08,    # 标准
    4: 0.04,    # 较难
    5: 0.01,    # 困难：几乎纯 DFS 迷宫
}

# 四方向偏移：右、下、左、上
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# ============================================================
# 边墙数据结构
# ============================================================

class WallEdgeSet:
    """
    边墙集合：记录每对相邻格子之间是否有墙壁

    使用 frozenset 存储边，确保 (a,b) 和 (b,a) 等价。
    水平边墙：格子 (x,y) 和 (x+1,y) 之间
    垂直边墙：格子 (x,y) 和 (x,y+1) 之间
    """

    def __init__(self, grid_dim):
        """初始化：所有相邻格子之间都有墙壁"""
        self.grid_dim = grid_dim
        self.walls = set()

        # 添加所有内部边墙
        for y in range(grid_dim):
            for x in range(grid_dim):
                # 右边墙（水平）
                if x + 1 < grid_dim:
                    self.walls.add(frozenset(((x, y), (x + 1, y))))
                # 上边墙（垂直）
                if y + 1 < grid_dim:
                    self.walls.add(frozenset(((x, y), (x, y + 1))))

    def has_wall(self, x1, y1, x2, y2):
        """检查两个相邻格子之间是否有墙壁"""
        return frozenset(((x1, y1), (x2, y2))) in self.walls

    def remove_wall(self, x1, y1, x2, y2):
        """移除两个相邻格子之间的墙壁（DFS 打通）"""
        self.walls.discard(frozenset(((x1, y1), (x2, y2))))

    def wall_count(self):
        """返回当前墙壁数量"""
        return len(self.walls)

    def get_all_walls(self):
        """返回所有墙壁边的列表，每条边为 ((x1,y1), (x2,y2))，保证 x1<=x2, y1<=y2"""
        result = []
        for edge in self.walls:
            a, b = sorted(edge)
            result.append((a, b))
        return result


# ============================================================
# DFS 迷宫生成（边墙模型）
# ============================================================

def generate_maze_edges(grid_dim, rng):
    """
    DFS 随机迷宫生成（边墙模型）

    在 grid_dim × grid_dim 的网格上，所有格子都是可通行的"房间"。
    初始状态所有相邻格子之间都有墙壁，DFS 遍历时移除墙壁形成通路。

    参数：
        grid_dim: 网格维度（N×N）
        rng: random.Random 实例

    返回：
        edge_set: WallEdgeSet 实例，包含所有剩余墙壁
    """
    edge_set = WallEdgeSet(grid_dim)

    # DFS 遍历所有格子
    visited = [[False] * grid_dim for _ in range(grid_dim)]
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        cx, cy = stack[-1]

        # 收集未访问的邻居格子
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_dim and 0 <= ny < grid_dim and not visited[ny][nx]:
                neighbors.append((nx, ny))

        if not neighbors:
            stack.pop()
            continue

        # 随机选择一个邻居，移除之间的墙壁
        nx, ny = rng.choice(neighbors)
        visited[ny][nx] = True
        edge_set.remove_wall(cx, cy, nx, ny)
        stack.append((nx, ny))

    return edge_set


def extra_open_walls(edge_set, ratio, rng):
    """
    额外随机打通墙壁，增加多路径

    从剩余墙壁中随机选择一定比例移除，使迷宫不再是完美迷宫（有多条路径）。

    参数：
        edge_set: WallEdgeSet 实例
        ratio: 打通比例（0~1）
        rng: random.Random 实例
    """
    remaining = list(edge_set.walls)
    count = int(len(remaining) * ratio)
    if count > 0:
        to_remove = rng.sample(remaining, min(count, len(remaining)))
        for edge in to_remove:
            edge_set.walls.discard(edge)


# ============================================================
# 起终点安全区清除
# ============================================================

def clear_safe_zone(edge_set, grid_dim, cx, cy, radius=2):
    """
    清除指定格子周围 radius 范围内的所有墙壁

    确保起终点周围有足够的空旷通道，Agent 不会被墙壁紧贴包围。

    参数：
        edge_set: WallEdgeSet 实例
        grid_dim: 网格维度
        cx, cy: 中心格子坐标
        radius: 清除半径（默认 2，即 5×5 范围内无墙壁）
    """
    for y in range(max(0, cy - radius), min(grid_dim, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(grid_dim, cx + radius + 1)):
            # 移除该格子与右侧格子之间的墙壁
            if x + 1 < grid_dim and x + 1 <= cx + radius:
                edge_set.remove_wall(x, y, x + 1, y)
            # 移除该格子与上方格子之间的墙壁
            if y + 1 < grid_dim and y + 1 <= cy + radius:
                edge_set.remove_wall(x, y, x, y + 1)


def clear_blocked_safe_zone(blocked, out_dim, gx, gy, radius=1):
    """
    清除 blocked 网格中指定坐标周围 (2*radius+1)×(2*radius+1) 范围内的所有 blocked 格子

    用于确保起终点在输出网格中周围有足够空间。

    参数：
        blocked: 网格 blocked 数组
        out_dim: 输出网格维度
        gx, gy: 中心网格坐标
        radius: 清除半径（默认 1，即 3×3 范围）
    """
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < out_dim and 0 <= ny < out_dim:
                blocked[ny][nx] = False


def add_room_obstacles(blocked, grid_dim, room_size, density, rng,
                       start_rx, start_ry, end_rx, end_ry, safe_radius=1):
    """
    在房间内随机放置障碍物，形成口袋/柱子结构

    遍历每个 DFS 房间的 room_size × room_size 区域，按概率将部分格子标记为 blocked。
    起终点所在房间及其安全半径内的房间不放置障碍物。

    参数：
        blocked: 输出网格 blocked 数组
        grid_dim: DFS 房间维度 N
        room_size: 每个房间的格子数
        density: 障碍物密度（0~1），每个房间内格子被标记为障碍的概率
        rng: random.Random 实例
        start_rx, start_ry: 起点所在的房间坐标
        end_rx, end_ry: 终点所在的房间坐标
        safe_radius: 起终点安全半径（房间坐标），此范围内的房间不放障碍
    """
    if density <= 0:
        return

    stride = room_size + 1

    for ry in range(grid_dim):
        for rx in range(grid_dim):
            # 跳过起终点安全区内的房间
            if (abs(rx - start_rx) <= safe_radius and abs(ry - start_ry) <= safe_radius):
                continue
            if (abs(rx - end_rx) <= safe_radius and abs(ry - end_ry) <= safe_radius):
                continue

            # 房间内部区域（不包括边缘，避免堵死走廊入口）
            base_x = rx * stride + 1
            base_y = ry * stride + 1

            # 对于 room_size <= 2 的小房间，直接在全部格子上放障碍
            # 对于 room_size >= 3 的大房间，避开边缘行/列（保留走廊入口通畅）
            if room_size >= 3:
                inner_range_x = range(1, room_size - 1)
                inner_range_y = range(1, room_size - 1)
            else:
                inner_range_x = range(room_size)
                inner_range_y = range(room_size)

            for dy in inner_range_y:
                for dx in inner_range_x:
                    if rng.random() < density:
                        blocked[base_y + dy][base_x + dx] = True


# ============================================================
# 边墙 → 墙壁线段转换（合并连续线段）
# ============================================================

def edges_to_blocked_grid(edge_set, grid_dim, room_size=1):
    """
    将边墙集合转换为可配置房间大小的 blocked 网格

    映射规则（Room & Corridor 模型）：
      - 每个 DFS 房间 (rx, ry) 展开为 room_size × room_size 的可通行区域
      - 墙壁宽度固定为 1 格
      - 房间之间的走廊宽度 = room_size 格（与房间等宽，自然连接）
      - 输出网格维度 = room_dim × (room_size + 1) + 1

    当 room_size=1 时退化为经典 2N+1 编码（向后兼容）。

    参数：
        edge_set: WallEdgeSet 实例（N×N 房间的边墙）
        grid_dim: 房间维度 N
        room_size: 每个房间展开的格子数（默认 1 = 经典模式）

    返回：
        blocked: 输出网格的二维列表，True=不可通行
        out_dim: 输出网格维度
    """
    stride = room_size + 1  # 每个房间占 stride 格（room_size 通道 + 1 墙壁）
    out_dim = grid_dim * stride + 1
    blocked = [[False] * out_dim for _ in range(out_dim)]

    # ---- 1. 先将所有格子标记为墙壁，再挖空房间和走廊 ----
    for gy in range(out_dim):
        for gx in range(out_dim):
            blocked[gy][gx] = True

    # ---- 2. 挖空每个房间（room_size × room_size 区域）----
    for ry in range(grid_dim):
        for rx in range(grid_dim):
            # 房间左下角在输出网格中的坐标
            base_x = rx * stride + 1
            base_y = ry * stride + 1
            for dy in range(room_size):
                for dx in range(room_size):
                    blocked[base_y + dy][base_x + dx] = False

    # ---- 3. 挖空走廊（移除边墙对应的格子）----
    # 对于没有墙壁的相邻房间对，打通它们之间的 1 格宽墙壁带
    for ry in range(grid_dim):
        for rx in range(grid_dim):
            # 检查右侧邻居 (rx+1, ry)
            if rx + 1 < grid_dim and not edge_set.has_wall(rx, ry, rx + 1, ry):
                # 打通水平走廊：墙壁列 = (rx+1)*stride，行范围 = 房间的 room_size 行
                wall_col = (rx + 1) * stride
                base_y = ry * stride + 1
                for dy in range(room_size):
                    blocked[base_y + dy][wall_col] = False

            # 检查上方邻居 (rx, ry+1)
            if ry + 1 < grid_dim and not edge_set.has_wall(rx, ry, rx, ry + 1):
                # 打通垂直走廊：墙壁行 = (ry+1)*stride，列范围 = 房间的 room_size 列
                wall_row = (ry + 1) * stride
                base_x = rx * stride + 1
                for dx in range(room_size):
                    blocked[wall_row][base_x + dx] = False

    return blocked, out_dim


def blocked_to_wall_segments(blocked, out_dim, grid_size, wall_thickness):
    """
    将 blocked 网格转换为合并后的墙壁线段列表（JSON 输出格式）

    策略：双向扫描合并
      1. 按列扫描，合并连续垂直 blocked 格子为垂直线段
      2. 按行扫描，合并剩余未被覆盖的格子为水平线段
      3. 使用已标记集合避免重复

    参数：
        blocked: 网格 blocked 数组
        out_dim: 网格维度
        grid_size: 输出网格大小（cm）
        wall_thickness: 墙壁厚度（cm）

    返回：
        walls: 墙壁线段列表 [{"x1", "y1", "x2", "y2", "thickness"}]
    """
    walls = []
    used = set()

    # ---- 1. 按列扫描，合并垂直连续 blocked 格子 ----
    for gx in range(out_dim):
        gy = 0
        while gy < out_dim:
            if blocked[gy][gx] and (gx, gy) not in used:
                start_gy = gy
                while gy < out_dim and blocked[gy][gx]:
                    gy += 1
                end_gy = gy  # end_gy 是第一个非 blocked 的格子索引

                # 垂直线段：x 在格子中心，y 从第一个 blocked 格子中心到最后一个 blocked 格子中心
                # 这样 AABB 扩展 halfThickness 后不会侵入相邻的通道格子
                x_center = round((gx + 0.5) * grid_size, 2)
                y1 = round((start_gy + 0.5) * grid_size, 2)
                y2 = round((end_gy - 1 + 0.5) * grid_size, 2)
                walls.append({
                    "x1": x_center, "y1": y1,
                    "x2": x_center, "y2": y2,
                    "thickness": wall_thickness
                })
                for mark_gy in range(start_gy, end_gy):
                    used.add((gx, mark_gy))
            else:
                gy += 1

    # ---- 2. 按行扫描，合并水平连续 blocked 格子（跳过已标记的）----
    for gy in range(out_dim):
        gx = 0
        while gx < out_dim:
            if blocked[gy][gx] and (gx, gy) not in used:
                start_gx = gx
                while gx < out_dim and blocked[gy][gx] and (gx, gy) not in used:
                    gx += 1
                end_gx = gx

                # 水平线段：y 在格子中心，x 从第一个 blocked 格子中心到最后一个 blocked 格子中心
                x1 = round((start_gx + 0.5) * grid_size, 2)
                x2 = round((end_gx - 1 + 0.5) * grid_size, 2)
                y_center = round((gy + 0.5) * grid_size, 2)
                walls.append({
                    "x1": x1, "y1": y_center,
                    "x2": x2, "y2": y_center,
                    "thickness": wall_thickness
                })
                for mark_gx in range(start_gx, end_gx):
                    used.add((mark_gx, gy))
            else:
                gx += 1

    return walls


# ============================================================
# BFS 可达性验证（边墙模型，4 方向）
# ============================================================

def bfs_reachable(edge_set, grid_dim, sx, sy, ex, ey):
    """
    BFS 验证从 (sx, sy) 到 (ex, ey) 是否可达（4 方向移动，检查边墙）

    在边墙模型中，所有格子都可通行，但相邻格子之间可能有墙壁阻隔。

    参数：
        edge_set: WallEdgeSet 实例
        grid_dim: 网格维度
        sx, sy: 起点网格坐标
        ex, ey: 终点网格坐标

    返回：
        (reachable, path_length): 是否可达，最短路径长度（-1 表示不可达）
    """
    visited = [[False] * grid_dim for _ in range(grid_dim)]
    visited[sy][sx] = True
    queue = deque([(sx, sy, 0)])

    while queue:
        cx, cy, dist = queue.popleft()
        if cx == ex and cy == ey:
            return True, dist

        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_dim and 0 <= ny < grid_dim and not visited[ny][nx]:
                # 检查两个格子之间是否有墙壁
                if not edge_set.has_wall(cx, cy, nx, ny):
                    visited[ny][nx] = True
                    queue.append((nx, ny, dist + 1))

    return False, -1


# ============================================================
# 随机起终点生成
# ============================================================

def generate_random_start_end(grid_dim, min_distance, rng):
    """
    随机生成起终点，确保几何距离 >= min_distance

    在边墙模型中所有格子都可通行，只需保证距离足够。

    参数：
        grid_dim: 网格维度
        min_distance: 最小几何距离（网格数）
        rng: random.Random 实例

    返回：
        (start_gx, start_gy, end_gx, end_gy) 或 None（失败）
    """
    for _ in range(500):
        sx = rng.randint(0, grid_dim - 1)
        sy = rng.randint(0, grid_dim - 1)
        ex = rng.randint(0, grid_dim - 1)
        ey = rng.randint(0, grid_dim - 1)
        dist = math.sqrt((sx - ex) ** 2 + (sy - ey) ** 2)
        if dist >= min_distance:
            return sx, sy, ex, ey

    return None


# ============================================================
# ASCII 预览
# ============================================================

def ascii_preview(edge_set, grid_dim, start_gx, start_gy, end_gx, end_gy):
    """
    终端 ASCII 预览地图（边墙模型）

    使用 2N+1 字符网格渲染：
      ██ = 墙壁交叉点或边墙
      S  = 起点
      E  = 终点
      ·  = 可通行格子
      -- = 水平边墙
      |  = 垂直边墙
    """
    # 使用 (2*grid_dim+1) × (2*grid_dim+1) 的字符网格
    char_dim = 2 * grid_dim + 1
    grid = [[' '] * char_dim for _ in range(char_dim)]

    # 所有交叉点（偶数,偶数）标记为墙壁柱
    for cy in range(grid_dim + 1):
        for cx in range(grid_dim + 1):
            grid[cy * 2][cx * 2] = '#'

    # 水平边墙：格子 (x,y) 和 (x+1,y) 之间 → 字符位置 (2*x+2, 2*y+1)... 不对
    # 垂直边墙：格子 (x,y) 和 (x,y+1) 之间 → 字符位置 (2*x+1, 2*y+2)

    # 上边界
    for x in range(grid_dim):
        grid[0][x * 2 + 1] = '-'
    # 下边界
    for x in range(grid_dim):
        grid[grid_dim * 2][x * 2 + 1] = '-'
    # 左边界
    for y in range(grid_dim):
        grid[y * 2 + 1][0] = '|'
    # 右边界
    for y in range(grid_dim):
        grid[y * 2 + 1][grid_dim * 2] = '|'

    # 内部边墙
    for (a, b) in edge_set.get_all_walls():
        ax, ay = a
        bx, by = b
        if ay == by:
            # 水平边墙（左右相邻）→ 垂直线 | 在字符网格中
            char_x = bx * 2  # bx = ax + 1
            char_y = ay * 2 + 1
            grid[char_y][char_x] = '|'
        else:
            # 垂直边墙（上下相邻）→ 水平线 - 在字符网格中
            char_x = ax * 2 + 1
            char_y = by * 2  # by = ay + 1
            grid[char_y][char_x] = '-'

    # 格子内容（奇数,奇数）
    for y in range(grid_dim):
        for x in range(grid_dim):
            char_x = x * 2 + 1
            char_y = y * 2 + 1
            if x == start_gx and y == start_gy:
                grid[char_y][char_x] = 'S'
            elif x == end_gx and y == end_gy:
                grid[char_y][char_x] = 'E'
            else:
                grid[char_y][char_x] = '·'

    # 从上到下（y 从大到小）输出
    lines = []
    for cy in range(char_dim - 1, -1, -1):
        lines.append(''.join(grid[cy]))

    return '\n'.join(lines)


def ascii_preview_blocked(blocked, grid_dim, start_gx, start_gy, end_gx, end_gy):
    """
    终端 ASCII 预览地图（基于 blocked 网格）

    ██ = 墙壁（blocked）
    S  = 起点
    E  = 终点
    ·  = 可通行
    """
    lines = []
    for gy in range(grid_dim - 1, -1, -1):
        row = ""
        for gx in range(grid_dim):
            if gx == start_gx and gy == start_gy:
                row += " S"
            elif gx == end_gx and gy == end_gy:
                row += " E"
            elif blocked[gy][gx]:
                row += "██"
            else:
                row += " ·"
        lines.append(row)

    return "\n".join(lines)


# ============================================================
# 地图生成主函数
# ============================================================

def generate_map(seed, grid_dim, grid_size, wall_thickness, extra_open_ratio,
                 start_end_mode, fixed_start, fixed_end, min_distance,
                 room_size=1, room_obstacle_density=0.0):
    """
    生成一张随机迷宫地图（混合模型 Room & Corridor）

    内部使用边墙模型在 room_dim×room_dim 的房间网格上 DFS 生成迷宫，
    输出时将每个房间展开为 room_size×room_size 的可通行区域，
    墙壁宽度固定 1 格，确保与 Client 端的 AABB 碰撞检测完全兼容。

    参数：
        seed: 随机种子
        grid_dim: 配置的网格维度
        grid_size: 配置的网格大小 (cm)（自动重算以保持地图总尺寸不变）
        wall_thickness: 墙壁厚度 (cm)
        extra_open_ratio: 额外打通墙壁比例
        start_end_mode: 起终点模式 ("default"/"fixed"/"random")
        fixed_start: (x, y) 固定起点坐标（连续坐标）
        fixed_end: (x, y) 固定终点坐标（连续坐标）
        min_distance: random 模式最小几何距离（网格数）
        room_size: 每个房间展开的格子数（1=经典 2N+1，3~5=宽敞房间）
        room_obstacle_density: 房间内障碍物密度（0~1）

    返回：
        map_data: 地图 JSON 字典
        edge_set: WallEdgeSet 实例
        reachable: 是否可达
        path_length: BFS 最短路径长度
    """
    rng = random.Random(seed)

    # 计算房间维度：room_dim = grid_dim // (room_size + 1)
    # 输出网格维度：out_dim = room_dim * (room_size + 1) + 1
    stride = room_size + 1
    room_dim = grid_dim // stride
    if room_dim < 2:
        room_dim = 2  # 最少 2×2 房间
    out_dim = room_dim * stride + 1

    # 保持地图总尺寸不变，重算输出网格大小
    map_size = grid_dim * grid_size
    out_grid_size = map_size / out_dim

    # 1. DFS 生成迷宫（边墙模型，room_dim×room_dim 房间）
    edge_set = generate_maze_edges(room_dim, rng)

    # 2. 额外打通墙壁（增加多路径）
    extra_open_walls(edge_set, extra_open_ratio, rng)

    # 3. 确定起终点（在房间坐标系中）
    if start_end_mode == "fixed":
        # 连续坐标 → 输出网格坐标 → 最近的房间中心
        start_out_gx = int(fixed_start[0] / out_grid_size)
        start_out_gy = int(fixed_start[1] / out_grid_size)
        end_out_gx = int(fixed_end[0] / out_grid_size)
        end_out_gy = int(fixed_end[1] / out_grid_size)
        # 转换为房间坐标（反推：gx = rx * stride + 1 + room_size//2 → rx = (gx - 1) // stride）
        start_rx = max(0, min(room_dim - 1, (start_out_gx - 1) // stride))
        start_ry = max(0, min(room_dim - 1, (start_out_gy - 1) // stride))
        end_rx = max(0, min(room_dim - 1, (end_out_gx - 1) // stride))
        end_ry = max(0, min(room_dim - 1, (end_out_gy - 1) // stride))
    elif start_end_mode == "random":
        result = generate_random_start_end(room_dim, min_distance, rng)
        if result is None:
            return None, edge_set, False, -1
        start_rx, start_ry, end_rx, end_ry = result
    else:
        # default：对角线（第一个/最后一个房间）
        start_rx, start_ry = 0, 0
        end_rx, end_ry = room_dim - 1, room_dim - 1

    # 4. 确保起终点周围安全区空旷
    clear_safe_zone(edge_set, room_dim, start_rx, start_ry, radius=1)
    clear_safe_zone(edge_set, room_dim, end_rx, end_ry, radius=1)

    # 5. BFS 验证可达性（在房间坐标系中）
    reachable, path_length = bfs_reachable(
        edge_set, room_dim, start_rx, start_ry, end_rx, end_ry
    )

    # 6. 边墙 → blocked 网格（可配置房间大小）
    blocked, _ = edges_to_blocked_grid(edge_set, room_dim, room_size)

    # 7. 起终点在输出网格中的坐标（房间中心）
    # 房间 (rx, ry) 的中心 = (rx * stride + 1 + room_size//2, ry * stride + 1 + room_size//2)
    start_gx = start_rx * stride + 1 + room_size // 2
    start_gy = start_ry * stride + 1 + room_size // 2
    end_gx = end_rx * stride + 1 + room_size // 2
    end_gy = end_ry * stride + 1 + room_size // 2

    # 7a. 房间内随机障碍物（口袋/柱子）
    if room_obstacle_density > 0 and room_size >= 2:
        add_room_obstacles(blocked, room_dim, room_size, room_obstacle_density, rng,
                           start_rx, start_ry, end_rx, end_ry, safe_radius=1)

    # 确保起终点在 blocked 网格中可通行
    clear_blocked_safe_zone(blocked, out_dim, start_gx, start_gy, radius=1)
    clear_blocked_safe_zone(blocked, out_dim, end_gx, end_gy, radius=1)

    # 8. blocked 网格 → 合并墙壁线段
    wall_segments = blocked_to_wall_segments(blocked, out_dim, out_grid_size, wall_thickness)

    # 9. 起终点连续坐标（输出网格中心，保留浮点精度）
    start_pos = {"x": round((start_gx + 0.5) * out_grid_size, 2), "y": round((start_gy + 0.5) * out_grid_size, 2)}
    end_pos = {"x": round((end_gx + 0.5) * out_grid_size, 2), "y": round((end_gy + 0.5) * out_grid_size, 2)}

    # 10. 组装地图数据（v3 格式，新增 room_size 字段）
    map_data = {
        "map_id": f"maze_{seed}",
        "version": 3,
        "seed": seed,
        "difficulty": 0,
        "grid_count": out_dim,
        "grid_size": round(out_grid_size, 2),
        "room_size": room_size,
        "room_dim": room_dim,
        "bounds": {"x_min": 0, "x_max": map_size, "y_min": 0, "y_max": map_size},
        "start_pos": start_pos,
        "end_pos": end_pos,
        "bfs_path_length": path_length if reachable else -1,
        "wall_count": len(wall_segments),
        "walls": wall_segments
    }

    return map_data, edge_set, reachable, path_length, blocked, out_dim, out_grid_size


# ============================================================
# 自动生成可达地图（带重试）
# ============================================================

MAX_RETRY = 100

def generate_reachable_map(seed=None, **kwargs):
    """
    生成一张保证可达的迷宫地图

    自动检查 BFS 可达性，不可达则递增种子重试，最多 MAX_RETRY 次。

    参数：
        seed: 随机种子（None 则基于时间戳自动生成）
        **kwargs: 传递给 generate_map 的其他参数

    返回：
        map_data: 地图 JSON 字典
        edge_set: WallEdgeSet 实例
        path_length: BFS 最短路径长度
        final_seed: 最终使用的种子
        retry_count: 重试次数
    """
    if seed is None:
        seed = int(time.time() * 1000) % (10 ** 9)

    for attempt in range(MAX_RETRY):
        current_seed = seed + attempt
        result = generate_map(current_seed, **kwargs)
        map_data, edge_set, reachable, path_length = result[0], result[1], result[2], result[3]
        blocked_grid = result[4] if len(result) > 4 else None
        out_dim = result[5] if len(result) > 5 else 0
        out_grid_size = result[6] if len(result) > 6 else 0

        if map_data is None:
            # random 模式下找不到合适的起终点
            if attempt < MAX_RETRY - 1:
                continue

        if reachable:
            return map_data, edge_set, path_length, current_seed, attempt, blocked_grid, out_dim

        if attempt < MAX_RETRY - 1:
            continue

    print(f"  ⚠ 严重警告：从 seed={seed} 开始连续 {MAX_RETRY} 个种子均不可达")
    return None, None, -1, seed, MAX_RETRY, None, 0


# ============================================================
# 命令行入口
# ============================================================

def main():
    # 脚本所在目录，用于计算默认路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "maps", "save"))
    default_config_path = os.path.join(script_dir, "map_config.yaml")

    parser = argparse.ArgumentParser(
        description="随机迷宫地图生成器（混合模型 Room & Corridor）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例：
  # 自动生成一张地图（随机种子，保存到 maps/save/）
  python generate_maze.py

  # 指定种子生成
  python generate_maze.py --seed 42

  # 批量生成 5 张
  python generate_maze.py --count 5

  # 生成并终端预览
  python generate_maze.py --seed 42 --preview

  # 输出到 test 目录（供 client 使用）
  python generate_maze.py --count 3 --output-dir ../../maps/test/

  # 指定配置文件
  python generate_maze.py --config my_config.yaml

默认保存目录: {default_save_dir}
默认配置文件: {default_config_path}
        """
    )

    parser.add_argument("--seed", type=int, default=None,
                        help="指定随机种子（不指定则自动随机生成）")
    parser.add_argument("--count", type=int, default=1,
                        help="生成地图数量（默认 1）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"输出目录（默认 maps/save/）")
    parser.add_argument("--preview", action="store_true",
                        help="终端 ASCII 预览")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help=f"配置文件路径（默认 {default_config_path}）")
    parser.add_argument("--extra-open", type=float, default=None,
                        help="额外打通墙壁比例（覆盖配置文件）")
    parser.add_argument("--difficulty", type=int, default=None,
                        help="难度等级 1-5（覆盖配置文件）")

    args = parser.parse_args()

    # ---- 加载配置 ----
    config = load_config(args.config)
    if config:
        print(f"已加载配置: {args.config}")

    # 地图参数
    grid_count = get_config_value(config, "map", "grid_count", 40, int)
    grid_size = get_config_value(config, "map", "grid_size", 500, int)
    room_size = get_config_value(config, "map", "room_size", 3, int)
    room_size = max(1, room_size)  # 最小为 1（经典 2N+1 模式）

    # 起终点参数
    start_end_mode = get_config_value(config, "start_end", "mode", "default")
    min_distance = get_config_value(config, "start_end", "min_distance", 5, int)
    fixed_start = (
        get_config_value(config, "start_end", "fixed_start_x", 750.0, float),
        get_config_value(config, "start_end", "fixed_start_y", 750.0, float),
    )
    fixed_end = (
        get_config_value(config, "start_end", "fixed_end_x", 19750.0, float),
        get_config_value(config, "start_end", "fixed_end_y", 19750.0, float),
    )

    # 难度和墙壁参数
    difficulty = args.difficulty or get_config_value(config, "maze", "difficulty", 3, int)
    difficulty = max(1, min(5, difficulty))
    wall_thickness = get_config_value(config, "maze", "wall_thickness", 10, int)

    # 房间内障碍物密度
    room_obstacle_density = get_config_value(config, "maze", "room_obstacle_density", 0.0, float)
    room_obstacle_density = max(0.0, min(1.0, room_obstacle_density))

    # 额外打通比例：命令行 > 配置文件 > 难度自动
    if args.extra_open is not None:
        extra_open_ratio = args.extra_open
    else:
        config_ratio = get_config_value(config, "maze", "extra_open_ratio", -1, float)
        if config_ratio >= 0:
            extra_open_ratio = config_ratio
        else:
            extra_open_ratio = DIFFICULTY_MAP.get(difficulty, 0.08)

    # 确定输出目录
    save_dir = args.output_dir if args.output_dir else default_save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 打印生成参数
    map_size = grid_count * grid_size
    stride = room_size + 1
    room_dim = grid_count // stride
    if room_dim < 2:
        room_dim = 2
    out_dim = room_dim * stride + 1
    out_grid_size = map_size / out_dim
    print(f"生成参数: {room_dim}×{room_dim} 房间 (room_size={room_size}) → {out_dim}×{out_dim} 输出网格, "
          f"格子≈{out_grid_size:.0f}cm, 地图={map_size}×{map_size}cm")
    print(f"难度: {difficulty}, 额外打通: {extra_open_ratio:.2f}, "
          f"房间障碍密度: {room_obstacle_density:.2f}, 起终点模式: {start_end_mode}")
    print(f"模型: Room & Corridor 混合模型（room_size={room_size}）")

    # 确定种子列表
    if args.seed is not None:
        base_seeds = [args.seed + i for i in range(args.count)]
    else:
        base_seeds = [None] * args.count

    # 生成地图
    success_count = 0
    total_retries = 0

    for i, base_seed in enumerate(base_seeds):
        map_data, edge_set, path_length, final_seed, retries, blocked_grid, out_dim_val = generate_reachable_map(
            seed=base_seed,
            grid_dim=grid_count,
            grid_size=grid_size,
            wall_thickness=wall_thickness,
            extra_open_ratio=extra_open_ratio,
            start_end_mode=start_end_mode,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
            min_distance=min_distance,
            room_size=room_size,
            room_obstacle_density=room_obstacle_density,
        )
        total_retries += retries

        if map_data is None:
            print(f"[{i+1}/{args.count}] ✗ 生成失败（连续 {MAX_RETRY} 次不可达）")
            continue

        # 填充难度
        map_data["difficulty"] = difficulty

        success_count += 1
        wall_count = map_data["wall_count"]
        retry_info = f" (重试 {retries} 次)" if retries > 0 else ""

        # 提取起终点网格坐标（使用输出网格大小）
        out_gs = map_data["grid_size"]
        start_gx = int(map_data["start_pos"]["x"] / out_gs)
        start_gy = int(map_data["start_pos"]["y"] / out_gs)
        end_gx = int(map_data["end_pos"]["x"] / out_gs)
        end_gy = int(map_data["end_pos"]["y"] / out_gs)

        # 统计内部边墙数（不含外围边界 4 条）
        inner_walls = wall_count - 4

        print(f"[{i+1}/{args.count}] ✓ seed={final_seed} | "
              f"起点=({start_gx},{start_gy}) 终点=({end_gx},{end_gy}) | "
              f"最短路径 {path_length} 步 | 线段数 {wall_count}（内部 {inner_walls}）{retry_info}")

        # ASCII 预览（使用 blocked 网格渲染）
        if args.preview and blocked_grid is not None:
            print()
            print(ascii_preview_blocked(blocked_grid, out_dim_val, start_gx, start_gy, end_gx, end_gy))
            print()

        # 写入文件
        filename = f"maze_{final_seed}.json"
        output_path = os.path.join(save_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, indent=4, ensure_ascii=False)
        print(f"  → 已保存: {output_path}")

    # 汇总
    print(f"\n生成完成: 成功 {success_count}/{args.count} 张", end="")
    if total_retries > 0:
        print(f" (累计重试 {total_retries} 次)", end="")
    print(f"\n输出目录: {save_dir}")


if __name__ == "__main__":
    main()
