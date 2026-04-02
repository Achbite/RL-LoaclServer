#!/usr/bin/env python3
"""
随机迷宫地图生成器

使用 DFS（深度优先搜索）算法生成随机迷宫，输出与 TrainClient 兼容的 JSON 格式。
支持种子控制、批量生成、BFS 可达性验证、ASCII 终端预览。

目录约定：
  maps/test/  — 实际 client 初始化随机读取的地图
  maps/save/  — 保存的生成地图（手动预览用）

用法：
  # 生成单张地图到 save 目录（预览用）
  python generate_maze.py --seed 42 --output ../../maps/save/maze_42.json

  # 生成单张地图到 test 目录（client 使用）
  python generate_maze.py --seed 42 --output ../../maps/test/maze_42.json

  # 批量生成到 test 目录
  python generate_maze.py --seed-range 0 10 --output-dir ../../maps/test/

  # 批量生成到 save 目录
  python generate_maze.py --seed-range 0 10 --output-dir ../../maps/save/

  # 终端 ASCII 预览（不写文件）
  python generate_maze.py --seed 42 --preview

  # 生成并同时预览
  python generate_maze.py --seed 42 --output ../../maps/save/maze_42.json --preview
"""

import argparse
import json
import math
import os
import random
import sys
from collections import deque


# ============================================================
# 常量定义
# ============================================================

MAP_SIZE = 20000        # 地图尺寸（cm）
GRID_SIZE = 500         # 网格大小（cm）
GRID_DIM = 40           # 网格维度（40×40）
WALL_THICKNESS = 100    # 墙壁厚度（cm）

# 房间粒度：2×2 网格为一个房间（1 格通道 + 1 格墙壁）
ROOM_SIZE = 2
ROOM_DIM = GRID_DIM // ROOM_SIZE  # 20×20 房间网格

# 额外打通比例（DFS 生成后随机打通的额外墙壁，增加多路径）
EXTRA_OPEN_RATIO = 0.10

# 起点和终点（网格坐标）
START_GX, START_GY = 1, 1
END_GX, END_GY = 39, 39

# 起点和终点（连续坐标，网格中心）
START_POS = {"x": (START_GX + 0.5) * GRID_SIZE, "y": (START_GY + 0.5) * GRID_SIZE}
END_POS = {"x": (END_GX + 0.5) * GRID_SIZE, "y": (END_GY + 0.5) * GRID_SIZE}

# 四方向偏移（上、右、下、左）
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


# ============================================================
# DFS 迷宫生成
# ============================================================

def generate_maze_dfs(room_dim, rng):
    """
    DFS 随机迷宫生成

    在 room_dim × room_dim 的房间网格上，使用 DFS 打通房间之间的墙壁。
    返回水平墙壁和垂直墙壁的状态矩阵（True=墙壁存在，False=已打通）。

    参数：
        room_dim: 房间网格维度（如 20 表示 20×20）
        rng: random.Random 实例

    返回：
        h_walls: 水平墙壁 [room_dim][room_dim-1]，h_walls[x][y] 表示房间 (x,y) 和 (x,y+1) 之间
        v_walls: 垂直墙壁 [room_dim-1][room_dim]，v_walls[x][y] 表示房间 (x,y) 和 (x+1,y) 之间
    """
    # 初始化所有墙壁为存在
    h_walls = [[True] * (room_dim - 1) for _ in range(room_dim)]
    v_walls = [[True] * room_dim for _ in range(room_dim - 1)]

    visited = [[False] * room_dim for _ in range(room_dim)]
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        cx, cy = stack[-1]

        # 收集未访问的邻居
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < room_dim and 0 <= ny < room_dim and not visited[nx][ny]:
                neighbors.append((nx, ny, dx, dy))

        if not neighbors:
            stack.pop()
            continue

        # 随机选择一个邻居
        nx, ny, dx, dy = rng.choice(neighbors)
        visited[nx][ny] = True

        # 打通墙壁
        if dx == 1:     # 向右：打通垂直墙壁
            v_walls[cx][cy] = False
        elif dx == -1:  # 向左：打通垂直墙壁
            v_walls[nx][ny] = False
        elif dy == 1:   # 向上：打通水平墙壁
            h_walls[cx][cy] = False
        elif dy == -1:  # 向下：打通水平墙壁
            h_walls[cx][ny] = False

        stack.append((nx, ny))

    return h_walls, v_walls


def extra_open_walls(h_walls, v_walls, room_dim, ratio, rng):
    """
    额外随机打通墙壁，增加多路径

    参数：
        h_walls: 水平墙壁矩阵
        v_walls: 垂直墙壁矩阵
        room_dim: 房间网格维度
        ratio: 打通比例（0.0 ~ 1.0）
        rng: random.Random 实例
    """
    # 收集所有仍然存在的墙壁
    existing_walls = []
    for x in range(room_dim):
        for y in range(room_dim - 1):
            if h_walls[x][y]:
                existing_walls.append(('h', x, y))
    for x in range(room_dim - 1):
        for y in range(room_dim):
            if v_walls[x][y]:
                existing_walls.append(('v', x, y))

    # 随机打通指定比例
    count = int(len(existing_walls) * ratio)
    if count > 0:
        to_open = rng.sample(existing_walls, min(count, len(existing_walls)))
        for wall_type, x, y in to_open:
            if wall_type == 'h':
                h_walls[x][y] = False
            else:
                v_walls[x][y] = False


# ============================================================
# 墙壁转换：房间网格 → 连续坐标线段
# ============================================================

def walls_to_segments(h_walls, v_walls, room_dim, room_size, grid_size):
    """
    将房间网格的墙壁状态转换为连续坐标线段

    房间 (rx, ry) 对应网格区域 [rx*room_size, (rx+1)*room_size) × [ry*room_size, (ry+1)*room_size)
    水平墙壁 h_walls[rx][ry]：房间 (rx,ry) 和 (rx,ry+1) 之间，即 y = (ry+1)*room_size 的水平线
    垂直墙壁 v_walls[rx][ry]：房间 (rx,ry) 和 (rx+1,ry) 之间，即 x = (rx+1)*room_size 的垂直线

    参数：
        h_walls: 水平墙壁矩阵
        v_walls: 垂直墙壁矩阵
        room_dim: 房间网格维度
        room_size: 房间粒度（网格数）
        grid_size: 网格大小（cm）

    返回：
        walls: 墙壁线段列表 [{"x1", "y1", "x2", "y2", "thickness"}]
    """
    walls = []

    # 水平墙壁（沿 Y 方向分隔的水平线段）
    for rx in range(room_dim):
        for ry in range(room_dim - 1):
            if h_walls[rx][ry]:
                wall_y = (ry + 1) * room_size * grid_size
                wall_x1 = rx * room_size * grid_size
                wall_x2 = (rx + 1) * room_size * grid_size
                walls.append({
                    "x1": int(wall_x1), "y1": int(wall_y),
                    "x2": int(wall_x2), "y2": int(wall_y),
                    "thickness": WALL_THICKNESS
                })

    # 垂直墙壁（沿 X 方向分隔的垂直线段）
    for rx in range(room_dim - 1):
        for ry in range(room_dim):
            if v_walls[rx][ry]:
                wall_x = (rx + 1) * room_size * grid_size
                wall_y1 = ry * room_size * grid_size
                wall_y2 = (ry + 1) * room_size * grid_size
                walls.append({
                    "x1": int(wall_x), "y1": int(wall_y1),
                    "x2": int(wall_x), "y2": int(wall_y2),
                    "thickness": WALL_THICKNESS
                })

    return walls


def merge_collinear_walls(walls):
    """
    合并共线且相邻/重叠的墙壁线段，减少 JSON 体积

    将同一条直线上的连续线段合并为一条长线段。
    """
    # 分离水平和垂直墙壁
    h_groups = {}  # key: y 坐标, value: [(x_min, x_max)]
    v_groups = {}  # key: x 坐标, value: [(y_min, y_max)]

    for w in walls:
        if w["y1"] == w["y2"]:  # 水平墙壁
            y = w["y1"]
            x_min = min(w["x1"], w["x2"])
            x_max = max(w["x1"], w["x2"])
            h_groups.setdefault(y, []).append((x_min, x_max))
        elif w["x1"] == w["x2"]:  # 垂直墙壁
            x = w["x1"]
            y_min = min(w["y1"], w["y2"])
            y_max = max(w["y1"], w["y2"])
            v_groups.setdefault(x, []).append((y_min, y_max))
        else:
            pass  # 斜线墙壁不合并（当前不会出现）

    merged = []

    # 合并水平墙壁
    for y, segments in h_groups.items():
        segments.sort()
        cur_min, cur_max = segments[0]
        for s_min, s_max in segments[1:]:
            if s_min <= cur_max:  # 相邻或重叠
                cur_max = max(cur_max, s_max)
            else:
                merged.append({
                    "x1": cur_min, "y1": y,
                    "x2": cur_max, "y2": y,
                    "thickness": WALL_THICKNESS
                })
                cur_min, cur_max = s_min, s_max
        merged.append({
            "x1": cur_min, "y1": y,
            "x2": cur_max, "y2": y,
            "thickness": WALL_THICKNESS
        })

    # 合并垂直墙壁
    for x, segments in v_groups.items():
        segments.sort()
        cur_min, cur_max = segments[0]
        for s_min, s_max in segments[1:]:
            if s_min <= cur_max:  # 相邻或重叠
                cur_max = max(cur_max, s_max)
            else:
                merged.append({
                    "x1": x, "y1": cur_min,
                    "x2": x, "y2": cur_max,
                    "thickness": WALL_THICKNESS
                })
                cur_min, cur_max = s_min, s_max
        merged.append({
            "x1": x, "y1": cur_min,
            "x2": x, "y2": cur_max,
            "thickness": WALL_THICKNESS
        })

    return merged


# ============================================================
# BFS 可达性验证
# ============================================================

def build_blocked_grid(walls, grid_dim, grid_size):
    """
    将墙壁线段映射到网格 blocked 数组（与 C++ AddWallToGrid 逻辑一致）

    参数：
        walls: 墙壁线段列表
        grid_dim: 网格维度
        grid_size: 网格大小（cm）

    返回：
        blocked: grid_dim × grid_dim 的二维列表，True=不可通行
    """
    blocked = [[False] * grid_dim for _ in range(grid_dim)]

    for w in walls:
        half_t = w["thickness"] * 0.5
        min_x = min(w["x1"], w["x2"]) - half_t
        max_x = max(w["x1"], w["x2"]) + half_t
        min_y = min(w["y1"], w["y2"]) - half_t
        max_y = max(w["y1"], w["y2"]) + half_t

        gx_min = max(0, int(math.floor(min_x / grid_size)))
        gx_max = min(grid_dim - 1, int(math.floor(max_x / grid_size)))
        gy_min = max(0, int(math.floor(min_y / grid_size)))
        gy_max = min(grid_dim - 1, int(math.floor(max_y / grid_size)))

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                blocked[gy][gx] = True

    return blocked


def bfs_reachable(blocked, grid_dim, sx, sy, ex, ey):
    """
    BFS 验证从 (sx, sy) 到 (ex, ey) 是否可达

    参数：
        blocked: 网格 blocked 数组
        grid_dim: 网格维度
        sx, sy: 起点网格坐标
        ex, ey: 终点网格坐标

    返回：
        (reachable, path_length): 是否可达，最短路径长度（-1 表示不可达）
    """
    if blocked[sy][sx] or blocked[ey][ex]:
        return False, -1

    visited = [[False] * grid_dim for _ in range(grid_dim)]
    visited[sy][sx] = True
    queue = deque([(sx, sy, 0)])

    while queue:
        cx, cy, dist = queue.popleft()
        if cx == ex and cy == ey:
            return True, dist

        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_dim and 0 <= ny < grid_dim:
                if not blocked[ny][nx] and not visited[ny][nx]:
                    visited[ny][nx] = True
                    queue.append((nx, ny, dist + 1))

    return False, -1


# ============================================================
# ASCII 预览
# ============================================================

def ascii_preview(blocked, grid_dim, start_gx, start_gy, end_gx, end_gy):
    """
    终端 ASCII 预览地图

    ██ = 墙壁（blocked）
    S  = 起点
    E  = 终点
    ·  = 可通行
    """
    lines = []
    # 从上到下打印（y 从大到小，因为终端从上往下）
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

def generate_map(seed, extra_open_ratio=EXTRA_OPEN_RATIO):
    """
    生成一张随机迷宫地图

    参数：
        seed: 随机种子
        extra_open_ratio: 额外打通墙壁比例

    返回：
        map_data: 地图 JSON 字典
        blocked: 网格 blocked 数组（用于预览和验证）
        reachable: 是否可达
        path_length: BFS 最短路径长度
    """
    rng = random.Random(seed)

    # 1. DFS 生成迷宫骨架
    h_walls, v_walls = generate_maze_dfs(ROOM_DIM, rng)

    # 2. 额外打通墙壁
    extra_open_walls(h_walls, v_walls, ROOM_DIM, extra_open_ratio, rng)

    # 3. 转换为连续坐标线段
    raw_walls = walls_to_segments(h_walls, v_walls, ROOM_DIM, ROOM_SIZE, GRID_SIZE)

    # 4. 合并共线墙壁
    merged_walls = merge_collinear_walls(raw_walls)

    # 5. 构建 blocked 网格并验证可达性
    blocked = build_blocked_grid(merged_walls, GRID_DIM, GRID_SIZE)

    # 确保起点和终点可通行
    blocked[START_GY][START_GX] = False
    blocked[END_GY][END_GX] = False

    reachable, path_length = bfs_reachable(
        blocked, GRID_DIM, START_GX, START_GY, END_GX, END_GY
    )

    # 6. 组装地图数据
    map_data = {
        "map_id": f"maze_seed_{seed}",
        "version": 1,
        "seed": seed,
        "bounds": {"x_min": 0, "x_max": MAP_SIZE, "y_min": 0, "y_max": MAP_SIZE},
        "start_pos": {"x": int(START_POS["x"]), "y": int(START_POS["y"])},
        "end_pos": {"x": int(END_POS["x"]), "y": int(END_POS["y"])},
        "walls": merged_walls
    }

    return map_data, blocked, reachable, path_length


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="随机迷宫地图生成器（DFS 算法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成单张地图到 save 目录
  python generate_maze.py --seed 42 --output ../../maps/save/maze_42.json

  # 生成单张地图到 test 目录（client 使用）
  python generate_maze.py --seed 42 --output ../../maps/test/maze_42.json

  # 批量生成到 test 目录（种子 0-9）
  python generate_maze.py --seed-range 0 10 --output-dir ../../maps/test/

  # 终端 ASCII 预览
  python generate_maze.py --seed 42 --preview

  # 生成并预览
  python generate_maze.py --seed 42 --output ../../maps/save/maze_42.json --preview
        """
    )

    # 种子参数（二选一）
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="指定随机种子（生成单张地图）")
    seed_group.add_argument("--seed-range", type=int, nargs=2, metavar=("START", "END"),
                            help="种子范围 [START, END)（批量生成）")

    # 输出参数（二选一）
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=str, help="输出文件路径（单张地图）")
    output_group.add_argument("--output-dir", type=str, help="输出目录（批量生成）")

    # 其他参数
    parser.add_argument("--preview", action="store_true", help="终端 ASCII 预览")
    parser.add_argument("--extra-open", type=float, default=EXTRA_OPEN_RATIO,
                        help=f"额外打通墙壁比例（默认 {EXTRA_OPEN_RATIO}）")

    args = parser.parse_args()

    # 确定种子列表
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))

    # 批量生成时必须指定输出目录
    if len(seeds) > 1 and not args.output_dir:
        parser.error("批量生成（--seed-range）必须指定 --output-dir")

    # 生成地图
    success_count = 0
    fail_count = 0

    for seed in seeds:
        map_data, blocked, reachable, path_length = generate_map(seed, args.extra_open)

        # 状态输出
        status = "✓ 可达" if reachable else "✗ 不可达"
        path_info = f"最短路径 {path_length} 步" if reachable else "无法到达终点"
        wall_count = len(map_data["walls"])
        print(f"[seed={seed:>6d}] {status} | {path_info} | 墙壁数 {wall_count}")

        if not reachable:
            fail_count += 1
            print(f"  ⚠ 警告：seed={seed} 生成的地图不可达，跳过保存")
            continue

        success_count += 1

        # ASCII 预览
        if args.preview:
            print()
            print(ascii_preview(blocked, GRID_DIM, START_GX, START_GY, END_GX, END_GY))
            print()

        # 写入文件
        output_path = None
        if args.output and len(seeds) == 1:
            output_path = args.output
        elif args.output_dir:
            output_path = os.path.join(args.output_dir, f"maze_seed_{seed}.json")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=4, ensure_ascii=False)
            print(f"  → 已保存: {output_path}")

    # 汇总
    print(f"\n生成完成: 成功 {success_count} 张, 失败 {fail_count} 张")


if __name__ == "__main__":
    main()
