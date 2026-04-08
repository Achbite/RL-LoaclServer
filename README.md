**🌍 Language / 语言**: [🇨🇳 中文](README.md) | [🇬🇧 English](README.en.md)

# 迷宫强化学习训练框架（RL-LocalServer）

> 本地 PPO 强化学习验证框架，用于在 2D 迷宫环境中训练 Agent 学会寻路通关。

---

## 项目简介

一个**本地强化学习验证框架**，使用程序化生成的 2D 迷宫作为训练环境，验证 PPO 算法在迷宫寻路任务上的学习效果。

- **TrainClient**（C++）：并行 Episode 数据采集，迷宫环境模拟，地图 JSON 加载，射线感知
- **AIServer**（C++）：特征提取、ONNX 推理、奖励计算、样本打包
- **RL-Learner**（Python/PyTorch）：PPO 训练、GAE 计算、ONNX 模型导出
- **地图生成器**（Python）：DFS 随机迷宫生成，支持难度、起终点模式、批量生成，100×100 网格分辨率（101×101 输出网格）
- **可视化**：HTML Canvas 播放器，浏览器回放 Agent 行走轨迹

全部组件在 Docker 容器中运行，通过 docker-compose 编排，gRPC + Protobuf 通信。

---

## 系统架构

```
TrainClient (C++, 多线程并行)
├─ Thread 0: EpisodeWorker 0 ──┐
├─ Thread 1: EpisodeWorker 1 ──┤
├─ ...                         ├── gRPC (session_id 隔离) ──→ AIServer (C++)
└─ Thread N: EpisodeWorker N ──┘                              ├─ 特征提取 (导航5维 + Client射线8维 = 13维)
                                                              ├─ ONNX 推理
                                                              ├─ 奖励计算
                                                              ├─ 样本打包
                                                              │       ↓ gRPC
                                                              │   RL-Learner (PyTorch)
                                                              │   ├─ GAE 计算
                                                              │   ├─ PPO 训练
                                                              │   └─ ONNX 导出 → models/p2p/
                                                              │           ↓ 轮询拉取
                                                              └─ 模型热更新 → 继续推理

TrainClient → log/viz/*.jsonl → HTTP 回放 → 浏览器 Canvas 播放器 (:9004)
```

**训练闭环**：TrainClient 采集数据（含射线观测）→ AIServer 推理+打包样本 → Learner PPO 训练 → 导出 ONNX → AIServer 热加载新模型 → 循环

---

## 环境要求

| 依赖           | 版本要求      | 说明            |
| -------------- | ------------- | --------------- |
| Windows 10/11  | —            | 开发宿主机      |
| WSL2           | Ubuntu 22.04+ | Docker 运行环境 |
| Docker Engine  | 20.10+        | 容器化构建运行  |
| Docker Compose | v2+           | 服务编排        |
| 浏览器         | Chrome/Edge   | 可视化回放      |

> 所有编译和运行均在 Docker 容器内完成，无需本地安装 C++ 工具链或 Python 环境。

---

## 快速上手

### 1. 环境检测

```bash
# 在 WSL 内执行，检测 Docker 环境是否就绪
cd /mnt/e/RL-LoaclServer
bash check_env.sh

# 如有缺失依赖，自动修复
bash check_env.sh --fix
```

### 2. 启动三端服务

按以下顺序启动（每个命令在独立的 WSL 终端中执行）：

```bash
cd /mnt/e/RL-LoaclServer

# 终端 1：启动 Learner（PPO 训练器）
bash docker_dev.sh learner
# 进入容器后：
./run.sh

# 终端 2：启动 AIServer（推理 + 样本收集）
bash docker_dev.sh aiserver
# 进入容器后：
./build.sh && ./run_local_train.sh

# 终端 3：启动 TrainClient（并行训练数据采集）
bash docker_dev.sh train
# 进入容器后：
./build.sh && ./run_train.sh
```

### 3. 观察训练效果

- **训练指标面板**：`http://localhost:9005`（Learner 端 Dashboard）
- **可视化回放**：`http://localhost:9004`（Agent 行走轨迹回放）
- **Learner 控制台**：观察 loss 下降、entropy 变化、通关率提升

### 4. 推理模式（使用训练好的模型）

```bash
# 启动 AIServer（推理模式，run_mode=2）
bash docker_dev.sh aiserver
# 进入容器后：
./build.sh && ./run.sh

# 启动 TrainClient（单 Episode 调试）
bash docker_dev.sh client
# 进入容器后：
./build.sh && ./run.sh
```

推理模式下 AIServer 从 `models/local/` 加载模型，不连接 Learner。

---

## 目录结构

```
RL-LoaclServer/
├── TrainClient/                    # 并行训练数据采集（C++17）
│   ├── main/                       # 主入口（main.cpp 单调试 / train_main.cpp 并行训练）
│   ├── src/                        # 源码（config / env / grpc / log / pool / viz）
│   ├── configs/client_config.yaml  # TrainClient 配置
│   ├── maps/                       # 预生成地图文件（JSON，由生成器产出）
│   ├── tools/                      # 地图生成器 + 可视化回放
│   │   ├── map_generator/          # DFS 随机迷宫生成器
│   │   │   ├── generate_maze.py    # 生成脚本（支持难度、起终点模式、批量）
│   │   │   ├── map_config.yaml     # 生成器配置（尺寸、难度、起终点）
│   │   │   └── maze_preview.html   # 浏览器端地图预览器
│   │   └── viz_player/             # 可视化回放
│   ├── build.sh / run.sh          # 容器内编译/运行脚本
│   └── run_train.sh               # 并行训练运行脚本
│
├── AIServer/                       # 推理 + 样本收集（C++）
│   ├── main/main.cpp              # gRPC 服务主入口
│   ├── src/                        # 源码（ai / config / grpc / log / session）
│   │   ├── ai/maze_reward.h/.cpp  # 奖励计算器（独立模块）
│   │   ├── ai/onnx_inferencer.h   # ONNX Runtime 推理封装
│   │   └── grpc/maze_service.h    # MazeService gRPC 服务
│   ├── configs/server_config.yaml # AIServer 配置
│   ├── models/                     # ONNX 模型（local/ 手动放置，p2p/ 自动同步）
│   ├── build.sh / run.sh         # 容器内编译/运行脚本
│   └── run_local_train.sh        # 训练模式启动脚本
│
├── RL-Learner/                     # PPO 训练（Python/PyTorch）
│   ├── main/train.py              # 训练主入口
│   ├── src/                        # 源码（grpc / training / metrics / log）
│   │   ├── training/ppo_trainer.py    # PPO 训练器
│   │   └── training/sample_buffer.py  # 线程安全样本缓存
│   ├── configs/learner_config.yaml    # Learner 配置（PPO 超参数）
│   ├── models/                     # 训练产出（p2p/ ONNX 分发，save/ checkpoint）
│   └── run.sh                     # 容器内运行脚本
│
├── docker/                         # Docker 构建文件
│   ├── Dockerfile.client          # TrainClient 容器
│   ├── Dockerfile.aiserver        # AIServer 容器（含 ONNX Runtime）
│   └── Dockerfile.learner         # Learner 容器（Python + PyTorch）
├── docker-compose.yml             # 服务编排
├── docker_dev.sh                  # 统一入口脚本（构建 + 进入容器）
├── check_env.sh                   # Docker 环境检测与修复
└── clean.ps1                      # Windows 本地清理脚本（日志/模型缓存）
```

---

## 服务端口

| 端口 | 服务        | 用途                                 |
| ---- | ----------- | ------------------------------------ |
| 9001 | TrainClient | 预留                                 |
| 9002 | AIServer    | gRPC 监听（TrainClient → AIServer） |
| 9003 | Learner     | gRPC 监听（AIServer → Learner）     |
| 9004 | 可视化      | HTTP 回放服务（浏览器访问）          |
| 9005 | Dashboard   | 训练指标面板（浏览器访问）           |

---

## 常用命令速查

### docker_dev.sh 统一入口

```bash
bash docker_dev.sh client              # 进入 TrainClient 容器（单 Episode 调试）
bash docker_dev.sh train               # 进入 TrainClient 容器（并行训练模式）
bash docker_dev.sh aiserver            # 启动 AIServer + 进入容器
bash docker_dev.sh learner             # 启动 Learner + 进入容器
bash docker_dev.sh <服务名> --build    # 强制重新构建镜像
bash docker_dev.sh <服务名> --restart  # 重启已有容器
```

### 容器内操作

```bash
./build.sh              # 编译 C++ 项目
./run.sh                # 运行（单 Episode 调试 / AIServer 推理模式）
./run_train.sh          # 并行训练（TrainClient 专用）
./run_local_train.sh    # 训练模式启动（AIServer 专用，run_mode=1）
```

### Docker Compose 直接操作

```bash
docker compose up -d maze-aiserver     # 后台启动 AIServer
docker compose up -d maze-learner      # 后台启动 Learner
docker compose logs -f maze-learner    # 查看 Learner 日志
docker compose stop                    # 停止所有服务
docker compose down                    # 停止并清理所有容器
```

### 本地清理（Windows PowerShell）

```powershell
# 清理训练日志和模型缓存
powershell -ExecutionPolicy Bypass -File .\clean.ps1
```

---

## 运行模式

AIServer 通过 `server_config.yaml` 中的 `run_mode` 控制行为：

| run_mode | 名称   | 策略行为                      | 模型来源                 | 样本收集 |
| -------- | ------ | ----------------------------- | ------------------------ | -------- |
| 1        | 训练   | ONNX 推理（无模型时随机策略） | `models/p2p/` 自动轮询 | ✅       |
| 2        | 推理   | ONNX 推理                     | `models/local/` 优先   | ❌       |
| 3        | A*测试 | A* 寻路                       | 不需要模型               | ❌       |

---

## 训练核心参数

### PPO 超参数（`learner_config.yaml`）

| 参数            | 默认值 | 说明             |
| --------------- | ------ | ---------------- |
| learning_rate   | 3e-4   | Adam 学习率      |
| gamma           | 0.99   | 折扣因子         |
| gae_lambda      | 0.95   | GAE lambda       |
| clip_epsilon    | 0.2    | PPO 裁剪系数     |
| entropy_coef    | 0.01   | 熵正则化系数     |
| n_epochs        | 4      | 每批数据训练轮数 |
| mini_batch_size | 64     | mini-batch 大小  |

### 奖励函数

| 奖励项   | 类型 | 值             | 说明                          |
| -------- | ---- | -------------- | ----------------------------- |
| 通关奖励 | 稀疏 | +10.0          | 到达终点                      |
| 射线检测 | 感知 | 8方向归一化距离 | Client 端计算，随 UpdateReq 传给 AIServer |
| 势能引导 | 密集 | γΦ(s')-Φ(s) | 防刷奖励的距离引导（Ng 1999） |
| 探索奖励 | 密集 | +0.05          | 首次访问新网格                |
| 徘徊惩罚 | 密集 | -0.02          | 8 步内重复访问同一网格        |
| 超时惩罚 | 稀疏 | -2.0           | 未通关超时                    |
| 排名奖励 | 稀疏 | ±3~8          | 多 Agent 竞争排名             |

### 并行训练参数（`client_config.yaml`）

| 参数              | 默认值 | 说明                 |
| ----------------- | ------ | -------------------- |
| thread_count      | 4      | 并行 Episode 线程数  |
| max_episodes      | 1000   | 总训练 Episode 数    |
| sample_batch_size | 32     | 每批样本帧数（TMax） |

---

## 可视化

训练过程中 TrainClient 自动记录 Agent 行走轨迹到 `log/viz/*.jsonl`。

1. 训练启动后，浏览器访问 `http://localhost:9004`
2. 下拉列表选择回放文件
3. 支持播放/暂停/步进/速度调节
4. 迷宫墙壁 + Agent 位置 + 行走轨迹 + 颜色渐变时间线

---

## 技术栈

| 组件                   | 技术             | 版本        |
| ---------------------- | ---------------- | ----------- |
| TrainClient / AIServer | C++17            | GCC 13+     |
| Learner                | Python + PyTorch | 3.11 / 2.0+ |
| 通信                   | gRPC + Protobuf  | v3.21       |
| 推理                   | ONNX Runtime     | 1.17.0      |
| 构建                   | CMake + Ninja    | 3.28+       |
| 容器                   | Docker + Compose | v2          |
| 可视化                 | HTML5 Canvas     | —          |

---

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。

Copyright © 2026 AChbite
