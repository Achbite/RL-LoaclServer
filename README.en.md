**🌍 Language / 语言**: [🇨🇳 中文](README.md) | [🇬🇧 English](README.en.md)

# Maze Reinforcement Learning Training Framework (RL-LocalServer)

> A local PPO reinforcement learning validation framework for training agents to navigate and solve 2D mazes.

---

## Project Overview

A **local reinforcement learning validation framework** that uses procedurally generated 2D mazes as training environments to validate the learning performance of the PPO algorithm on maze navigation tasks.

- **TrainClient** (C++): Parallel episode data collection, maze environment simulation, map JSON loading, ray perception
- **AIServer** (C++): Feature extraction, ONNX inference, reward computation, sample packaging
- **RL-Learner** (Python/PyTorch): PPO training, GAE computation, ONNX model export
- **Map Generator** (Python): DFS random maze generation with configurable difficulty, start/end point modes, batch generation, 100×100 grid resolution (101×101 output grid)
- **Visualization**: HTML Canvas player for browser-based replay of agent trajectories

All components run inside Docker containers, orchestrated via docker-compose, communicating through gRPC + Protobuf.

---

## System Architecture

```
TrainClient (C++, Multi-threaded Parallel)
├─ Thread 0: EpisodeWorker 0 ──┐
├─ Thread 1: EpisodeWorker 1 ──┤
├─ ...                         ├── gRPC (session_id isolation) ──→ AIServer (C++)
└─ Thread N: EpisodeWorker N ──┘                                   ├─ Feature Extraction (Nav 5D + Client Ray 8D = 13D)
                                                                   ├─ ONNX Inference
                                                                   ├─ Reward Computation
                                                                   ├─ Sample Packaging
                                                                   │       ↓ gRPC
                                                                   │   RL-Learner (PyTorch)
                                                                   │   ├─ GAE Computation
                                                                   │   ├─ PPO Training
                                                                   │   └─ ONNX Export → models/p2p/
                                                                   │           ↓ Polling
                                                                   └─ Model Hot-reload → Continue Inference

TrainClient → log/viz/*.jsonl → HTTP Replay → Browser Canvas Player (:9004)
```

**Training Loop**: TrainClient collects data (with ray observations) → AIServer inference + sample packaging → Learner PPO training → Export ONNX → AIServer hot-loads new model → Repeat

---

## Requirements

| Dependency     | Version       | Description           |
| -------------- | ------------- | --------------------- |
| Windows 10/11  | —            | Development host      |
| WSL2           | Ubuntu 22.04+ | Docker runtime        |
| Docker Engine  | 20.10+        | Containerized build   |
| Docker Compose | v2+           | Service orchestration |
| Browser        | Chrome/Edge   | Visualization replay  |

> All compilation and execution happen inside Docker containers — no local C++ toolchain or Python environment required.

---

## Quick Start

### 1. Environment Check

```bash
# Run inside WSL to verify Docker environment readiness
cd /mnt/e/RL-LoaclServer
bash check_env.sh

# Auto-fix missing dependencies
bash check_env.sh --fix
```

### 2. Start All Three Services

Start in the following order (each command in a separate WSL terminal):

```bash
cd /mnt/e/RL-LoaclServer

# Terminal 1: Start Learner (PPO Trainer)
bash docker_dev.sh learner
# Inside container:
./run.sh

# Terminal 2: Start AIServer (Inference + Sample Collection)
bash docker_dev.sh aiserver
# Inside container:
./build.sh && ./run_local_train.sh

# Terminal 3: Start TrainClient (Parallel Training Data Collection)
bash docker_dev.sh train
# Inside container:
./build.sh && ./run_train.sh
```

### 3. Monitor Training Progress

- **Training Dashboard**: `http://localhost:9005` (Learner Dashboard)
- **Visualization Replay**: `http://localhost:9004` (Agent trajectory replay)
- **Learner Console**: Observe loss decrease, entropy changes, and pass rate improvement

### 4. Inference Mode (Using a Trained Model)

```bash
# Start AIServer (Inference mode, run_mode=2)
bash docker_dev.sh aiserver
# Inside container:
./build.sh && ./run.sh

# Start TrainClient (Single episode debugging)
bash docker_dev.sh client
# Inside container:
./build.sh && ./run.sh
```

In inference mode, AIServer loads models from `models/local/` without connecting to Learner.

---

## Directory Structure

```
RL-LoaclServer/
├── TrainClient/                    # Parallel training data collection (C++17)
│   ├── main/                       # Entry points (main.cpp single debug / train_main.cpp parallel training)
│   ├── src/                        # Source code (config / env / grpc / log / pool / viz)
│   ├── configs/client_config.yaml  # TrainClient configuration
│   ├── maps/                       # Pre-generated map files (JSON, produced by generator)
│   ├── tools/                      # Map generator + visualization replay
│   │   ├── map_generator/          # DFS random maze generator
│   │   │   ├── generate_maze.py    # Generation script (supports difficulty, start/end modes, batch)
│   │   │   ├── map_config.yaml     # Generator config (size, difficulty, start/end)
│   │   │   └── maze_preview.html   # Browser-based map previewer
│   │   └── viz_player/             # Visualization replay
│   ├── build.sh / run.sh          # In-container build/run scripts
│   └── run_train.sh               # Parallel training run script
│
├── AIServer/                       # Inference + sample collection (C++)
│   ├── main/main.cpp              # gRPC service entry point
│   ├── src/                        # Source code (ai / config / grpc / log / session)
│   │   ├── ai/maze_reward.h/.cpp  # Reward calculator (standalone module)
│   │   ├── ai/onnx_inferencer.h   # ONNX Runtime inference wrapper
│   │   └── grpc/maze_service.h    # MazeService gRPC service
│   ├── configs/server_config.yaml # AIServer configuration
│   ├── models/                     # ONNX models (local/ manual, p2p/ auto-sync)
│   ├── build.sh / run.sh         # In-container build/run scripts
│   └── run_local_train.sh        # Training mode startup script
│
├── RL-Learner/                     # PPO training (Python/PyTorch)
│   ├── main/train.py              # Training entry point
│   ├── src/                        # Source code (grpc / training / metrics / log)
│   │   ├── training/ppo_trainer.py    # PPO trainer
│   │   └── training/sample_buffer.py  # Thread-safe sample buffer
│   ├── configs/learner_config.yaml    # Learner config (PPO hyperparameters)
│   ├── models/                     # Training output (p2p/ ONNX distribution, save/ checkpoint)
│   └── run.sh                     # In-container run script
│
├── docker/                         # Docker build files
│   ├── Dockerfile.client          # TrainClient container
│   ├── Dockerfile.aiserver        # AIServer container (with ONNX Runtime)
│   └── Dockerfile.learner         # Learner container (Python + PyTorch)
├── docker-compose.yml             # Service orchestration
├── docker_dev.sh                  # Unified entry script (build + enter container)
├── check_env.sh                   # Docker environment check & fix
└── clean.ps1                      # Windows local cleanup script (logs/model cache)
```

---

## Service Ports

| Port | Service     | Purpose                                  |
| ---- | ----------- | ---------------------------------------- |
| 9001 | TrainClient | Reserved                                 |
| 9002 | AIServer    | gRPC listener (TrainClient → AIServer)  |
| 9003 | Learner     | gRPC listener (AIServer → Learner)      |
| 9004 | Viz         | HTTP replay service (browser access)     |
| 9005 | Dashboard   | Training metrics panel (browser access)  |

---

## Command Reference

### docker_dev.sh Unified Entry

```bash
bash docker_dev.sh client              # Enter TrainClient container (single episode debug)
bash docker_dev.sh train               # Enter TrainClient container (parallel training mode)
bash docker_dev.sh aiserver            # Start AIServer + enter container
bash docker_dev.sh learner             # Start Learner + enter container
bash docker_dev.sh <service> --build   # Force rebuild image
bash docker_dev.sh <service> --restart # Restart existing container
```

### In-Container Operations

```bash
./build.sh              # Compile C++ project
./run.sh                # Run (single episode debug / AIServer inference mode)
./run_train.sh          # Parallel training (TrainClient only)
./run_local_train.sh    # Training mode startup (AIServer only, run_mode=1)
```

### Docker Compose Direct Operations

```bash
docker compose up -d maze-aiserver     # Start AIServer in background
docker compose up -d maze-learner      # Start Learner in background
docker compose logs -f maze-learner    # View Learner logs
docker compose stop                    # Stop all services
docker compose down                    # Stop and remove all containers
```

### Local Cleanup (Windows PowerShell)

```powershell
# Clean training logs and model cache
powershell -ExecutionPolicy Bypass -File .\clean.ps1
```

---

## Run Modes

AIServer behavior is controlled by `run_mode` in `server_config.yaml`:

| run_mode | Name     | Policy Behavior                        | Model Source              | Sample Collection |
| -------- | -------- | -------------------------------------- | ------------------------- | ----------------- |
| 1        | Training | ONNX inference (random if no model)    | `models/p2p/` auto-poll  | ✅                |
| 2        | Inference| ONNX inference                         | `models/local/` priority | ❌                |
| 3        | A* Test  | A* pathfinding                         | No model needed           | ❌                |

---

## Core Training Parameters

### PPO Hyperparameters (`learner_config.yaml`)

| Parameter       | Default | Description              |
| --------------- | ------- | ------------------------ |
| learning_rate   | 3e-4    | Adam learning rate       |
| gamma           | 0.99    | Discount factor          |
| gae_lambda      | 0.95    | GAE lambda               |
| clip_epsilon    | 0.2     | PPO clipping coefficient |
| entropy_coef    | 0.01    | Entropy regularization   |
| n_epochs        | 4       | Epochs per batch         |
| mini_batch_size | 64      | Mini-batch size          |

### Reward Function

| Reward Item       | Type   | Value           | Description                                              |
| ----------------- | ------ | --------------- | -------------------------------------------------------- |
| Goal Reward       | Sparse | +10.0           | Reaching the goal                                        |
| Ray Detection     | Percep | 8-dir normalized| Computed on Client side, sent with UpdateReq to AIServer |
| Potential Shaping | Dense  | γΦ(s')-Φ(s)   | Distance-based guidance to prevent reward hacking (Ng 1999) |
| Exploration Bonus | Dense  | +0.05           | First visit to a new grid cell                           |
| Loitering Penalty | Dense  | -0.02           | Revisiting the same cell within 8 steps                  |
| Timeout Penalty   | Sparse | -2.0            | Failed to reach goal before timeout                      |
| Ranking Reward    | Sparse | ±3~8           | Multi-agent competitive ranking                          |

### Parallel Training Parameters (`client_config.yaml`)

| Parameter         | Default | Description                    |
| ----------------- | ------- | ------------------------------ |
| thread_count      | 4       | Parallel episode thread count  |
| max_episodes      | 1000    | Total training episodes        |
| sample_batch_size | 32      | Sample frames per batch (TMax) |

---

## Visualization

During training, TrainClient automatically records agent trajectories to `log/viz/*.jsonl`.

1. After training starts, open `http://localhost:9004` in your browser
2. Select a replay file from the dropdown
3. Supports play/pause/step/speed control
4. Displays maze walls + agent position + trajectory path + color-gradient timeline

---

## Tech Stack

| Component              | Technology       | Version     |
| ---------------------- | ---------------- | ----------- |
| TrainClient / AIServer | C++17            | GCC 13+     |
| Learner                | Python + PyTorch | 3.11 / 2.0+ |
| Communication          | gRPC + Protobuf  | v3.21       |
| Inference              | ONNX Runtime     | 1.17.0      |
| Build System           | CMake + Ninja    | 3.28+       |
| Containerization       | Docker + Compose | v2          |
| Visualization          | HTML5 Canvas     | —          |

---

## License

This project is open-sourced under the [MIT License](LICENSE).

Copyright © 2026 AChbite
