# Multi-UAV-TA Framework (v02.0)

Multi-UAV-TA is an open-source framework that implements a custom environment for training and evaluating algorithms (including Multi-Agent Reinforcement Learning - MARL) in Multi-UAV (Unmanned Aerial Vehicle) Target Assignment scenarios. 

The environment is designed to simulate multiple UAVs working together to assign, track, and engage targets in real-time. It is a valuable resource for researchers and engineers working on UAV coordination, multi-agent systems, and aerial surveillance applications.

**Version 02.0 Major Updates**:
- **High-Performance Rust Core**: Mathematical and physics bottlenecks (like Line of Sight and Obstacle Avoidance) have been rewritten in compiled Rust (`core_sim`) using `PyO3`, achieving massive speedups for training.
- **Strict Headless Training Support**: `pygame` has been completely stripped from the core simulation loop. The PettingZoo environment is now natively headless, making it extremely efficient for RL frameworks like `tianshou` or `ray[rllib]`.
- **Decoupled 3D Web Visualization**: A new FastAPI backend (`server/`) streams live simulation data via WebSockets to a React + Vite + Three Fiber frontend (`frontend/`), allowing browser-based, beautiful 3D monitoring without blocking the simulation threads.

## Architecture

1. **Environment (`mUAV_TA/DroneEnv.py`)**: A PettingZoo Parallel API environment handling the Multi-Agent state, steps, and rewards.
2. **Rust Core (`core_sim/`)**: A compiled Rust library wrapping physics logic. Called synchronously from the Python environment.
3. **API Backend (`server/api.py`)**: A FastAPI WebSocket server that instantiates the environment and streams state to the frontend.
4. **3D Frontend (`frontend/`)**: A modern React-Three-Fiber visualization dashboard.

## Installation

### 1. Python Environment Setup
We recommend using a virtual environment (Python 3.10+):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

Install the required RL and Gym dependencies:
```bash
pip install gymnasium pettingzoo tianshou torch numpy
```

### 2. Build the Rust Core
You need the Rust toolchain installed (`cargo`). Then, build the Python bindings using `maturin`:
```bash
pip install maturin
cd core_sim
maturin develop --release
cd ..
```

### 3. Frontend Setup
Make sure you have Node.js installed.
```bash
cd frontend
npm install
```

## Running the 3D Visualization

The dashboard plays a deterministic WPS_commit log with dual-region bursts,
online arrivals, local sensing, delayed knowledge, hard windows, UAV failures,
commit locks, replanning, and the S_WPS metrics.

Generate (or regenerate) the example replay:
```bash
.venv\Scripts\python.exe experiments\generate_simulation_replay.py --seed 0
```

Then use two terminals.

**Terminal 1: Start the API Backend**
```bash
.venv\Scripts\activate
python server\api.py
```
*Runs on `http://localhost:8000`*

**Terminal 2: Start the Web Dashboard**
```bash
cd frontend
npm run dev
```
*Visit the localhost URL provided by Vite in your browser.*

The dashboard supports play/pause, scrubbing, speed control, an event timeline,
assignment and sensing overlays, and JSON-log download. The generated log is
stored at `experiments/results/wps_commit_replay.json`.

## AI Development Guidelines
If you are using AI coding assistants to develop this framework further, please refer to the `AI_DEVELOPMENT_GUIDE.md` for architectural context and instructions.
