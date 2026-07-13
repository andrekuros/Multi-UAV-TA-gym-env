# Multi-UAV-TA: AI Development Guide

If you are an AI assistant helping a user develop this repository, please read this document carefully to understand the v02.0 architectural paradigms. 

## 1. Project Architecture
The project is split into three highly decoupled domains:
1. **The Core RL Environment (Python)**: Found in `mUAV_TA/`. This is a strict `PettingZoo` Parallel API environment (`DroneEnv.py`). **Rule**: It must NEVER contain rendering loops, `pygame` imports, or UI blocking code. It must remain purely headless for maximum RL training performance.
2. **The Physics Engine (Rust)**: Found in `core_sim/`. High-frequency math operations (like obstacle avoidance, line-of-sight checks, distance matrices) are delegated to this Rust module. It is bound to Python using `PyO3`. **Rule**: If you encounter a Python `for` loop over many agents calculating geometry, port it to `core_sim.rs` and expose it via PyO3.
3. **The 3D Visualization (React/ThreeJS + FastAPI)**: Found in `server/api.py` and `frontend/`. 
   - The backend runs a `FastAPI` WebSocket server that iterates the environment and broadcasts JSON state. 
   - The frontend is a React app using `@react-three/fiber` and `@react-three/drei`. **Rule**: All UI and visual debugging must be done in the React application. 

## 2. Dependencies and Tooling
- **Python Package Manager**: `pip` (use the local `.venv`).
- **Rust Toolchain**: `cargo` and `maturin` (for building the PyO3 bindings). Run `maturin develop` inside `.venv` to compile changes made to `core_sim/`.
- **Frontend Package Manager**: `npm` (run inside `frontend/`).

## 3. Workflow for Extending the Environment
When a user asks to add a new feature (e.g. "Add wind physics to the drones"):
1. Define the data structures for the feature in Python (`DroneEnvComponents.py`).
2. If the feature requires heavy O(N^2) math, define it in Rust (`core_sim/src/sim_core.rs`), run `maturin develop`, and call it from `DroneEnv.py`.
3. If the feature needs to be visualized, add it to the state JSON payload in `server/api.py`.
4. Update the React frontend (`frontend/src/App.jsx`) to parse the new JSON data and render the 3D meshes accordingly.

## 4. Troubleshooting Context
- **"TypeError: 'module' object is not callable"** in PettingZoo: In older versions, `agent_selector` was imported directly. In the current version, it's `from pettingzoo.utils.agent_selector import agent_selector`.
- **"ModuleNotFoundError: No module named 'core_sim'"**: The Rust library hasn't been compiled. Run `maturin develop` in `core_sim/`.
- **Gym vs Gymnasium**: This project uses `gymnasium`. Do not import `gym`. 
