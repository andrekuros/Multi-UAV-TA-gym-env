# Multi-UAV-TA Framework (v02.0)

Open-source PettingZoo environment and hybrid task-allocation stack for heterogeneous multi-UAV Target Assignment under delayed sensing, hard time windows, and coalition escort dynamics.

## Architecture

```
mUAV_TA/          PettingZoo ParallelEnv (headless; S_WPS / S_ESC metrics)
core_sim/         Rust (PyO3) physics: LoS, distances, obstacle avoidance
TaskAllocation/   Classical + hybrid allocators
experiments/      Scenario registry, train/eval, figures, replay JSON
server/ + frontend/   FastAPI WebSocket + React/Three Fiber 3D dashboard
```

1. **Environment** (`mUAV_TA/DroneEnv.py`) — multi-agent state, steps, rewards, WPS/escort metrics.
2. **Rust core** (`core_sim/`) — compiled geometry; called synchronously from Python.
3. **Allocators** (`TaskAllocation/`) — Hungarian, CBBA, PI, Cap-Greedy, and hybrids (Att-RAH, Att-Commit, Att-Coalition).
4. **Experiments** (`experiments/`) — canonical scenarios in `paper_scenarios.py`; train/eval scripts.
5. **Visualization** (`server/`, `frontend/`) — decoupled replay dashboard (does not block training).

Legacy entry points (`main.py`, `Training/`, `Evaluations/`) remain for older TBTA/UCF workflows; prefer `experiments/` for new work.

## Installation

### 1. Python environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install gymnasium pettingzoo tianshou torch numpy fastapi uvicorn scipy
```

### 2. Rust core
```bash
pip install maturin
cd core_sim && maturin develop --release && cd ..
```

### 3. Frontend (optional, for 3D viz)
```bash
cd frontend && npm install && cd ..
```

## Scenarios

Canonical registry: [`experiments/paper_scenarios.py`](experiments/paper_scenarios.py).

| Suite | Purpose |
|-------|---------|
| Static / D1–D3 | Legacy UCF-style strike, attrition, pop-up threats |
| `WPS_easy` / `WPS_hard` / `WPS_burst` | Windowed Pop-up Strike (primary paper claim on `WPS_hard`) |
| `WPS_attn` | Multi-front attention stress (no knowledge sharing) |
| `WPS_commit` | Dual-front + timed commits + rematch penalty |
| `WPS_escort` | Coalition escort: protect Rec UAVs; multi-fighter slots |

Env presets: `WPS_ENV_FLAGS` (full horizon, time windows on).

**Primary metrics:** `S_WPS` (on-time / miss / distance); escort uses `S_ESC` = `S_WPS` + protected-rec bonus − recon-loss penalty + coverage term.

## Hybrid methods

All hybrids keep a classical assignment engine; learning only reshapes inputs.

| Method | Module | Interface |
|--------|--------|-----------|
| Att-RAH / MLP-RAH | `AttentionRAH.py`, `ReserveAwareHybrid.py` | Task priorities + soft reserve → Local-Hungarian |
| Att-Commit / MLP-Commit / Urgency-Commit | `AttentionCommit.py` | Priorities + per-agent commit locks → Hungarian on free agents |
| Att-Coalition / MLP-Coalition / Urgency-Coalition | `AttentionEscort.py` (v2) | Local tokens → cross-attention pair logits → coalition Hungarian |

Import Attention\* classes from their modules (not `Hybrid/__init__.py`).

**Classical baselines:** Local/Global Hungarian, Cap-Greedy, CBBA-Replan; for escort also Coalition-Hungarian, Local-CBBA-Coalition, Local-PI-Coalition.

## Train / eval quick-start

Checkpoints write to `dqn_Custom/` (gitignored). Result CSVs under `experiments/results/` are generated locally (`*.csv` gitignored).

```bash
# Att-RAH on WPS_hard
python experiments/train_att_rah.py --episodes 400 --case WPS_hard
python experiments/wps_eval.py --suite WPS_hard --episodes 100

# Att-Commit on WPS_commit
python experiments/train_att_commit.py --episodes 280 --case WPS_commit
python experiments/wps_eval.py --suite WPS_commit --episodes 100 \
  --algorithms "Local-Hungarian,Urgency-Commit,MLP-Commit,Att-Commit,Global-Hungarian"

# Att-Coalition v2 on WPS_escort
python experiments/train_escort.py --episodes 400 --case WPS_escort
python experiments/train_escort.py --mlp --episodes 400 --case WPS_escort
python experiments/escort_eval.py --episodes 100 --tag att_v2_n100
```

Optional Att-Coalition hyperparam search: `python experiments/search_att_escort.py`.

Smoke tests: `python experiments/test_escort.py`.

## 3D visualization

The API prefers `experiments/results/wps_escort_replay.json`, then `wps_commit_replay.json`.

```bash
python experiments/generate_simulation_replay.py --scenario WPS_escort --seed 0
# or: --scenario WPS_commit
```

**Terminal 1 — API** (`http://localhost:8000`):
```bash
python server/api.py
```

**Terminal 2 — dashboard:**
```bash
cd frontend && npm run dev
```

Play/pause, scrubbing, event timeline, sensing/assignment overlays, JSON download.

## Paper

IEEE-style draft: [`paper/main.tex`](paper/main.tex). See [`paper/README.md`](paper/README.md) for compile steps and claim boundaries.

Standalone (Overleaf-friendly) mirror: [andrekuros/Multi-UAV-TA-paper](https://github.com/andrekuros/Multi-UAV-TA-paper) — synced from `paper/` via `git subtree` (Cursor keeps this full monorepo):

```bash
git subtree push --prefix=paper paper main   # after committing paper changes here
git subtree pull --prefix=paper paper main --squash
```

## AI development

See [`AI_DEVELOPMENT_GUIDE.md`](AI_DEVELOPMENT_GUIDE.md) for architectural rules when extending the env, Rust core, or hybrids.

## Legacy TBTA track

[`RL_EXPERIMENT_PLAN.md`](RL_EXPERIMENT_PLAN.md) documents the older shared-DQN / `F_Reward` / UCF scaling experiments. Prefer the WPS hybrid pipeline above for publication work.
