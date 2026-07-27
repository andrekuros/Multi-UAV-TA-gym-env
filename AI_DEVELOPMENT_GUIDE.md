# Multi-UAV-TA: AI Development Guide

If you are an AI assistant helping develop this repository, read this document before changing architecture.

## 1. Project architecture

Four decoupled domains:

1. **Core RL environment (Python)** — `mUAV_TA/`. PettingZoo Parallel API (`DroneEnv.py`).  
   **Rule:** Never add rendering loops, `pygame`, or UI blocking code. Keep the env headless for training.
2. **Physics (Rust)** — `core_sim/`. LoS, distances, obstacle avoidance via PyO3.  
   **Rule:** If you find Python `for` loops over many agents doing geometry, port to `core_sim` and call from `DroneEnv.py`.
3. **Task allocation** — `TaskAllocation/`. Classical solvers (Hungarian, CBBA, PI, Cap-Greedy) plus hybrids.  
   **Rule:** Hybrids reshape costs, priorities, commit locks, or pair scores; they do not replace the combinatorial engine.
4. **Visualization** — `server/api.py` + `frontend/`. FastAPI WebSocket + React Three Fiber.  
   **Rule:** All visual debugging belongs here, not in the env.

## 2. Scenarios and flags

**Single source of truth:** `experiments/paper_scenarios.py` (`CASE_SPECS`).

| Keys | Role |
|------|------|
| Static / D1–D3 | Legacy strike / attrition / threats |
| `WPS_easy`, `WPS_hard`, `WPS_burst` | Windowed Pop-up Strike |
| `WPS_attn` | Multi-front attention stress |
| `WPS_commit` | Dual-front + `commit_horizon` + `reassign_penalty` |
| `WPS_escort` | Escort coalitions (`escort_enabled`, radii, fighter types) |

Use `WPS_ENV_FLAGS` for WPS evals (full horizon, time windows on). Do not duplicate case definitions in new scripts—import `CASE_SPECS`.

## 3. Metrics

- **`S_WPS`** — primary perishable score: on-time bonus − miss penalty − scaled distance (see `DroneEnv.compute_s_wps` / paper eq.).
- **`S_ESC`** — escort score in `DroneEnv.compute_s_esc`:
  - `S_WPS + 20 * protected_rec_completed - 30 * recon_losses + 20 * escort_coverage_rate`
- Legacy **`F_Reward`** — TBTA / UCF secondary only.

## 4. Hybrid modules

Import Attention classes from their files ( `Hybrid/__init__.py` does not export them):

| Class | Path | Decision interface |
|-------|------|--------------------|
| `AttentionRAH` | `TaskAllocation/Hybrid/AttentionRAH.py` | Priorities + reserve ρ → Local-Hungarian |
| `AttentionCommit` / `UrgencyCommit` | `AttentionCommit.py` | Priorities + timed commits → Hungarian on free agents |
| `AttentionEscort` / `UrgencyCoalition` | `AttentionEscort.py` | Pair logits → coalition Hungarian (v2 actor-critic) |

**Att-Coalition v2 notes:**

- Token window defaults: `max_tasks=48`, `max_agents=16`; local known-task union; urgency/pressure sort before truncation.
- Checkpoints include `version: 2` and arch hyperparams; v1 `.pth` files are incompatible.
- Train: `experiments/train_escort.py`; eval: `experiments/escort_eval.py`; smoke: `experiments/test_escort.py`.

**Claim discipline:** Do not claim “attention beats X” unless paired bootstrap CIs exclude zero on the stated suite and metric. WPS_hard supports Att-RAH vs Local-Hungarian; WPS_commit and WPS_escort did not support attention-specific gains vs MLP/CBBA.

## 5. Extending the environment

When adding a feature (e.g. wind):

1. Data structures in `DroneEnvComponents.py` / config in `MultiDroneEnvUtils.py`.
2. Heavy O(N²) math → Rust + `maturin develop`.
3. Viz fields → JSON payload in `server/api.py` and React meshes in `frontend/`.
4. If the feature is scenario-specific, add knobs to `paper_scenarios.py` and document metrics.

## 6. Dependencies and tooling

- Python: local `.venv` + `pip`.
- Rust: `cargo` + `maturin` inside `core_sim/`.
- Frontend: `npm` inside `frontend/`.

## 7. Troubleshooting

- **PettingZoo `agent_selector`:** use `from pettingzoo.utils.agent_selector import agent_selector` (or `AgentSelector` where required).
- **`ModuleNotFoundError: core_sim`:** run `maturin develop` in `core_sim/`.
- **Gym vs Gymnasium:** use `gymnasium`; do not import `gym`.
- **Escort checkpoint load errors:** retrain with `train_escort.py` (v2 nets).
