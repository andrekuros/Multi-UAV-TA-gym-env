# Multi-UAV-TA 3D Dashboard

React + Vite + Three Fiber frontend for replaying deterministic WPS simulation logs streamed by the FastAPI backend.

## Prerequisites

- Node.js 18+
- Python `.venv` with the repo installed and `core_sim` built
- A replay JSON under `experiments/results/` (API prefers `wps_escort_replay.json`, then `wps_commit_replay.json`)

## Generate a replay

From the repo root:

```bash
python experiments/generate_simulation_replay.py --scenario WPS_escort --seed 0
# or: --scenario WPS_commit
```

## Run

**Terminal 1 — API** (`http://localhost:8000`):

```bash
python server/api.py
```

**Terminal 2 — Vite:**

```bash
cd frontend
npm install   # first time
npm run dev
```

Open the URL printed by Vite. Controls: play/pause, scrub, speed, event timeline, sensing/assignment overlays, download JSON.

## Project layout

- `src/App.jsx` — scene, HUD, WebSocket / REST client
- Backend payload schema is defined in `server/api.py`

This is not a generic Vite template; keep UI changes aligned with the Multi-UAV-TA replay schema.

## Related docs

- Repo overview / train-eval: [`../README.md`](../README.md)
- Architecture rules: [`../AI_DEVELOPMENT_GUIDE.md`](../AI_DEVELOPMENT_GUIDE.md)
