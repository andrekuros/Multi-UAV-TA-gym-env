# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Locked dual thesis + systems contribution
0. **Open WPS stack** — PettingZoo env, scenario registry, visibility-masked allocator API, eval harness, optional 3D replay UI.
1. **Incomplete COP dominates.** AWACS: Local-Hungarian = Global. Sense / cueing sweeps close the Local–Global gap; oversized fleets under incomplete COP raise idle with little Local gain.
2. **Hybrid edge shaping mitigates under the mask.** Urgency / Att / MLP / GNN beat Local on WPS_attn (CIs exclude 0); CBBA/PI do not; **Att ≈ MLP ≈ GNN ≈ Urgency**.

## Figures
- `figures/fig_framework.png` — software stack schematic (`python experiments/plot_framework_fig.py`)
- `figures/fig_env_ui.png` — 3D replay dashboard screenshot (WPS_escort)
- `figures/fig_cop_sweep.png` — COP quality sweeps

## Artifacts
- `experiments/results/WPS_ATTN_FINAL.md`
- `experiments/results/WPS_ATTN_AWACS.md`
- `experiments/results/WPS_ATTN_OVERSIZED.md`
- `experiments/results/WPS_ATTN_COP_SWEEP.md`

## Reproduce camera-ready tables
```bash
# Full confirmatory + COP (long):
#   powershell -File experiments/run_final_camera_ready.ps1
python experiments/wps_eval.py --suite WPS_attn_COP_cue --episodes 100 \
  --algorithms "Local-Hungarian,Urgency-Pair,Global-Hungarian" \
  --out experiments/results/wps_attn_cop_cue.csv \
  --episodes-out experiments/results/wps_attn_cop_cue_episodes.csv
python experiments/summarize_final.py
python experiments/summarize_diagnostics.py
python experiments/summarize_cop_sweep.py
python experiments/plot_cop_sweep.py
python experiments/plot_framework_fig.py
```

## Replay UI (qualitative)
```bash
python experiments/generate_simulation_replay.py --scenario WPS_escort --seed 0
python -m uvicorn server.api:app --host 127.0.0.1 --port 8000
cd frontend && npx vite --host 127.0.0.1 --port 5173
```

## Sync
```bash
git subtree push --prefix=paper paper main
```
