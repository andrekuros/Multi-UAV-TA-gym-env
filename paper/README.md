# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Double-blind vs camera-ready URLs

- **Double-blind submission:** `main.tex` must **not** contain live GitHub URLs. Use the anonymized release sentence; keep `% CAMERA-READY: https://github.com/andrekuros/Multi-UAV-TA-gym-env` comments next to each site. Bib `@misc{multi_uav_ta_gym}` note stays “URL withheld for double-blind review.”
- **Camera-ready:** uncomment / restore the live URL in abstract, `\thanks`, contribution 1, Sec. env, and appendix; restore the bib URL note.

## Locked dual thesis + enabling stack
0. **Open WPS stack** — enabling artifact (PettingZoo env, scenario registry, visibility-masked allocator API, eval harness, optional 3D replay UI); not a co-equal scientific claim with (1)–(2).
1. **Incomplete COP dominates.** AWACS: Local-Hungarian = Global. Sense / cueing sweeps close the Local–Global gap; oversized fleets under incomplete COP raise idle with little Local gain.
2. **Hybrid edge shaping mitigates under the mask.** Urgency / Att / MLP / GNN beat Local on WPS_attn (CIs exclude 0); CBBA/PI do not; **Att ≈ MLP ≈ GNN ≈ Urgency**.

## Figures
- `figures/fig_framework.png` — primary systems schematic (`python experiments/plot_framework_fig.py`)
- `figures/fig_env_ui.png` — mid-episode 3D replay dashboard (qualitative only; not used for metrics)
- `figures/fig_cop_sweep.png` — COP quality sweeps

## Artifacts
- `experiments/results/WPS_ATTN_FINAL.md` — primary suite means (do not overwrite with SCALE Local)
- `experiments/results/WPS_ATTN_AWACS.md`
- `experiments/results/WPS_ATTN_OVERSIZED.md`
- `experiments/results/WPS_ATTN_COP_SWEEP.md`
- `experiments/results/WPS_ATTN_SENS.md` / `WPS_ATTN_SENS_MLP.md` — primary-suite metric-weight sensitivity (Urgency / MLP vs Local)
- `experiments/results/WPS_ATTN_SCALE.md` — zero-shot L/XL pad-transfer (separate protocol; Att−MLP CIs include 0)

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
python experiments/wps_metric_sensitivity.py
python experiments/check_paper_tex.py
```

## Replay UI (qualitative)
```bash
python experiments/generate_simulation_replay.py --scenario WPS_escort --seed 0
python -m uvicorn server.api:app --host 127.0.0.1 --port 8000
cd frontend && npx vite --host 127.0.0.1 --port 5173
```
Seek ~35–50% of the episode for `fig_env_ui.png` captures.

## Sync
```bash
git subtree push --prefix=paper paper main
```
