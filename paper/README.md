# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Locked dual thesis
1. **Incomplete COP dominates.** AWACS: Local-Hungarian = Global. Sense / cueing sweeps close the Local–Global gap; oversized fleets under incomplete COP raise idle with little Local gain.
2. **Hybrid edge shaping mitigates under the mask.** Urgency / Att / MLP / GNN beat Local on WPS_attn (CIs exclude 0); CBBA/PI do not; **Att ≈ MLP ≈ GNN ≈ Urgency**.

## Artifacts
- `experiments/results/WPS_ATTN_FINAL.md`
- `experiments/results/WPS_ATTN_AWACS.md`
- `experiments/results/WPS_ATTN_OVERSIZED.md`
- `experiments/results/WPS_ATTN_COP_SWEEP.md`
- `paper/figures/fig_cop_sweep.png`

## Reproduce camera-ready tables
```bash
# Full confirmatory + COP (long): 
#   powershell -File experiments/run_final_camera_ready.ps1
# Then cueing panel (share=True, R=0):
python experiments/wps_eval.py --suite WPS_attn_COP_cue --episodes 100 \
  --algorithms "Local-Hungarian,Urgency-Pair,Global-Hungarian" \
  --out experiments/results/wps_attn_cop_cue.csv \
  --episodes-out experiments/results/wps_attn_cop_cue_episodes.csv
python experiments/summarize_final.py
python experiments/summarize_diagnostics.py
python experiments/summarize_cop_sweep.py
python experiments/plot_cop_sweep.py
```

## Sync
```bash
git subtree push --prefix=paper paper main
```
