# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Locked claim (ContextPair / WPS_attn)
On **WPS_attn** (N=100), Urgency-Pair / MLP-ContextPair / Att-ContextPair improve S_WPS over Local-Hungarian (CIs exclude 0).  
Across **3 training seeds × 2 input feature sets**, the paired **Att−MLP** contrast includes 0 in every cell, and neither learned encoder separates from **Urgency-Pair**. The gain is edge shaping, not attention.  
On **WPS_hard**, simpler Att-Pair / MLP-Pair CIs vs Local **include** 0.

### Retracted
An earlier claim that MLP significantly beats Att (Att−MLP ≈ −20, CI excluding 0) came from a single-state imitation update. Switching to mini-batched IL with warmup — optimizer only, everything else fixed — removed the effect. See `sec:optconfound` in `main.tex`.

## Artifacts
- `experiments/results/WPS_ATTN_ABLATION.md` (3-seed raw-vs-engineered ablation)
- `experiments/results/wps_attn_ablation_{raw,eng}[_s1,_s2].csv`
- `experiments/results/wps_attn_context_gate.csv` / `WPS_ATTN_CONTEXT_GATE.md`
- Checkpoints: `dqn_Custom/policy_{Att,MLP}ContextPair{,Raw}{,B}[S1,S2]_WPS_attn_il.pth`

## Reproduce the ablation
```bash
python experiments/train_pair_cost.py --context --phase il --episodes 900 --case WPS_attn \
  --seed 0 --il-batch 8 --il-warmup 50 [--raw] [--mlp] [--suffix B]
python experiments/wps_eval.py --suite WPS_attn --episodes 100 --raw \
  --algorithms "Local-Hungarian,Urgency-Pair,MLP-ContextPair,Att-ContextPair"
```

## Sync
```bash
git subtree push --prefix=paper paper main
```
