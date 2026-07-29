# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Locked claim (WPS_attn final)
Edge shaping (Urgency / Att / MLP / GNN ContextPair) beats **Local-Hungarian** (CIs exclude 0).  
**Local-CBBA** and **Local-PI** do **not** beat Local under the same `agent_known_ids` mask.  
**Att ≈ MLP ≈ GNN ≈ Urgency** (all pairwise CIs include 0). Gain = hybrid cost shaping, not architecture.  
Scale transfer and raw-feature ablations remain honest negatives for Att>MLP.

## Artifacts
- `experiments/results/WPS_ATTN_FINAL.md`
- `experiments/results/wps_attn_final_s{0,1,2}.csv`
- GNN: `dqn_Custom/policy_GNNContextPairRawGnn[S1,S2]_WPS_attn.pth`

## Reproduce final table
```bash
python experiments/train_pair_cost.py --context --gnn --raw --phase il --episodes 900 \
  --case WPS_attn --seed 0 --il-batch 8 --il-warmup 50 --suffix Gnn
python experiments/wps_eval.py --suite WPS_attn --episodes 100 \
  --algorithms "Local-Hungarian,Local-CBBA-Replan,Local-PI,Urgency-Pair,MLP-ContextPair,Att-ContextPair,GNN-ContextPair,Global-Hungarian" \
  --att-ctx dqn_Custom/policy_AttContextPairRaw_WPS_attn.pth \
  --mlp-ctx dqn_Custom/policy_MLPContextPairRaw_WPS_attn.pth \
  --gnn-ctx dqn_Custom/policy_GNNContextPairRawGnn_WPS_attn.pth
python experiments/summarize_final.py
```

## Sync
```bash
git subtree push --prefix=paper paper main
```
