# Paper (Overleaf / GitHub) — IEEE TAES track

Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper).

## Locked claim (ContextPair / WPS_attn)
On **WPS_attn** (\(N{=}100\)), Urgency-Pair / MLP-ContextPair / Att-ContextPair improve \(S_{\mathrm{WPS}}\) over Local-Hungarian (CIs exclude 0).  
**MLP-ContextPair** is best among learned methods; **Att does not beat MLP** (Att−MLP CI excludes 0 against Att).  
On **WPS_hard**, simpler Att-Pair / MLP-Pair CIs vs Local **include** 0.

## Artifacts
- `experiments/results/wps_attn_context_gate.csv` / `WPS_ATTN_CONTEXT_GATE.md`
- Checkpoints: `dqn_Custom/policy_{Att,MLP}ContextPair_WPS_attn_il.pth`

## Sync
```bash
git subtree push --prefix=paper paper main
```
