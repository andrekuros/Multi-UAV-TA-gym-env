# RL Improvement Experiment Plan

**Goal:** Beat Swarm-GAP / CBBA on mean `F_Reward` on the UCF eval suite (`scal_None`, `scal_Agents`, `scal_Tasks`), staying on **TBTA / shared DQN**.

**Baseline (committed CSVs):** Swarm-GAP ≈ 500, CBBA ≈ 497, TBTA ≈ 423 on `scal_None`; TBTA collapses further on agent/task scaling.

---

## Diagnosis (why TBTA loses)

1. **Train–eval mismatch:** training used `4×F2 + 25×Att`; UCF agent scaling uses `F1+R1` with `Att=6, Rec=24`.
2. **No early termination:** episodes always run to 150 steps after tasks finish → dilute returns.
3. **Weak action mask:** padded-only; agents can pick zero-cap / already-saturated tasks.
4. **Disabled time/alloc shaping:** only distance + quality + S_quality active.
5. **Hyperparams:** ε starts at 0.7 and decays slowly; n-step = 150 (= horizon); prioritized replay off.

Env toggles (defaults **off** = legacy behavior):

| Flag | Effect |
|------|--------|
| `early_terminate` | End when all tasks `status==2` |
| `capability_mask` | Mask tasks with zero agent capability |
| `saturate_mask` | Mask tasks already fully allocated |
| `reward_weights` | Scale individual reward terms |

---

## Success criteria

| Suite | Primary metric | Pass |
|-------|----------------|------|
| `scal_None` | mean `F_Reward` (10 eps) | TBTA ≥ Swarm-GAP − 2% |
| `scal_Agents` / `scal_Tasks` | mean `F_Reward` vs curve | TBTA ≥ CBBA mean across cases |

Always report vs Random / Greedy / Swarm-GAP / CBBA with the **same seeds**.

---

## Experiment matrix

Run in order. Each experiment = train + eval. Change **one** factor from previous winner unless noted.

### E0 — Reproducible baseline (no env changes)

- Config: legacy flags off, scenario `F2:4, Att:25` (match old notebook).
- Eval: `main.py` UCF suites with current checkpoint naming.
- **Purpose:** confirm harness + numbers before claiming gains.

### E1 — Match train distribution to UCF

Train on a **mixture** each episode (uniform pick):

| Mix | Agents | Tasks |
|-----|--------|-------|
| A | F2:2 | Att:15 |
| B | F1:⌊i/2⌋, R1:i (i∼U{2..8}) | Att:6, Rec:24 |
| C | F1:2, F2:2, R1:3, R2:3 | Att:2+⌊k/5⌋, Rec:k (k∼U{4..20}) |

Hyperparams: same as E0.  
**Hypothesis:** largest single gain on `scal_Agents` / `scal_Tasks`.

### E2 — Early terminate

- On E1 winner: `early_terminate=True`.
- **Hypothesis:** denser effective horizon → higher `F_Reward`, lower wasted steps.

### E3 — Action masking

- On E2: `capability_mask=True`, `saturate_mask=True`.
- **Hypothesis:** fewer wasteful allocations (−1.5 S_quality hits).

### E4 — Reward ablation (one weight change each)

Base weights: `distance=1, quality=1, s_quality=1`, others `0`.

| Run | Change |
|-----|--------|
| E4a | `time_penaulty=0.25` |
| E4b | `alloc=0.1` |
| E4c | `time=0.1` + `time_penaulty=0.25` |
| E4d | `distance=0.5` (less path pressure) |

Keep best of E4*.

### E5 — DQN hyperparams (on best env of E1–E4)

| Run | Change |
|-----|--------|
| E5a | eps: 0.5 to 0.05 over first 40 epochs |
| E5b | `estimation_step=8` (was 150) |
| E5c | lr `1e-4`, target update `3000` |
| E5d | Prioritized replay on |

### E6 — Light curriculum (optional if still behind Swarm-GAP)

1. Epochs 1–30: Mix A only  
2. Epochs 31–60: Mix A+B  
3. Epochs 61+: Mix A+B+C  

---

## How to run

```bash
# Train an experiment config
python experiments/train_tbta.py --exp E1

# Eval against baselines (set algorithm list / case in main.py, or:)
python experiments/train_tbta.py --exp E1 --eval-only --policy dqn_Custom/<checkpoint>.pth
```

Checkpoints: `dqn_Custom/policy_TBTA_<exp>_<tag>.pth`  
Logs: `Logs/dqn/<exp>/`

---

## Logging checklist (every run)

- exp id, git hash, seed  
- env flags + reward_weights  
- train mix + max_epoch  
- mean/std `F_Reward`, `S_Reward`, `F_time`, `F_distance`  
- wall-clock train time  

---

## Out of scope (this cycle)

- MAPPO / CTDE redesign  
- Full PPO rebuild (eval often returns 0 today)  
- ILP upper bound  

Revisit only if E1–E6 still leave TBTA >5% behind Swarm-GAP on `scal_None`.
