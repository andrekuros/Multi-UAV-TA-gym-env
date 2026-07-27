# Paper (Overleaf / GitHub)

## Contents
- `main.tex` — IEEEtran draft: WPS hybrids (Att-RAH primary claim) plus commit/escort follow-ons
- `refs.bib` — bibliography
- `figures/` — plots from `experiments/results/figs/`

## Compile (Overleaf)
1. Upload the entire `paper/` folder to a new Overleaf project.
2. Set compiler to **pdfLaTeX**.
3. Set main document to `main.tex`.
4. Recompile (bibliography may need two passes).

## Compile (local)
```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Data sources

Generate CSVs locally (`*.csv` is gitignored). Analysis markdown is written next to them under `experiments/results/`.

| Suite | Script | Typical artifacts |
|-------|--------|-------------------|
| WPS_hard / WPS_attn | `experiments/wps_eval.py` | `wps_*_eval.csv`, `WPS_*_ANALYSIS.md` |
| WPS_commit | `wps_eval.py --suite WPS_commit` + `analyze_wps_commit.py` | `wps_commit_eval.csv`, `WPS_COMMIT_ANALYSIS.md` |
| WPS_escort | `experiments/escort_eval.py --tag att_v2_n100` | `WPS_escort_escort_eval_*.csv`, `WPS_ESCORT_ANALYSIS.md` |

## Claim boundaries

- **Supported:** Att-RAH improves \(S_{\mathrm{WPS}}\) over Local-Hungarian on **WPS_hard** (paired bootstrap CI excludes zero).
- **Negative / inconclusive for attention-specific gains:**
  - **WPS_attn** — Att-RAH does not separate from MLP or urgency ablations.
  - **WPS_commit** — Att-Commit ≈ MLP-Commit ≈ Urgency-Commit (CIs vs MLP/Urgency include zero).
  - **WPS_escort** — MLP-Coalition outperforms Att-Coalition; Att does not significantly beat CBBA.

Do not frame the paper as “Transformers beat Hungarian” or “attention wins escort.”

## Notes before submission
- Fill author / affiliation block.
- Checkpoints under `dqn_Custom/` are not committed; retrain with scripts in `experiments/`.
