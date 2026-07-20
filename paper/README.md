# Paper (Overleaf / GitHub)

## Contents
- `main.tex` — IEEEtran journal draft aligned with WPS + Att-RAH experiments
- `refs.bib` — bibliography
- `figures/` — plots copied from `experiments/results/figs/`

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

## Data source
- Main tables: `experiments/results/wps_final_eval.csv` + `WPS_FINAL_ANALYSIS.md` (WPS_hard claim).
- Attention stress (negative architecture result): `experiments/results/wps_attn_eval.csv` + `WPS_ATTN_ANALYSIS.md`.

## Notes before submission
- Fill author / affiliation block.
- Lead claim is hybrid cost shaping on WPS_hard, not “Transformers beat Hungarian.”
- WPS_attn is an honest negative result for set-attention under the priority→Hungarian interface.
