# Paper (Overleaf / GitHub) — IEEE TAES track

This folder is the manuscript for hybrid cost shaping under WPS / incomplete COP.
Monorepo: [`Multi-UAV-TA-gym-env`](https://github.com/andrekuros/Multi-UAV-TA-gym-env).  
Standalone mirror: [`Multi-UAV-TA-paper`](https://github.com/andrekuros/Multi-UAV-TA-paper) via `git subtree`.

## Target venue
**IEEE Transactions on Aerospace and Electronic Systems (Regular Paper)**  
Framing: aerospace multi-UAV mission management under delayed sensing and perishable windows—not attention novelty.

## Locked claim
On **WPS_hard** (\(N{=}100\)), Att-RAH mean \(\Delta S_{\mathrm{WPS}}\approx +8.7\) vs Local-Hungarian; paired bootstrap 95\% CI **includes zero** (\([-8.1,+26.0]\)).  
Local-Hungarian remains a strong fair baseline under the evaluated budget.  
**Not claimed:** significant Att-RAH win over Local-Hungarian; set-attention superiority over matched MLP / urgency.

## TAES readiness checklist
- [x] Phase 0 — Aerospace / C2 framing
- [x] Phase 1 — Expanded verified related work
- [x] Phase 2 — System model + suite table
- [x] Phase 3 — Repro appendix; commit/escort moved to appendix
- [x] Phase 4 — Sensitivity appendix + primary table updated to TAES retrain/eval
- [x] Phase 5–7 — Author/ORCID placeholders, cover letter aligned, claim audit, page estimate ~7–9

## Page estimate
Body ≈ 2.5k words + 5 figures + 6 tables under IEEEtran two-column → roughly **7–9 pages** before final author bios; target ≤10 for Regular Paper.

## Sync
```bash
git subtree push --prefix=paper paper main
git subtree pull --prefix=paper paper main --squash
```
