# Handoff: stationary-numerics lane (pool-qxszj1.21, closed), 31 Oct 2026 17:40Z

Owner: opus-mats-numerics (Claude subagent of main). Report: `outbox/stationary-numerics-2026-10-31.md`, revised after
three GPT-6 astra review rounds (13, 9 and 5 findings, verdict "major revision" each time; every finding applied,
report section 9; reviews in `review/review_gpt6*.md`). A fourth round was not run.
Theory notes: `notes/proofs.md`. Registration: `PREREG.md` + `prereg_hashes.txt` (Parts R, G, I, S, D; Amendments
S1, S2, G1, G2; one bug fix in response_net.py, registered version kept).

## State
- Scored: R (`results/partR_scores.json`), G (`results/partG_scores.json`, `results/partG7_scores.json` incl. G8),
  I (`results/partI_scores.json`, `results/fits.json`), S (`results/partS_scores.json`, `results/partS4_exact.json`),
  D (`results/partD_scores.json`); post hoc: `results/posthoc_review_fixes.json`, `results/posthoc_review2.json`,
  `results/tangent_fits.json`, `results/vertex_fixed_point.json`. Scorecard 17 of 35 hit (report section 8).
- Nothing running. Last check: n = 3 boundary densities at grid step 0.14 match step 0.2 to three digits
  (`runs/BD_n3_s1000_dx0.14.json`).
- Figures in `figs/`. Python: /home/agent/.venvs/arc-noisy/bin/python. CPU: nice 19, ionice -c3, <= 3 processes.
- Beads: all leaves .1-.16 and the lane bead .21 closed (review-reject hiccup recorded on .12). The topic bead
  pool-qxszj1 (Kelvin's ask) now has all leaves closed; closing it is main's call.

## Main findings
1. Radius: exact at every n; closed-form 1/n, 1/n^2 corrections; left-tail rate K(1) = 0.3107 (not 0.4).
2. PDLT recursions with noise: no one-input fixed point; the replica Gram chain is exact and O(n)/layer.
3. Replica angle at the noisy-kernel fixed point delta* ~ 2.23 Q^-2/3 (exact chain within 0-12%, converging).
4. The 5-8x: 2.1 Q delta*/(H_n - 1) with an assumed 1/k shape; 21 of 22 nets within 0.86-1.42 (G7, G8 miss on clauses).
5. Interaction order grows: h_0 1.41 -> 3.87 (n = 8 -> 256), slope 0.26; L0 within 8-15% of networks (direction).
6. Interventions: gates within ~4 reference-noise SDs in the last n layers reproduce the excess; frozen gates give 1.
7. Tangent response along finite-window CLVs ~ rank-conditioned exponents (median ratio 1.04); finite perturbations keep
   a slow plateau ~5.7 decades above the tangent median at t = n.
8. One noise-decided flip: ~0.2n gate differences; 100x-decay lifetime median 1.4-1.9n; exact refresh cascade ~ log n.
9. n = 2-4 exact densities: gate-pattern mixtures; non-Gaussianity tracks the radius; Gaussian misstates boundary
   densities by up to 38%, the mixture by up to 13%.

## Open
1/k angular spectrum (not derived); radial variance at the fixed point (closures 1.1-3.7x exact); per-net vs annealed
statistics; whether the response plateau is the radial mode; n = 6 grids (infeasible with dense grids).
