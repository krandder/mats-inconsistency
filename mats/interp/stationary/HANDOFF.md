# Handoff: stationary-regime lane (pool-qxszj1), 31 Oct 2026 14:20Z

Owner so far: opus-mats-stationary (Claude, subagent of main). Report: `outbox/stationary-regime-2026-10-31.md`
(final; Summary for Kelvin at the top). Theory: `FORMAL.md` v2.2. Every measured number: `notes/RESULTS_INDEX.md`.

## State
- Done and scored: Part A (PA1-PA12, PR1-PR2), Amendments A2 (bit cascade), A3 (finite-angle law, fresh nets: hit),
  Part B (estimators; HB holds: no width-growing gain), Part C (input regime, ARC 256x32), Part I (interaction
  structure, Kelvin's follow-up). Scorers: `scratch/analyze_{A,B,C,I}.py` -> `results/part{A,B,C,I}_scores.json`.
- Co-prover: astra rounds 1-2 done (`coprover/astra_r{1,2}_answer.md`, `coprover/PROVENANCE.md`).
- GPT-6 review done (14:25Z, major revision, 14 findings; `review/review_gpt6.md`) and applied (report section 6,
  FORMAL v2.2). Amendments B1 (antithetic to n = 1024) and B2 (RQMC to N = 2^16) run and scored
  (`results/partB12_scores.json`). Bug fixes: CV beta guard (fix 1), radial-tilt weight (fix 2), both reported.
- Beads: my leaves .1-.20 closed (.17 report closed last). .21 belongs to the sibling numerics lane.

## Main findings (one line each)
1. Model S stationary measure exists/unique, random point attractor with noise fixed (PROVEN [C+A]).
2. sigma is a pure scale; noise-to-state ratio ~ n^-1/2 set by width (PROVEN).
3. Radius = exact Kesten chain; chain median Q/n 0.575-0.579 at all n (limit 0.5771 proven); tail C_n/x, C_n = 2n/5 + 13/75 + O(1/n).
4. Cloud: thin, near-Gaussian at large n; size 1.2-8x the Lyapunov formula; finite-angle law delta = 2.23 X^-2/3 and
   radial CV ~ 1.36 n^-1/3 passed a fresh test (A3).
5. Births 0.5-0.6 sqrt(n)/layer (exact formula); one injection flips 1.6-4.5 n gates over 2n layers.
6. Interactions: mean order 2-3 layers, angular coordinates carry >= 96%, gate switching is the main mechanism (Kesten
   log Q order 1.2; radius-clamped net keeps most), no clear width growth for h/a; output gates' order grows 5 -> 15.
7. Estimators: all exponent 1 (RQMC fitted 1.0-1.2); gains 2-8x shrinking with n; antithetic 1.9 -> 2.4x free; K2
   closure floor ~0.12/n.
8. Input regime: forgotten by T_mem; additive budget (interaction <= 0.26); ARC 256x32 sigma=1e-3 input-dominated
   (noise < 0.1% of variance), 38 input gate bits left at layer 31.

## Open questions worth a next lane (not started)
- An estimator using the finite-angle law (cloud size from the radius) that avoids the gate interactions.
- Whether the 25% even (nonlinear-in-noise) variance fraction vanishes at larger width (it fell only 0.27 -> 0.21
  from n = 8 to 256).
- Third-order Sobol indices; astra's exact 1-D final-noise integration for tails (untested).
- Sibling lane stationary-numerics (coordinator, 14:1xZ) covers transfer-operator radius, finite-width effective
  theory, n = 2-4 grids and data collapse; it reads this directory.
