# Handoff: input-bits lane (pool-qxszj1.22), 31 Oct 2026 ~20:30Z

Owner: opus-mats-inputbits (Claude subagent of main). Report: `outbox/input-bits-2026-10-31.md` (Summary for Kelvin at
the top). Theory: `FORMAL.md` v2 (v1 frozen 18:07:44Z: `coprover/FORMAL_v1_as_written.md`). Registration: `PREREG.md` +
`prereg_hashes.txt` (Parts 1-3, Amendments 2A-2E, each hashed before its runs).

## State
- Scored: Part 1 (`results/part1_scores.json`, tables `results/part1_tables.md`; 12/20 hit), Part 2 + 2A-2C
  (`results/part2_scores.json`; H2.1-H2.5 hit, H2.6 REFUTED x3), 2D (`results/a2d_scores.json`, 3/3), Part 3
  (`results/part3_scores.json`, `results/part3_tables.md`; 2/5, P3.6 PROGRESS by its rate clause, attributed to noise).
- Co-prover astra: round 1 (independent), round 2 (audit: 3 bugs fixed and reported both ways). `coprover/PROVENANCE.md`.
- GPT-6 review: three rounds (major, major, minor; `review/review_gpt6*.md`), all findings applied (report section 10).
  Coordinator's reconciliation request answered in report section 6.7 with registered Amendments 2D and 2E.
- Beads: all leaves .1-.18 closed.

## Main findings (one line each)
1. Input variance = |x| (3-9%) + amplitude + direction; direction dominates below ~2 sqrt(n) layers (half at 256x32),
   amplitude beyond; kernel + 1/n theory within 2x; amplitude -> C_inf = 10.65 (astra proof).
2. Input gate bits decay algebraically ~3n/l (kernel), entropy 2.3x; 38 bits at 256x32 = kernel 44.6 within scatter.
3. Input map: mean degree n/E[V] (19 at ARC), mean ANOVA dimension 10-15 growing with n, but ~30% additive, ~20% pairwise
   at every width.
4. Synthesis: RQMC gain predicted by 1/(1 - S1 - (1 - rho_2(N/d)) W2_off) (92% of 48 input cells within 1.5x); low-order
   shares show no detected change with width in both regimes; same noise integrand in worse Sobol dims loses its gain
   (8x -> 3x); the mean interaction order grows (confirmed, 2D/2E) inside the non-additive part; a high-order tail is a
   possible but unmeasured reconciliation.
5. Samplers: no robust rate improvement; best constant multilevel+RQMC 2-7x; only dyadic multilevel grows with width.

## Open
Pilot-limited rotations/quadratic controls (need pilots ~n^2); a validated (not plug-in) multilevel risk at larger L;
the tail's order distribution (C(t) near t = 1); per-net vs annealed statistics under heavy tails.
