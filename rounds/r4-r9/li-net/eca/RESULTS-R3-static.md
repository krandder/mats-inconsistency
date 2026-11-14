## 2. What ran

- **Why a round 3.** Kelvin did not like rounds 1-2. gemini's diagnosis: at W = 8 there are only 256 row-0 states, about 82%
  of test grids also occurred in training, so the one-pass network could have memorised attractors instead of composing the
  CA rule; and a 1-layer network (round 2) learned the rule poorly (one-step KL 0.037 / 0.077 vs 0.008 at 4 layers). Round 3 removes the overlap and searches over
  depth.
- **Data.** Noisy elementary CA, rules 110 and 90, eps = 0.015, **W = 12, T = 10** (120 cells), periodic. The 4,096 row-0
  states are split once into disjoint sets of 3,072 / 384 / 320 / 320 (train / cal / val / test); every grid draws its row 0
  from its own split's states, so no validation or test grid shares its row 0 (or its whole grid) with any training grid:
  measured overlap 0.000 for both rules (`ecadata3.overlap_report`). 50,000 / 4,000 / 2,000 / 2,000 grids. Single later rows
  still recur (99.99% of test rows t >= 1 occur somewhere in training), which no split can prevent with 4,096 row states.
- **Exact posteriors.** `ecadata3.fb_marginals`: forward-backward over the 4,096 row states, each row step being the
  deterministic CA map (scatter forward, gather backward) followed by the independent per-bit flip channel (gemini's form,
  O(W 2^W)). Checked against the dense round-1 code (W = 8, T 6/10; W = 6), brute force (W = 4, 5) and an independent
  bit-sweep implementation (W = 12, 14): max difference 9.1e-14. 0.020 s per grid at W = 12 on one CPU core (W = 14 0.065 s;
  the sweep at W = 16 1.07 s).
- **Query families (spec).** F: row 0 known, query rows 6-9. B1: rows 0 and 9 known, query rows 4-5. B2: row 0 cols 0-5,
  row 9 and columns 0-1 known, query rows 4-5, cols 3-10. S1: the query's three parents known (one-step check).
  Mean exact H(Y | E) on test, rule 110 / 90: F 0.56 / 0.52, B1 0.20 / 0.28, B2 0.22 / 0.07 nats.
- **Network and training.** The round-1/2 network (smallnet block, d 128, 4 heads) with L = 2, 3 or 4 layers, plain masked
  training (T1), early stopping on validation outcomes (patience 10 x 1,000 steps, cap 70 min), stiffness head (Mechanism A)
  and the round-2 key without h_i fitted afterwards on the cal split. 2 rules x 3 depths x 3 seeds = 18 runs.
- **Depth gate.** Per rule, the largest L whose seed-0 run passes on validation (one-step S1 KL <= 0.015 and one-pass KL >=
  0.08 for rule 110 / 0.05 for rule 90 on two of F, B1, B2). All 18 runs were launched together to meet the deadline; the
  selection reads only seed 0, as the sequential procedure would.
- **Search.** 16 chains per query, M up to 16 cells, on 500 validation and 500 test queries per family. Selectors: stiff
  (sample by exp(5 beta_int)), conf (most peaked cell), voi (VoI among the 6 stiffest), voiu (VoI among the 6 most uncertain),
  hyb (P6-Hybrid: materialise the lowest-entropy cell while some cell has H < 0.20 nats, else branch with voiu), rand.
  Pools: sw_mix (anchored at the one-pass belief, primary), plain mixture, Gram dedup with the h_i-free keys.
- **Epistemic diagnostic (R3-D3).** On 2,000 held-out (cell, evidence) pairs: how well the stiffness head, plain confidence
  ln 2 - H(q), and 1-step self-consistency rank pairs by the network's own error KL(P* || q).
- **Compute.** GCP on-demand L4s, one priced smoke VM and 9 main VMs (2 runs each), budget group overnight-ca, round-3 cap $40.
