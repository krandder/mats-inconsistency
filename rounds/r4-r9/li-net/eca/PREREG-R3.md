# E-CA round 3 pre-registration (eurisko, 2026-10-15, written 09:46Z, before any round-3 launch command)

Ask: Kelvin 09:2xZ via opus-mats, "Please run new experiments with gemini; we didn't like the previous results."
Spec: gemini, `/home/agent/workspaces/mats/outbox/overnight-followups-round-3.md`, Rank 2 (E-CA-R3); opus-mats' notes, mail #3649.
Cap: $40 for round 3 (group overnight-ca; the guard stops the group at $105 overall; night spend so far $19.90). Run ids
`eca3-*`. Rounds 1-2 code paths are unchanged; round 3 is selected by `--data-module ecadata3` and flags below.
Honesty rules as before: outcomes only in training and selection; nothing hand-coded about the domain in any arm; every rule
reported pass/fail with numbers; no retuning after test. Any change after this file is written goes into
`PREREG-R3-AMENDMENTS.md` with its time, never into this file.

## 1. Data (spec "Exact Changes" 1)
- Noisy elementary CA, rules 110 and 90, eps = 0.015, **W = 12, T = 10** (N = 120 cells), periodic, row 0 iid uniform.
- All 4,096 row-0 states are permuted once (seed 30015010 + W) and split **3072 / 384 / 320 / 320** into train / cal / val /
  test; each split draws its row 0 uniformly from its own states (`ecadata3.row0_buckets`). Grids: train 50,000, cal 4,000,
  val 2,000, test 2,000. Overlap check (`ecadata3.overlap_report`, run 09:45Z): cal, val and test row-0 states and whole
  grids found in train: 0.000 for both rules. (Single later rows still recur: 99.99% of test rows t >= 1 occur somewhere in
  the training rows; unavoidable with 4,096 row states.)
- Fixed permutation of the 120 cell ids (`ecadata.make_perm(10, 12)`), as in rounds 1-2.
- Query families (spec): **F** row 0 revealed, query rows 6-9; **B1** rows 0 and 9 revealed, query rows 4-5; **B2** row 0
  cols 0-5, row 9 and columns 0-1 revealed, query rows 4-5, cols 3-10; **S1** the query's 3 past parents revealed (plus a
  random reveal set); **R** random (kept from rounds 1-2, not used by any R3 rule). Mean exact H(Y | E) on test (nats),
  rule 110 / 90: F 0.560 / 0.523, B1 0.203 / 0.280, B2 0.221 / 0.067, S1 0.032 / 0.028.
- Exact posteriors (scoring, the D0 gate and the R3-D3 diagnostic only): `ecadata3.fb_marginals`, forward-backward over the
  4,096 row states where each row step is the deterministic CA map (scatter forward, gather backward) followed by the
  independent per-bit flip channel (gemini's form, O(W 2^W)). Checked: max |diff| vs the dense round-1 code (W = 8, T 6/10,
  W = 6) <= 1.6e-15, vs brute force (W = 4, 5) <= 9.1e-14, vs an independent bit-sweep implementation at W = 12, 14
  <= 1.3e-15 (rules 110/90/30, eps 0.015/0.05). CPU time per grid (T = 10, all marginals, 1 thread, busy box): W = 12 0.020 s,
  W = 14 0.065 s; the sweep: W = 14 0.20 s, W = 16 1.07 s.

## 2. Network, training, heads
- ecanet.Net (smallnet Block, d 128, 4 heads, ff 256, dropout 0.1), **L in {2, 3, 4} layers**; training arm **T1** (plain
  masked loss, reveal count k ~ U{1..119}); AdamW 1e-3, batch 256, EMA 0.995, early stopping on validation OUTCOME log loss
  (2,000 val grids with random reveal masks), evaluated every 1,000 steps, patience 10, cap 70 minutes.
- Post-hoc heads on the cal split (backbone frozen, outcomes only): stiffness (Mechanism A, beta floor) and the round-2 key
  without h_i (`--key-mode v2`: LayerNormed attention fingerprint over the newly materialised cells S minus E_0, else S).

## 3. D0 gate and depth selection (spec "Exact Changes" 3)
- Per rule, L is the **largest L in {2, 3, 4} whose seed-0 run passes the gate on validation**: P0 S1 KL <= 0.015 AND
  P0 KL >= 0.08 (rule 110) / >= 0.05 (rule 90) on at least two of F, B1, B2. If no L passes for a rule, R3-D0 is FAIL for that
  rule and its R3-D2/D3 rules are reported as "gate failed" with the numbers of L = 4 (descriptive only).
- For time (results due ~13:00Z), all 18 runs (2 rules x L 2/3/4 x seeds 0/1/2) are launched together instead of
  sequentially; the selection reads ONLY the seed-0 validation P0 numbers, exactly as the sequential procedure would. Seeds
  1-2 at the other depths are reported as secondary.

## 4. Search arms and pools (spec "Exact Changes" 4)
- K = 16 chains per query, M in {0, 1, 2, 4, 8, 16} materialised cells (sampled from q(z | S_{t-1})), on F, B1, B2
  (500 validation and 500 test instances per family; validation outcomes fit the pools).
- Selectors: **stiff** (sample z by exp(tau beta_int), tau = 5), **conf** (most peaked), **voi** (VoI among the 6 stiffest),
  **voiu** (VoI among the 6 highest-variance cells), **hyb** (P6-Hybrid: if some unrevealed cell has entropy
  H(q(z | S)) < 0.20 nats, materialise the lowest-entropy one; otherwise use voiu), **rand**.
- Pools: **sw** = sw_mix (primary; mu = (lam P0 + sum w_r q_r)/(lam + sum w_r), w_r = beta_int^eta, (lam, eta) fitted on
  validation outcomes per selector and M), **mix** (plain mixture), **dd** (Gram dedup with v2 keys); lw, g0, g1 also logged.
- Compute per query: measured forward passes (hyb's cost varies with how often it branches; its branch share is logged).

## 5. Decision rules (gemini's R3 rules, operationalised; 3-seed means at the selected L, paired 95% t-intervals, 2 df)
- **R3-D0**: at the selected L, 3-seed mean validation P0 S1 KL <= 0.015 and P0 KL >= 0.08 (rule 110) / >= 0.05 (rule 90) on
  at least two of F, B1, B2. Per rule.
- **R3-D2a** (per rule, both reported): on F at M = 16 under sw, EACH of stiff, conf, hyb: (i) 1 - KL(arm)/KL(P0) >= 0.25
  (mean of per-seed cuts) with the paired CI of KL(P0) - KL(arm) above 0, and (ii) 1 - KL(arm)/KL(rand) >= 0.15 with the
  paired CI of KL(rand) - KL(arm) above 0. PASS iff all three arms pass (i) and (ii).
- **R3-D2b** (rule 110; rule 90 descriptive): under sw at M = 16, hyb cuts KL vs P0 by >= 25% on each of F, B1, B2 (paired CI
  above 0), beats stiff on B1 and on B2 (paired CI of KL(stiff) - KL(hyb) above 0), and beats rand on F (paired CI of
  KL(rand) - KL(hyb) above 0). PASS iff all hold.
- **R3-D3** (per rule): on the 2,000 held-out (i, S) pairs (test grids, k ~ U{1..119}), Spearman with -KL(P*(X_i | x_S) ||
  q(X_i | x_S)) of (a) beta_int, (b) plain confidence ln 2 - H(q), (c) 1-step self-consistency
  -E_{z ~ q(z* | S)}[(q(X_i | S, z*) - q(X_i | S))^2] with z* the unrevealed cell (not i) of highest beta_int. PASS iff
  max(Spearman(a), Spearman(c)) - Spearman(b) >= 0.05 (3-seed means). Also reported: the same within 10 quantile bins of
  H(q) (entropy-matched, bin-size-weighted mean).
- Also reported (no rule): D1 (stiffness vs exact information gain, blanket AUC), D3a/D3b, all selectors x pools x M, the
  replication of round 2's sw_mix finding.

## 6. Runs and compute
- Priced smoke first: one L4, rule 110 L = 4 and rule 90 L = 2, seed 0, 8 training minutes, full evaluation sizes, to time
  training throughput and the evaluation (results not used for anything else).
- Main: 18 runs on 9 on-demand L4 VMs (2 runs per VM, pairing a deep with a shallow run), REMOTE_MINUTES <= 220
  (GPU_RUN_TIMEOUT <= 232 min), launched through the eca-local zone-flexible gpu-run copy, budget group overnight-ca.
  Expected ~1.5-2 h per VM, ~$12-16; hard round-3 cap $40 checked before each launch.
- Analysis: `analyze_r3.py` (written before the results are read) applies Sec. 3 and 5 mechanically.
