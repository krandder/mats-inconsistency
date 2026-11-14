# E-CA pre-registration (eurisko, 2026-10-15)

Written before the main launch. Sources: base design `/home/agent/workspaces/mats/outbox/overnight-experiments-design-2026-10-15.md`,
accepted review `overnight-experiments-review-gemini-2026-10-15.md` and design v2 `overnight-experiments-design-v2-2026-10-15.md`
(where they disagree, v2 wins). Code: `ecadata.py` (data, exact inference, families), `ecanet.py` (model),
`ecarun.py` (one run: training + full evaluation), `gpu/` (launcher), `analyze.py` (verdicts; written after launch, applies
the rules below mechanically). Authorised by Kelvin 2026-10-15 00:3xZ; coordinator opus-mats; executor eurisko (this subagent).
Standing rules: training uses outcomes only; nothing hand-coded about the domain in any arm; every arm could run on
real-world questions. Exact posteriors are used only for scoring, the D0 gate and the D1 stiffness score.

## 1. Data
- Elementary CA, rules **110 and 90** (rule 30 dropped, v2 change 6). x[t+1,c] = rule(x[t,c-1], x[t,c], x[t,c+1]) XOR Bern(eps),
  periodic, row 0 iid Bern(1/2). **eps = 0.015** (v2 change 2). **W = 8, T = 6** (48 cells).
- Fixed permutation of the 48 cell ids, seed 20261015 + 1000 T + W (`ecadata.make_perm`), identical in every run:
  token j holds grid cell PERM[j]. T=6, W=8: PERM = [28, 39, 3, 43, 46, 40, 37, 17, 2, 14, 21, 42, 45, 1, 4, 20, 22, 47, 31, 32, 26,
  25, 15, 11, 24, 44, 10, 34, 6, 30, 5, 41, 18, 27, 38, 16, 0, 35, 36, 23, 12, 9, 8, 7, 13, 29, 33, 19]. The model sees token
  ids only (learned position embeddings, no relative or 2-D bias).
- Splits (disjoint simulator seeds): train 50,000 grids; **cal 5,000 grids** (held out from the main loss; used only to fit the
  stiffness and key heads after training); val 3,000; test 3,000.
- Exact inference: forward-backward over the row chain (256 row states). Brute-force check over all 2^16 grids at W = 4,
  T = 4, rules 110/90/30, eps 0.015 and 0.05, 60 random evidence masks each: max |difference| = 3.8e-15 (run 01:00Z).

## 2. Query families (evaluation-side constructions; instance i uses grid i of the split)
- **F** forward: row 0 revealed; Y = a cell in rows 3..5.
- **B1** two-row bridge: rows 0 and 5 revealed; Y = a cell in rows 2..3.
- **B2** anchor bridge: row 0 cols 0..3, row 5, columns 0 and 1 (all rows) revealed; Y = an unrevealed interior cell
  (rows 1..4, cols 2..7).
- **R** random: reveal count k ~ U{1..47}; Y = a random hidden cell.
- **S1** one-step: like R but Y's three past parents are revealed (D0's one-step check only).
- Mean exact H(Y | E) on test (nats), rule 110 / 90: F 0.355 / 0.331, B1 0.064 / 0.180, B2 0.084 / 0.027, R 0.175 / 0.179,
  S1 0.037 / 0.036.

## 3. Model and training
- smallnet `netdef.Block` unchanged (pre-LN, d = 128, 4 layers, 4 heads, ff 256, dropout 0.1); 48 tokens; input per token:
  MASK / revealed 0 / revealed 1; output head logit q(x_i = 1 | x_S) for every cell.
- **Training reveal count k ~ U{1..47}**, uniform random subset (v2 change 3).
- **T1** plain masked loss: sum over hidden cells of the outcome log loss, mean over the batch (B = 256 grids/step).
- **T2** rbtd8: smallnet's reveal-chain TD with 8 levels and lam = 0. Level 0 = the episode's reveal set (k0 ~ U{1..47});
  levels 1..7 reveal further true cells along the same random order (sorted uniform cut points in [k0, 47]); level i's
  prediction for cells still hidden at level i+1 is trained toward level i+1's detached prediction, cells revealed at level
  i+1 toward their true value, the last level toward true values; levels summed (smallnet convention). B = 64 grids/step x 8.
- AdamW lr 1e-3 (200-step warmup), wd 0.05, clip 1.0, bf16 autocast on GPU, EMA 0.995 (evaluated model = EMA).
- **Early stopping on validation OUTCOME log loss** (3,000 val grids, fixed masks with k ~ U{1..47}, mean over hidden cells),
  evaluated every 500 steps, patience 10 evaluations, plus a wall-clock cap (`--max-minutes`, set per launch so that every
  run finishes before 07:00Z); best EMA checkpoint restored. Never the posterior.
- Seeds 0, 1, 2 for every (rule, arm): 2 rules x 2 arms x 3 seeds = 12 runs.

## 4. Heads fitted after training (backbone frozen, cal split, outcomes only)
- **Stiffness** (Mechanism A): s_hat(i | S) = softplus(MLP(h_i)) regressed (MSE) onto the realised held-out surprise
  -log q(x_i | x_S) of hidden cells; 3,000 steps x 256 grids. **beta_int(i | S) = max(0, 1/(s_hat + 0.01) - 1/(ln 2 + 0.01))**
  (stiffness floor, v2 change 5; e = 0.01 nats).
- **Key** (v2 change 4): k(i | S) = normalise(W_K [sum_{j in S} alpha_{i->j} pos_emb(j) ; h_i]) in R^16, alpha = token i's
  last-layer attention (head average) on revealed sources. W_K and a scale are trained by the outcome log loss of the
  Gram-crowded pool on multi-view bags (2..5 views of one target cell): half **shared-core bags** (each view = a common random
  reveal set + ~3 random extra cells) and half **disjoint bags** (a random reveal set split at random into disjoint views).
  Interpretation note: v2 says "half same-blanket views, half disjoint directions"; "same blanket" would require the CA
  topology, which training may not use, so redundant views are built as shared-core views (no topology). The topology enters
  only the D3a evaluation views below.

## 5. Inference arms (frozen network; one query cell Y per instance)
- **P0** direct: q(Y | E), one pass.
- **Chains** (v2 change 1, sequential materialisation): **K = 16 chains** per query, S_0 = E, S_t = S_{t-1} U {z_t}; one pass on
  S_{t-1} gives q(z | S_{t-1}) and beta_int(z | S_{t-1}) for every masked z; z_t (never Y) is materialised by sampling
  z_t ~ q(z_t | S_{t-1}). Selection rules:
  - **P2-Stiff**: sample z_t with probability proportional to exp(tau beta_int(z | S_{t-1})), tau = 1 (explores several
    stiff fronts across chains). 1 pass per step.
  - **P2-VoI**: among the 6 masked cells with the highest beta_int, pick argmax_z E_{z~q}[(q(Y | S_{t-1}, z) - q(Y | S_{t-1}))^2].
    13 passes per step.
  - **P3-Conf**: the most peaked masked cell, argmax |q(z | S_{t-1}) - 1/2|. 1 pass per step.
  - **P4-Rand**: a uniformly random masked cell. 1 pass per step.
  Each chain emits the claim q(Y | S_M), beta_int(Y | S_M), k(Y | S_M) at **M in {0, 1, 2, 4, 8, 16}** (prefixes of one
  16-step chain; selection never depends on M).
- **Pools** over the K claims, p0 = the network's own q(Y | nothing revealed):
  - **lw** (P2's pool): prior-corrected gamma = 0 pool with learned w: logit mu = logit p0 + sum_a w_a (logit q_a - logit p0),
    w_a = beta_int,a / beta_0; 1/beta_0 fitted per (selector, M) on VALIDATION OUTCOMES (1,000 val instances per family
    F, B1, B2, R) by Newton with the closed-form gradient dL/dw_a = -(l_a(y) - E_mu l_a). The prior edge carries the
    residual weight 1 - sum_a w_a.
  - **dd** (**P5**): lw with Gram crowding w_a,eff = w_a / sum_b max(0, <k_a, k_b>); its own beta_0 fitted the same way.
  - **P6** aggregators on the same claims: **mix** (mixture of claims; SNIS with unit weights, since proposal = target = the
    network's chain rule), **g0** (plain gamma = 0: mean of logits, weights sum to 1, no prior), **lw**, **g1** (gamma = 1
    product: logit p0 + sum_a (logit q_a - logit p0)).
- P1 (random-order rollouts with SNIS) is covered by P4-Rand chains with the mix aggregator (v2 replaced one-cell
  contraction with materialising chains; a separate P1 would duplicate P4).
- Test: 1,000 instances per family (F, B1, B2, R) for chains; P0 on all 3,000 per family.
- Why K = 16 (fixed 01:15Z from first principles, before any trained-model result was read): a materialising chain with
  sampled intermediate cells emits a nearly sharp claim, so pooling K chains is a K-sample Monte Carlo estimate whose
  noise floor is O(1/K) nats (about 0.07 nats at K = 4 for a typical F query with p* ~ 0.93, about 0.02 at K = 16).
  K = 4 would make the test about Monte Carlo noise, not about routing; K = 16 costs ~3,300 passes per query for VoI
  and 272 for the 1-pass-per-step rules, minutes on one L4. D3b keeps one chain per direction as v2 specifies.

## 6. Metrics
Primary: KL(exact posterior of Y || estimate), mean over instances, per family, rule, training arm, selector, pool and M.
Secondary: outcome log loss, ECE (15 bins). Compute per query: forward passes and FLOPs (passes x 2-FLOP-per-MAC forward
cost of the network). Stiffness: Spearman and AUC (below). Dedup: drift ratio (below). Paired statistics are over the 3 seeds
(same test instances in every seed): mean and 95% t-interval (2 df, t = 4.303).

## 7. Decision rules (v2's D0-D3; operationalised here, applied by `analyze.py` without change)
- **D0 gate** (validation, per run, recorded before the run's own sweep): PASS iff P0 KL >= 0.05 nats on each of F, B1, B2
  and P0 KL <= 0.02 on S1. Contingency: if the GPU smoke models (8 training minutes) already have KL < 0.01 on all of F, B1,
  B2, the main launch uses T = 8 instead of T = 6 (T = 8 data are pre-built). The final D0 verdict uses the 12 main runs
  (3-seed mean per rule and arm).
- **D1 stiffness**: per (rule, arm): over 2,000 held-out (i, S) pairs on TEST grids (i random cell, S random with k ~ U{1..47},
  i not in S), Spearman(beta_int(i | S), ln 2 - H_exact(X_i | X_S = x_S)) >= 0.60 AND AUC(beta_int; "S contains a valid
  deterministic local blanket of i": i's 3 past parents, or for one of i's 3 children, the child plus its other 2
  parents) >= 0.85. PASS iff both 3-seed means clear the thresholds; the strict version (CI lower bounds clear them) is
  also reported. Secondary: the expected-entropy version (500 pairs x 8 draws of x_S).
- **D2 search**: per (rule, arm), the P2 variant is **declared from validation only**: the one of {Stiff, VoI} with the lower
  3-seed mean validation outcome log loss of the lw pool at M = 16 over F, B1, B2. Then for EACH of F, B1, B2:
  (a) relative KL cut vs P0 at M = 16, 1 - KL(P2, lw, 16)/KL(P0) >= 0.30, and (b) vs P4-Rand (lw, M = 16)
  1 - KL(P2)/KL(P4) >= 0.20; each PASS iff the 3-seed mean cut clears the threshold and the paired 95% CI of the KL
  difference excludes 0. D2 passes iff (a) and (b) pass on all three families. Also reported: the cut vs P2 at M = 0 (the
  recalibrated one-pass belief), which separates search from recalibration.
- **D3 dedup**: (a) 1,000 test targets in rows 2..5; m in {1, 2, 5} redundant views = Y's 3 past parents + 3 random cells
  from rows <= t-2 (exactly redundant given the parents). Drift = mean |logit mu_m - logit mu_1| with a common unit house
  scale. PASS iff drift(P5 dd) / drift(scalar lw) < 0.20 at m = 5 and at m = 2 (3-seed means).
  (b) On B1 and B2, one forward chain (hidden cells in rows above Y, top-down) and one backward chain (rows below Y,
  bottom-up), 16 materialisations each by sampling; P5 pool of {fwd, bwd} vs P5 of each alone (each pool's scale fitted on
  validation). PASS iff KL and log loss of {fwd, bwd} are strictly below both singles on B1 and on B2 (3-seed means; paired
  CIs reported).
A failed rule is a result. No tuning after looking at test. Crashed runs may be rerun with the identical configuration.

## 8. Compute and stop rules
GCP L4 (g2-standard-8) via `logical-soup/infra/gpu-run.sh`, budget group overnight-ca, cap $110 at the $0.87/h ceiling;
`guard-eca.sh` deletes any eca VM older than 4.2 h and kills the youngest at $105 estimated. On-demand (STANDARD) by default
(smallnet saw 10 of 14 spot L4s preempted). MATS cluster: dropped (opus-mats 01:0xZ: 8/8 L40s allocated until 11-19).
Runs stop at patience or the wall-clock cap; results come home as CSV/JSON/PNG only. All VMs deleted at the end.

## 9. Changes log (UTC)
- 00:59Z read design v2 + gemini review; all seven accepted changes apply to E-CA as follows (implemented 01:00-01:08Z,
  before any training run):
  1. P2/P3/P4/P5 -> sequential materialisation with K chains (Sec. 5). Applied.
  2. eps 0.05 -> 0.015. Applied.
  3. Training reveal count k ~ U{1..47}; families F, B1, B2, R (+ S1 for D0). Applied.
  4. Key = attention-weighted source fingerprint + h_i, W_K trained on multi-view bags (shared-core vs disjoint; see Sec. 4
     for why not "same-blanket"). Applied with that interpretation.
  5. Stiffness floor beta = max(0, 1/(s+e) - 1/(ln 2+e)), e = 0.01. Applied.
  6. Rule 30 dropped (rule 54 not added). Applied.
  7. (E-SUD only) not applicable.
  Decision rules D0-D3 of v2 replace the base design's D1-D4. Applied (Sec. 7).
- 01:0xZ MATS cluster lane dropped on opus-mats' instruction. Applied.
- Deviation: the cal split (5,000 extra training-distribution grids) is added so the stiffness head learns HELD-OUT surprise.
- Deviation: P1 folded into P4-Rand + mix (Sec. 5).
- 01:41Z infrastructure (no effect on the science): on-demand L4s were stocked out in all three us-central1 zones at
  01:36-01:39Z, so launches retry through an eca-local copy of infra/gpu-up.sh and gpu-run.sh (`gpu/gpu-up-zones.sh`,
  `gpu/gpu-run-zones.sh`; same lib.sh, GPU-LOG, budget lock and caps) whose zone list adds us-east4, us-east1, us-west1 and
  us-west4. The shared infra files are unchanged.
- 01:50Z analyze.py (written after launch, before any main-run result existed) adds SECONDARY readouts only: the base
  design's rules (VoI vs Conf and vs Rand at equal M on B1/B2, P2-VoI at M = 8 vs P0), a compute-per-query table, and
  figures (KL vs M, KL vs passes, stiffness scatter). The Sec. 7 verdict logic is unchanged.
- 03:47Z analyze.py adds two more SECONDARY readouts, after the replication batch and 4 primary runs had been read: the
  P1 arm (random-order chains + mixture, and every selector under the mixture) vs P0 with paired CIs, and the fraction of
  queries whose stiffness-weighted pool at M = 0 equals the prior. They are labelled as secondary; Sec. 7 is unchanged.
- 04:35Z correction (after the audit): the 03:47Z line above says 4 primary runs had been read; in fact the provisional
  analysis read at 03:46-03:47Z already included 9 of the 12 primary (eca-b) runs.
- 03:50Z analyze.py adds one more SECONDARY readout: the AUC of the exact information gain itself on D1's blanket label
  (the ceiling for any entropy-based score), computed from d1-pairs.csv. D1's verdict logic is unchanged.

## 10. GPU smoke (priced) and launch configuration (written 01:34-01:35Z, before the main launch at 01:35:05Z)
- Smoke `eca-smoke`: one on-demand L4, VM up 01:09-01:35Z (~26 min, about $0.38 at the $0.87/h ceiling), 4 jobs at once
  (rules 110/90 x T1/T2, seed 0, 8 training minutes, K = 4 chains, full evaluation). All 4 finished rc = 0 in ~19.5 min.
  Throughput with 4 jobs per L4: T1 31 steps/s (256 grids), T2 14 steps/s (64 grids x 8 levels); heads 88 s; evaluation
  ~10 min at K = 4 (VoI dominates: ~57 s per family-split).
- Read from the smoke: timing and the VALIDATION D0 numbers only (no test metric was read). D0 (validation P0 KL, nats,
  8-minute models): r110-T1 F 0.154 B1 0.138 B2 0.103 S1 0.007; r110-T2 F 0.203 B1 0.199 B2 0.190 S1 0.016;
  r90-T1 F 0.020 B1 0.026 B2 0.030 S1 0.039; r90-T2 F 0.022 B1 0.037 B2 0.067 S1 0.059. No model is below 0.01 on all of
  F, B1, B2, so the pre-registered contingency does not trigger: **T = 6 stays**. (Rule 90's 8-minute models fail the gate
  the other way: one-step S1 is not yet learned, KL 0.039-0.059 > 0.02, and long-range KL is already below 0.05. The final
  D0 verdict uses the 12 trained runs.)
- Main launch: 6 on-demand L4 VMs (`eca-r{rule}-s{seed}`), each running T1 and T2 of one (rule, seed) at once.
  Per run: `--max-steps 400000 --eval-every 1000 --patience 10 --max-minutes 140`, K = 16, other settings as above.
  REMOTE_MINUTES 205 (gpu-run timeout 217 min <= 235). Reserve per VM 0.87 x (217/60 + 0.6) = $3.67; 6 VMs <= $22.

## 11. Secondary hedge, rule 90 at T = 8 (written 01:36Z, before its launch at 01:36:50Z)
The smoke's rule-90 models already had long-range KL 0.02-0.07 at 8 training minutes, so trained rule-90 models may show
no depth gap at T = 6 (D0 fails without reaching the < 0.01 trigger). As a bounded hedge, the same 6 rule-90 runs
(T1/T2 x seeds 0-2) are also run at **T = 8** (64 cells, same eps, W, model, training and evaluation; its own fixed
permutation, seed 20261015 + 8008; families as defined in ecadata.family_masks for T = 8: F queries rows 4-7, B1 rows 3-4,
B2 interior rows 1-6). Launch: 3 on-demand L4 VMs (`eca-t8-s{seed}`), training cap 130 min, REMOTE_MINUTES 205.
Reported separately as **secondary**; the D0-D3 verdicts for T = 6 stand on their own. The same rules (Sec. 7) are
applied to the T = 8 runs and labelled as such.

## 12. Evaluation bug and crash-rerun `eca-b` (written 02:35Z)
- 02:29Z, reading the first finished run (r90-T1-s2): the pools were computed in float32, so a confident pooled
  probability rounded to exactly 1.0 and gave infinite KL / NaN log loss. Affected: lw and dd pools at small M, the g1
  product everywhere, D3b's B2 pools, and the matching CSV logits (stored as +inf). Not affected: P0, D0, D1, D3a, the
  pool scale fits, and most M = 16 statistics (finite in that run). Training is unaffected. The models are not kept on the
  VMs, so the original runs cannot be re-evaluated.
- Fix (02:31Z, `ecarun.py` header note; the as-run 01:35Z file is kept at /tmp/eca/ecarun-as-run-0135Z.py and in each
  frozen src copy): every probability, pool and metric in float64; probabilities clipped to [1e-12, 1 - 1e-12] in both the
  metrics and the CSV logits; CSV logits to 7 decimals. Training code unchanged.
- 02:33Z: crash-rerun of the 12 main runs with the identical configuration (same seeds, splits, caps, 2 runs per L4), run ids
  `eca-b-r{rule}-s{seed}`, outputs `main2/` (PREREG Sec. 7 allows rerunning crashed runs with the identical configuration).
- Primary-source rule, declared now (02:35Z): for each (rule, arm, seed) the eca-b run is primary if its result.json is home
  by 06:15Z; otherwise the original run is used and every probability re-derived from its CSV is capped to
  [6e-8, 1 - 6e-8] (float32 resolution; a lenient cap for over-confident pools, flagged in the tables), and any statistic
  that is non-finite in its result.json is reported as not evaluable. The original runs are also reported as a replication
  (same rules, capped). The T = 8 hedge runs (Sec. 11) carry the same bug and are not rerun; they are reported capped.
