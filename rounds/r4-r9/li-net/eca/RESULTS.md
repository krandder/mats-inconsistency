# E-CA: stiffness and search on a permuted cellular-automaton grid: results

eurisko study, 2026-10-15 (overnight). Written by an Opus 5.5 subagent for eurisko. Design: `/home/agent/workspaces/mats/outbox/overnight-experiments-design-2026-10-15.md`, with gemini's review accepted in full as design v2 (`overnight-experiments-design-v2-2026-10-15.md`). Pre-registration, change log and every deviation with its time: `PREREG.md`. Independent audit: `AUDIT.md` (it reproduced every verdict and number; its wording findings M1-M4 and L1-L13 are
applied below, see Sec. 14). Every number below is in `results/summary.json` (primary) and is re-derivable from the per-run files under `results/raw/`.

## 1. TL;DR

- **Verdicts (primary runs, 3 seeds per rule and training arm, 4 cells = 2 rules x 2 training arms):** D0 gate **FAIL** in all four. D1 stiffness **PASS** for rule 110 with both arms (Spearman 0.95, AUC 0.87) and for rule-90 T2 (0.925, 0.857); **FAIL** for rule-90 T1 on AUC (0.848 against a 0.85 bar). D2 search **FAIL** in all four. D3 dedup **PASS** in three of four; rule-90 T2 fails D3b on B2 (pooled forward + backward 0.058 vs forward alone 0.041). Table in Sec. 3.
- **There is no depth gap to close for rule 90.** The one-pass network is already near-exact: test P0 KL to the exact posterior is 0.006-0.010 nats on F, B1 and B2 (T1; T2 is the same except one seed on B1), and it stays near-exact on a deeper grid (T = 8 hedge, T1: 0.001-0.011). **Rule 110 has a real gap on bridge queries** (P0 KL 0.09-0.10 on B1 and 0.06-0.07 on B2, against exact entropies of 0.06-0.08 nats) but not on forward queries (0.019-0.025).
- **Search through stiff arcs does not deliver; random-order sampling does (secondary readout).** The pre-registered P2 (VoI chains, declared from validation) with the stiffness-weighted pool is worse than one pass, or no better, on every family. The P1 arm (random-order chains averaged by a plain mixture) was pre-registered as an arm but not as a decision rule, and its comparison with P0 was added to the analysis at 03:47Z, after the pattern had been seen in the replication batch: it **cuts rule-110 bridge KL by 28-50%**: T1 B1 0.096 -> 0.048 (cut 0.50 [0.44, 0.56], 3 seeds), B2 0.069 -> 0.040 (0.41 [0.22, 0.60]); T2 B1 0.088 -> 0.049 (0.44 [0.39, 0.49]), B2 0.058 -> 0.041 (0.28 [0.21, 0.35]), at M = 16 cells per chain, falling steadily up to M = 8, and it replicates in the original batch (0.36-0.50). Stiffness-first chains with the mixture give smaller gains on rule-110 B2 only (cut 0.09-0.17, paired CIs above 0 in both arms); VoI and confidence-first chains give none. Where one pass is already good, search barely helps: on rule-110 forward queries random order helps at M = 1-2 (T2 cut 0.21 [0.06, 0.36]) and hurts at larger M; on rule 90 no arm gains more than 3%.
- **The stiffness-weighted pool is itself harmful.** With w_a = beta_int,a / beta_0 and one fitted scale, pooling 16 copies of the one-pass claim (M = 0) turns rule-110 forward KL from 0.025 into 0.21 (T1; T2 0.019 into 0.20): soft-but-informative claims get tiny weights and are pulled to the prior, stiff claims get large ones.
- **Stiffness is learned, but it is the network's own confidence.** beta_int ranks 2,000 held-out (cell, evidence) pairs by exact information gain with Spearman 0.92-0.95, and the network's plain predictive entropy ranks them just as well (0.93-0.95). AUC for "the evidence holds a deterministic local blanket of the cell" is 0.87 (rule 110) and 0.85-0.86 (rule 90), while the exact information gain itself scores 0.87 and 0.89 on this label (not an upper bound: beta exceeds it in 3 of 6 rule-110 runs). The label also counts child triples that cannot determine the cell (6% of rule-90 positives); with a rule-aware label rule-90 T1's AUC is 0.850, so its pre-registered FAIL is by a hair.
- **Round 2** (gemini's follow-ups, pre-registered in `PREREG-R2.md`): a 1-layer net, an anchored pool, keys without h_i and variance-gated VoI; results in `RESULTS-R2.md` (page `eca-r2.html`).
- **Cost: $17.02** at the on-demand ceiling rate for 16 L4 VMs (smoke, original batch, T = 8 hedge, eca-b rerun), well under the $110 group cap; zero eca VMs remain (checked with gcloud at 04:09Z); the MATS cluster was full and was not used (zero jobs).

## 2. What ran

- Question (Kelvin, 10-14, via the gemini chat): can one any-order network over homogeneous cells learn which of its own
  conditionals are stiff one-step laws (stiffness head beta_int), and can a search policy that routes through stiff arcs
  improve beliefs about long-range queries with compute, where the one-pass conditional fails? Secondary: do key
  fingerprints deduplicate redundant arcs?
- Data: noisy elementary CA, rules 110 and 90, eps = 0.015, W = 8, T = 6 (48 cells, row 0 iid), periodic. One fixed random
  permutation of the 48 cell ids; the network sees permuted token ids only, with learned position embeddings and no
  topology bias. Splits: train 50,000 grids, cal 5,000 (held out for the stiffness and key heads), val 3,000, test 3,000.
- Exact posteriors: forward-backward over rows (256 states), checked against brute force over all 2^16 grids at W = T = 4
  (max difference 3.8e-15). Used only for scoring, the D0 gate and the D1 score.
- Network: smallnet's transformer block unchanged (d 128, 4 layers, 4 heads); heads: Bernoulli logit per cell, stiffness
  (predicted held-out surprise, with the v2 floor), 16-d key (attention-weighted source fingerprint + h_i).
- Training arms, 3 seeds each: T1 plain masked loss; T2 reveal-chain TD with 8 levels (smallnet's rbtd, lam = 0). Reveal
  count k ~ U{1..47}. Early stopping on validation outcome log loss.
- Search: K = 16 chains per query; each chain materialises M cells one at a time (S_t = S_{t-1} + z_t, z_t sampled from
  q(z_t | S_{t-1})); selection P2-Stiff (sample by exp(beta_int)), P2-VoI (top-6 stiff cells, max expected squared change of
  q(Y)), P3-Conf (most peaked), P4-Rand (uniform). Claims q(Y | S_M) pooled by the prior-corrected learned-w pool (lw, scale
  fitted on validation outcomes), with Gram dedup (P5, dd), and by the P6 aggregators.
- Query families: F (row 0 known, query rows 3-5), B1 (rows 0 and 5 known, query rows 2-3), B2 (half of row 0, row 5,
  columns 0-1 known, query the interior), R (random reveal), S1 (one-step check: query's 3 parents known).
- Runs: 2 rules x 2 training arms x 3 seeds = 12 primary runs (the `eca-b` batch, 6 L4 VMs with 2 runs each, launched
  02:33Z after the float32 fix, Sec. 10), the same 12 as the original 01:35Z batch (now the replication, Sec. 9), 6
  rule-90 runs at T = 8 (secondary hedge, Sec. 9), and one priced 4-job smoke. Every training run stopped on patience
  (10 validation evaluations without improvement) after 17-76 minutes of training (primary runs); the 140-minute cap
  never bound.
- Test sets: P0 on 3,000 instances per family; chains, pools and D2 on the first 1,000 instances per family (same
  instances for every seed, selector and M, so seed-paired comparisons are on identical queries); D1 on 2,000 pairs.
- Compute: GCP on-demand L4s (g2-standard-8), budget group overnight-ca (cap $110). MATS cluster not used (full).

## 3. Decision rules (pre-registered in PREREG.md Sec. 7; v2's D0-D3 replace the base design's D1-D4)

Primary runs (eca-b, float64 evaluation), 3 seeds per row; numbers are 3-seed means, 95% intervals in Sec. 13 and
`results/tables.md`. "cut vs P0" = 1 - KL(P2)/KL(P0); negative = worse than one pass.

| rule, arm | D0 gate (val P0 KL F / B1 / B2; S1) | D1 (Spearman; AUC) | D2 (declared P2; cut vs P0 at M=16, F / B1 / B2) | D3a (drift ratio m=5; m=2) | D3b (KL both vs best single, B1; B2) | D3 |
|---|---|---|---|---|---|---|
| r110-T1 | **FAIL** (0.025 / 0.080 / 0.058; 0.008) | **PASS** (0.948; 0.871) | **FAIL** (voi; -6.91 / -0.03 / -0.35) | **PASS** (0.043; 0.133) | **PASS** (0.093 vs 0.125; 0.103 vs 0.134) | **PASS** |
| r110-T2 | **FAIL** (0.020 / 0.077 / 0.048; 0.008) | **PASS** (0.945; 0.869) | **FAIL** (voi; -8.76 / -0.21 / -0.46) | **PASS** (0.049; 0.153) | **PASS** (0.104 vs 0.125; 0.117 vs 0.135) | **PASS** |
| r90-T1 | **FAIL** (0.006 / 0.010 / 0.007; 0.008) | **FAIL** (0.923; 0.848) | **FAIL** (voi; -4.76 / -5.41 / -5.36) | **PASS** (0.017; 0.051) | **PASS** (0.049 vs 0.082; 0.033 vs 0.035) | **PASS** |
| r90-T2 | **FAIL** (0.006 / 0.052 / 0.009; 0.021) | **PASS** (0.925; 0.857) | **FAIL** (voi; -14.27 / -1.71 / -6.93) | **PASS** (0.024; 0.073) | **FAIL** (0.098 vs 0.146; 0.058 vs 0.041) | **FAIL** |

How each rule was applied: D0 on validation P0 KL (PASS needs >= 0.05 on each of F, B1, B2 and <= 0.02 on the one-step
family S1). D1 PASS needs both 3-seed means over the bar (Spearman >= 0.60 with ln 2 - H_exact, AUC >= 0.85). D2 needs, on
each of F, B1 and B2, a 3-seed mean cut >= 30% vs P0 and >= 20% vs P4-Rand at M = 16, each with the paired 95% CI of the
KL difference excluding 0; the P2 variant (Stiff or VoI) was declared per row from validation outcome log loss only (VoI
in all four rows). D3a needs the dedup/scalar drift ratio < 0.20 at m = 5 and m = 2; D3b needs the pooled forward +
backward chains to beat both single chains in KL and log loss on B1 and on B2.

Notes on the verdicts:
- **D0 fails for different reasons per rule.** Rule 90: no gap anywhere (one-pass KL <= ~0.01, except one T2 seed on B1).
  Rule 110: B1 clears the bar (0.08), B2 does in T1 (0.059) but not in T2 (0.048), the one-step check passes (S1 0.008), but forward
  queries are already easy for one pass (0.020-0.025 < 0.05). The pre-registered contingency (switch to T = 8) did not
  trigger, because it was keyed to the 8-minute smoke models and none of them was below 0.01 (PREREG Sec. 10); the rule-90
  T = 8 hedge (Sec. 9) shows that a deeper grid would not have created a gap for rule 90 anyway.
- **Rule-90 T2 is unstable across seeds.** In 4 of 9 rule-90 T2 runs (primary s2, replication s1 and s2, T = 8 s1) the
  one-step check fails (S1 KL 0.04-0.07) and one-pass bridge KL is 0.07-0.14; no rule-90 T1 run exceeds 0.015 / 0.011.
- **D2's failure is not only about search.** The same pool at M = 0 (16 copies of the one-pass claim, no search) is already
  far worse than P0 (rule-110 T1: F 0.213 vs 0.025, B1 0.141 vs 0.096, B2 0.109 vs 0.069). Relative to that pooled
  baseline, VoI search does recover some ground (cut vs P2 at M = 0: F 0.12, B1 0.29, B2 0.16), but never reaches P0.
- **D3b passes (three of four cells), but not because the keys separate directions.** The cosine between the forward and
  backward chains' keys is 0.83-0.97, close to the 0.96-0.99 of truly redundant views, so Gram crowding mostly averages the
  two claims. Averaging two chains that condition on different sampled cells beats either chain alone; the keys did not
  learn "past parents vs future children" as the review intended. Where one direction is much better (rule-90 T2 on B2:
  forward 0.041, backward 0.074), averaging loses to the better chain and D3b fails.

## 4. Search: KL vs materialised cells, and vs compute

Top row: the pre-registered stiffness-weighted pool (lw). Bottom row: the same chains averaged by a plain mixture (the P1
aggregator; for random-order chains this is the P1 arm). Dashed line: one pass (P0). Shaded: 95% interval over 3 seeds.

IMG: results/kl-vs-M-r110-T1.png

IMG: results/kl-vs-M-r110-T2.png

IMG: results/kl-vs-M-r90-T1.png

IMG: results/kl-vs-M-r90-T2.png

What the curves say (rule 110 unless stated; full numbers in Sec. 13):
- **Random-order chains + mixture are the only search that helps, and only where one pass is bad.** Rule-110 bridges improve
  monotonically up to M = 8 and hold at M = 16 (T1: B1 0.096 -> 0.047 at M = 8, cut 0.51 [0.46, 0.56]; B2 0.069 -> 0.039, cut
  0.43 [0.33, 0.53]; T2: B1 0.42 [0.38, 0.46], B2 0.32 [0.23, 0.42]). On forward queries a little sampling helps (M = 2:
  T1 0.025 -> 0.017, cut 0.28, CI includes 0; T2 cut 0.21 [0.06, 0.36]) and more hurts (M = 16: 0.049 and 0.044). On rule 90, where one pass is near-exact, no search arm gains more than 3% and most add error.
- **Status of this finding.** The mixture-vs-P0 readout is secondary: P1 was a pre-registered arm, but no decision rule
  covers it, and the readout was added at 03:47Z after the pattern was visible in the replication batch and in 9 of the 12
  primary runs (not 4, as PREREG's 03:47Z line first said; corrected there). It is large, holds in all four rule-110 bridge
  cells of both batches, and is a natural target for a pre-registered confirmation.
- **VoI and confidence-first chains do not improve bridges; stiffness-first chains help a little on B2 only.** B1 mixture KL
  stays at 0.09-0.12 for M = 0..16 (T1) for all three. On B2, stiffness-first chains with the mixture beat one pass with
  paired CIs above 0 in both arms (T1 M = 8 cut 0.124, 0.0685 -> 0.0604; T2 M = 16 cut 0.171), a third to a half of random
  order's gain (audit M2; one of 100 uncorrected selector x pool x M comparisons per family, but present in both arms). A stiff cell is one the evidence nearly determines, so materialising it adds almost nothing to
  what the network already knew; the budget is spent re-deriving the neighbourhood of the evidence (this is an
  interpretation consistent with the curves; chain trajectories were not logged).
- **Compute.** P0 costs 1 forward pass per query. At M = 16 with K = 16 chains, Stiff, Conf and Rand cost 272 passes per
  query and VoI 3,344 (13 passes per step). The one search that helps costs 144 passes per query at M = 8 (272 at M = 16) for about half the bridge KL.

IMG: results/kl-vs-compute-r110-T1.png

## 5. Stiffness (D1)

IMG: results/d1-stiffness-scatter.png

- beta_int (predicted held-out surprise, floored) ranks 2,000 held-out (cell, evidence) pairs by exact information gain
  ln 2 - H_exact(X_i | x_S) with Spearman 0.948 (rule 110 T1) and 0.923 (rule 90 T1); the expected-entropy version
  (500 pairs x 8 evidence draws) gives 0.95 and 0.94.
- **The head adds no ranking beyond the network's own confidence:** ln 2 - h(q(x_i | x_S)), the information gain the
  network itself claims, has the same Spearman (0.949 and 0.934). Mechanism A recovers calibration, not a new signal.
- Blanket AUC (S holds the cell's 3 past parents, or a child with its other 2 parents): 0.871 (rule 110 T1), 0.848
  (rule 90 T1). Past parents alone: 0.80 and 0.76. **The 0.85 bar is near what this label allows:** the exact information
  gain itself, used as the score, reaches only 0.87 (rule 110) and 0.89 (rule 90), because 43-49% of pairs WITHOUT a local
  blanket are also nearly determined (exact H < 0.1 nats) through longer chains of evidence (`results/raw/main2/*/d1-pairs.csv`;
  secondary readout added 03:50Z). This is not a strict ceiling (beta's AUC exceeds it in 3 of 6 rule-110 runs). Rule 90's
  beta is 0.03-0.04 below it. The label, as pre-registered, also counts a child triple even when the rule cannot pass the
  cell's value through it (for rule 90 the child directly below ignores the cell); such triples are the only route to a
  positive for 6.3% of rule-90 and 11.5% of rule-110 positives. With a rule-aware label rule-90 T1's AUC is 0.8502 instead
  of 0.8476, so its pre-registered FAIL is by a hair (audit M4).
- On average 2-9% of pairs get beta_int = 0 (the floor), i.e. the head predicts surprise at or above ln 2.

## 6. Dedup (D3)

- **D3a (redundant views).** For a cell with its 3 past parents known plus 3 far-past distractors (exactly redundant given
  the parents), pooling m = 5 views with scalar weights moves the pooled logit by 160 (rule 110 T1; the claims are confident
  and weights add up), Gram-crowded pooling by 6.8: ratio 0.043 (m = 2: 0.133). Redundant views get near-identical keys
  (mean cosine 0.96-0.99), so crowding divides each weight by about m. Passes in all four cells, with margin (ratios
  0.017-0.049 at m = 5, 0.051-0.153 at m = 2).
- **D3b (forward + backward chain on bridges).** Passes in KL and log loss on B1 and B2 for rule 110 (e.g. T1 B1: 0.093
  vs 0.125 forward / 0.134 backward) and for rule-90 T1; fails for rule-90 T2 on B2 (0.058 vs 0.041 forward). The keys of
  the two chains are 0.83-0.97 similar (Sec. 3), so this is the gain of averaging two different chains, not of adding
  complementary evidence.

## 7. Aggregators (P6) on the same claims

At M = 16 on the declared P2 (VoI) chains (rule 110 T1, test KL; the pattern below holds for r110-T1 only: in r110-T2 the
mixture is lowest on all four families, audit L4): mixture 0.085 / 0.119 / 0.112 / 0.037 (F / B1 / B2 / R),
plain gamma = 0 (mean of logits) 0.143 / 0.140 / 0.167 / 0.092, learned-w lw 0.183 / 0.100 / 0.092 / 0.112, dedup dd
0.191 / 0.103 / 0.095 / 0.121, gamma = 1 product 2.83 / 0.90 / 1.12 / 1.61. The product, which treats every chain as
independent evidence, is badly over-confident. The mixture wins where claims are soft (F, R); the learned pools win where
claims are stiff (B1, B2). Full table per row in Sec. 13.

## 8. What this means for the thesis

- On this domain the thesis's premise fails first: a 4-layer any-order network trained on 50k grids already gets the
  multi-step inference right in one pass for rule 90, and for rule-110 forward prediction. Caveat (audit L6): the support is
  small, about 82% of test grids also occur verbatim in the training set (55% of grids have no noise flip at all), so this
  design cannot separate "infers in one pass" from "fits a small, heavily sampled support". The only real amortisation gap is on
  rule-110 bridge queries.
- Where the gap exists, more compute does buy better beliefs, but through plain random-order Monte Carlo (ancestral
  sampling of intermediate cells, Rao-Blackwellised on the query), not through stiffness-guided routing or VoI.
- The stiffness head is a well-calibrated confidence readout. Used as a log-linear pool weight it is harmful; used to pick
  which cell to materialise next it picks cells that carry no new information.
- Key fingerprints deduplicate redundant views well, but do not separate forward from backward evidence; two of the six
  passing D3b family comparisons have paired CIs that include 0 (the rule uses 3-seed means).

## 9. Secondary results: the replication batch and the rule-90 T = 8 hedge

**Replication (the original 01:35Z runs; float32 pooling, probabilities capped to [6e-8, 1 - 6e-8], PREREG Sec. 12).**
Same seeds and configuration as the primary runs; GPU nondeterminism makes them a second draw of every training run.
Statistics that are non-finite in their result.json (the float32 bug) are "not evaluable".

| rule, arm | D0 gate (val P0 KL F / B1 / B2; S1) | D1 (Spearman; AUC) | D2 (declared P2; cut vs P0 at M=16, F / B1 / B2) | D3a (drift ratio m=5; m=2) | D3b (KL both vs best single, B1; B2) | D3 |
|---|---|---|---|---|---|---|
| r110-T1 | **FAIL** (0.027 / 0.079 / 0.055; 0.008) | **PASS** (0.948; 0.871) | **FAIL** (voi; -5.92 / -0.07 / -0.22) | **PASS** (0.044; 0.132) | **PASS** (0.093 vs 0.121; 0.107 vs 0.129) | **PASS** |
| r110-T2 | **FAIL** (0.020 / 0.080 / 0.057; 0.008) | **PASS** (0.949; 0.876) | **FAIL** (voi; -7.88 / -0.12 / -0.30) | **PASS** (0.050; 0.157) | **PASS** (0.107 vs 0.134; 0.123 vs 0.142) | **PASS** |
| r90-T1 | **FAIL** (0.003 / 0.007 / 0.005; 0.005) | **FAIL** (0.928; 0.849) | **FAIL** (voi; -7.94 / -4.94 / -9.68) | **PASS** (0.017; 0.050) | **NOT EVALUABLE** (0.055 vs 0.105; nan vs nan) | **NOT EVALUABLE** |
| r90-T2 | **FAIL** (0.005 / 0.054 / 0.037; 0.045) | **FAIL** (0.899; 0.848) | **FAIL** (voi; -14.36 / -2.33 / -2.64) | **PASS** (0.028; 0.078) | **NOT EVALUABLE** (0.108 vs 0.152; 0.089 vs nan) | **NOT EVALUABLE** |

The replication reproduces every primary verdict that it can evaluate, except rule-90 T2's D1 (AUC 0.848 here vs 0.857 in
the primary runs: the 0.85 bar sits inside the run-to-run spread). D3b is not evaluable for rule 90 here because of the
float32 bug.

**Rule 90 at T = 8 (secondary hedge, PREREG Sec. 11; also float32, capped).** A deeper grid does not create a gap for
rule 90: one pass stays near-exact (validation KL 0.001-0.011 on F, B1 and B2 for T1), so D2 has nothing to win; D1's AUC
drops to 0.82.

| rule, arm | D0 gate (val P0 KL F / B1 / B2; S1) | D1 (Spearman; AUC) | D2 (declared P2; cut vs P0 at M=16, F / B1 / B2) | D3a (drift ratio m=5; m=2) | D3b (KL both vs best single, B1; B2) | D3 |
|---|---|---|---|---|---|---|
| r90-T1 | **FAIL** (0.001 / 0.011 / 0.006; 0.006) | **FAIL** (0.904; 0.824) | **FAIL** (stiff; -0.23 / -6.80 / -8.97) | **PASS** (0.023; 0.070) | **NOT EVALUABLE** (0.027 vs 0.031; nan vs nan) | **NOT EVALUABLE** |
| r90-T2 | **FAIL** (0.002 / 0.009 / 0.033; 0.028) | **FAIL** (0.884; 0.823) | **FAIL** (stiff; -1.10 / -12.30 / -2.68) | **PASS** (0.032; 0.090) | **NOT EVALUABLE** (0.026 vs 0.040; nan vs nan) | **NOT EVALUABLE** |

Full tables: `results/replication/tables.md`, `results/t8/tables.md`; figures `results/replication/*.png`, `results/t8/*.png`.

## 10. Deviations from the design (full log with times: PREREG.md Sec. 9-12)

- **Design v2 applied in full before any training run** (00:59-01:08Z): sequential materialisation chains, eps 0.015, reveal
  count k ~ U{1..47}, families F/B1/B2/R, attention-source key head, stiffness floor, rule 30 dropped, decision rules D0-D3.
- **Key-head training bags without topology.** v2 asks for "same-blanket" redundant views; knowing the blanket would hand
  the CA topology to training, so redundant views share a random core set instead. Topology enters only the D3a evaluation.
- **cal split.** 5,000 extra training-distribution grids, held out from the main loss, fit the stiffness and key heads, so
  the stiffness head learns held-out surprise.
- **P1 folded into P4-Rand + mix.** v2 replaced one-cell contraction with materialising chains; random-order chains averaged
  by the plain mixture are the P1 arm (SNIS weights are 1 because proposal and target are the same network chain rule).
- **K = 16 chains**, fixed at 01:15Z from a first-principles Monte Carlo noise argument, before any trained model existed.
- **Rule-90 T = 8 hedge** (secondary, Sec. 9 here), launched 01:36Z because the smoke's rule-90 models were already close to
  exact at T = 6.
- **Float32 evaluation bug and the eca-b rerun** (PREREG Sec. 12). The 01:35Z code pooled claims in float32, so confident
  pooled probabilities rounded to exactly 1.0 (infinite KL, NaN log loss). Fixed at 02:31Z (float64 everywhere, training
  untouched); all 12 main runs were rerun with the identical configuration as eca-b (02:33Z). The primary-source rule was
  declared at 02:35Z, after reading one finished run (r90-T1-s2) and before any other result. All 12 eca-b runs came home,
  so every primary number below is from eca-b; the original runs are the replication (Sec. 9).
- **MATS cluster not used** (opus-mats, 01:0xZ: 8/8 L40s allocated until 11-19). Zero jobs submitted.
- **Infrastructure only:** on-demand L4s were stocked out in all three us-central1 zones at 01:36-01:39Z; launches retried
  through an eca-local copy of gpu-up/gpu-run with more zones (`gpu/gpu-up-zones.sh`, `gpu/gpu-run-zones.sh`). Every VM
  still landed in us-central1. Seven original-batch and T = 8 rentals lost the `END` line in their launch logs because
  `gpu/launch.sh` was edited while they ran; their VMs were deleted normally (GPU-LOG.md DELETE lines).
- **Training length.** Every run stopped on patience (10 evaluations without a better validation outcome log loss); the
  140-minute cap never bound.
- **D1 pair construction.** S is drawn with k ~ U{1..47} and the target is then removed, so about half the pairs have
  k - 1 reveals (one pair has an empty S). Validation ran every 1,000 steps (PREREG Sec. 10), not every 500 (Sec. 3).
- **Brute-force check.** The auditor's rerun gives max |difference| 5.3e-14 (this report first quoted 3.8e-15 from the
  01:00Z run); both are float64 round-off.
- **Stretch goal not run.** Candidate 2 (modular-arithmetic constraint nets) needs a categorical model and new exact
  inference; not attempted tonight.

## 11. Cost and clean-up

- GPU: **$19.90** at the on-demand ceiling rate ($0.87/h per g2-standard-8 + L4; Cloud Billing API disabled, so this is
  an upper bound) for 21 VMs: the priced smoke, 12 original runs on 6 VMs, the rule-90 T = 8 hedge on 3 VMs and the
  12-run eca-b rerun on 6 VMs. Per-VM ledger: `GPU-LOG.md` (rebuilt from `logical-soup/GPU-LOG.md` by `gpu/ledger.py`).
  Group cap $110 (overnight-ca); `guard-eca.sh` (4.2 h age kill, $105 stop) never fired.
- Live eca VMs at the final check: none. MATS cluster: no jobs submitted (cluster full).
- Primary runs from eca-b: 12 of 12 (the rest fall back to the original runs, capped, per PREREG Sec. 12).

## 12. Files

- Code: `ecadata.py` (CA simulator, fixed permutation, exact forward-backward inference, brute-force check, query families),
  `ecanet.py` (network + stiffness and key heads; smallnet's `netdef.Block` unchanged; edited for round 2 at
  04:26Z, the as-run round-1 file is `results/raw/ecanet-as-run-r1.py`), `ecarun.py` (one run: training,
  heads, P0/D0, D1, chains, pools, D3; float64 version, the as-run 01:35Z version is `results/raw/ecarun-as-run-0135Z.py`),
  `analyze.py` (verdicts, tables, figures), `build_results.py` (this file), `render.sh`, `gpu/` (launch, driver, remote,
  ledger, zone-flexible copies of the shared gpu-up/gpu-run).
- Primary per-run outputs: `results/raw/main2/r{rule}-{arm}-s{seed}/` with `result.json` (all metrics, pool scales,
  training history summary), `test-estimates.csv` (per-instance pooled logits for every selector x pool x M, P0, the exact
  posterior and the outcome), `d1-pairs.csv` (the 2,000 stiffness pairs), `train-history.csv`.
- Replication: `results/raw/main/`; T = 8 hedge: `results/raw/t8/`.
- Derived: `results/summary.json`, `results/tables.md`, `results/verdicts.md`, figures `results/*.png`; the same for
  `results/replication/` and `results/t8/`.
- Pre-registration and change log: `PREREG.md`. Audit: `AUDIT.md`. GPU ledger: `GPU-LOG.md`.

## 14. Audit and post-audit corrections

The independent audit (`AUDIT.md`, Opus 5.5, separate pass, 04:27Z) re-derived every pre-registered verdict and every number
in `results/summary.json` from the raw files with its own code (0 differences above 1e-6 relative), confirmed the float32
bug story, the identical training code, the fixed permutation, the stop rules and the GPU total, and found no leak of
exact posteriors or rule knowledge into training or selection. It flagged four prose overstatements (M1-M4) and thirteen
minor items (L1-L13). This report was corrected after the audit (04:35Z): the P1 headline is labelled as a secondary,
post hoc readout (M1); stiffness-first chains' smaller gain on rule-110 B2 is stated (M2); "every search arm makes things
worse" is replaced by the measured small-M gains (M3); the D1 label caveat and the "ceiling" wording are fixed (M4); and
L1-L8, L12-L13 are folded into Sec. 3, 7, 8, 10 and 12. No verdict changed. The audit's line numbers refer to the 04:10Z
version of this file.

## 13. Full tables (primary runs; 3-seed means with 95% t-intervals)


### r110-T1 (seeds [0, 1, 2]; float32-capped seeds: none)
D0 gate (validation P0 KL, nats; 3-seed mean [95% CI]): F 0.0252 [0.0104, 0.0400]; B1 0.0804 [0.0662, 0.0945]; B2 0.0585 [0.0489, 0.0680]; R 0.0212 [0.0198, 0.0227]; S1 0.0077 [0.0058, 0.0095] -> **FAIL**
D1: Spearman 0.9484 [0.9413, 0.9556], AUC 0.8708 [0.8663, 0.8752] -> **PASS** (strict PASS); expected-entropy Spearman 0.9521 [0.9345, 0.9698]; AUC parents-only 0.8040 [0.7822, 0.8257]; oracle AUC of the exact information gain 0.8705 [0.8602, 0.8808]; Spearman(-s_hat) 0.9483 [0.9412, 0.9555]; Spearman(ln2 - h(q)) 0.9487 [0.9380, 0.9595]; frac beta = 0 0.0382 [-0.0090, 0.0854]
D2 (declared P2 = voi, val log loss {"stiff": 0.3018, "voi": 0.2878}) -> **FAIL**
| family | KL P0 | KL P2 M=0 | KL P2 M=16 | KL P4-Rand M=16 | cut vs P0 | cut vs P4 | cut vs P2 M=0 | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.0245 [0.0092, 0.0398] | 0.2128 [0.1401, 0.2855] | 0.1833 [0.1687, 0.1979] | 0.0887 [0.0813, 0.0961] | -6.915 [-13.363, -0.467] | -1.068 [-1.386, -0.751] | 0.124 [-0.263, 0.512] | FAIL | FAIL |
| B1 | 0.0963 [0.0885, 0.1041] | 0.1407 [0.1061, 0.1753] | 0.0995 [0.0789, 0.1201] | 0.0566 [0.0488, 0.0644] | -0.033 [-0.236, 0.169] | -0.757 [-1.002, -0.511] | 0.292 [0.211, 0.372] | FAIL | FAIL |
| B2 | 0.0685 [0.0460, 0.0911] | 0.1091 [0.0946, 0.1235] | 0.0921 [0.0618, 0.1224] | 0.0501 [0.0468, 0.0534] | -0.347 [-0.570, -0.124] | -0.838 [-1.388, -0.287] | 0.158 [-0.006, 0.322] | FAIL | FAIL |
D3a: drift ratio m=2 0.1329 [0.1011, 0.1647], m=5 0.0433 [0.0308, 0.0559] (scalar drift m=5 159.5986 [109.2080, 209.9892], dedup 6.8488 [6.5443, 7.1533]; mean cos between redundant keys 0.9927 [0.9848, 1.0006]) -> **PASS**
D3b B1: KL both 0.0933 [0.0726, 0.1140] vs fwd 0.1250 [0.0843, 0.1657] / bwd 0.1345 [0.1200, 0.1489]; log loss both 0.1771 [0.1679, 0.1864] vs fwd 0.2101 [0.1711, 0.2491] / bwd 0.2151 [0.1903, 0.2398]; cos(k_fwd, k_bwd) 0.9500 [0.8489, 1.0511] -> PASS
D3b B2: KL both 0.1028 [0.0769, 0.1287] vs fwd 0.1341 [0.1143, 0.1539] / bwd 0.1605 [0.1217, 0.1993]; log loss both 0.1932 [0.1641, 0.2223] vs fwd 0.2256 [0.1971, 0.2541] / bwd 0.2498 [0.1790, 0.3206]; cos(k_fwd, k_bwd) 0.9455 [0.8604, 1.0306] -> PASS
D3 overall -> **PASS**

P1 arm (random-order chains + plain mixture) and the other selectors under the mixture: test KL and cut vs P0 (3-seed mean [95% CI]; a positive cut = better than P0):

| family | selector | KL M=2 | cut M=2 | KL M=8 | cut M=8 | KL M=16 | cut M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0223 | 0.081 [-0.042, 0.204] | 0.0247 | -0.064 [-0.871, 0.743] | 0.0489 | -1.140 [-3.318, 1.037] |
| F | voi | 0.0191 | 0.197 [-0.187, 0.582] | 0.0484 | -1.093 [-2.884, 0.698] | 0.0852 | -2.646 [-5.189, -0.102] |
| F | conf | 0.0227 | 0.066 [-0.027, 0.159] | 0.0252 | -0.082 [-0.881, 0.716] | 0.0533 | -1.299 [-3.128, 0.531] |
| F | rand | 0.0169 | 0.281 [-0.161, 0.724] | 0.0326 | -0.416 [-1.728, 0.896] | 0.0494 | -1.146 [-3.093, 0.801] |
| B1 | stiff | 0.0951 | 0.012 [-0.053, 0.077] | 0.0914 | 0.051 [-0.038, 0.139] | 0.0965 | -0.003 [-0.163, 0.157] |
| B1 | voi | 0.0946 | 0.017 [-0.071, 0.105] | 0.1058 | -0.099 [-0.207, 0.009] | 0.1190 | -0.237 [-0.533, 0.060] |
| B1 | conf | 0.0961 | 0.001 [-0.084, 0.087] | 0.0939 | 0.023 [-0.125, 0.171] | 0.1039 | -0.080 [-0.307, 0.147] |
| B1 | rand | 0.0702 | 0.271 [0.231, 0.311] | 0.0470 | 0.512 [0.465, 0.560] | 0.0481 | 0.501 [0.441, 0.560] |
| B2 | stiff | 0.0662 | 0.034 [-0.052, 0.121] | 0.0604 | 0.124 [-0.017, 0.265] | 0.0723 | -0.047 [-0.446, 0.352] |
| B2 | voi | 0.0700 | -0.025 [-0.155, 0.105] | 0.0745 | -0.086 [-0.260, 0.088] | 0.1122 | -0.642 [-0.804, -0.479] |
| B2 | conf | 0.0668 | 0.026 [-0.047, 0.098] | 0.0664 | 0.032 [-0.010, 0.073] | 0.0899 | -0.315 [-0.395, -0.234] |
| B2 | rand | 0.0509 | 0.253 [0.108, 0.399] | 0.0387 | 0.432 [0.332, 0.532] | 0.0399 | 0.411 [0.218, 0.603] |
| R | stiff | 0.0183 | -0.001 [-0.131, 0.130] | 0.0204 | -0.120 [-0.355, 0.116] | 0.0230 | -0.256 [-0.520, 0.008] |
| R | voi | 0.0223 | -0.225 [-0.465, 0.016] | 0.0286 | -0.563 [-0.834, -0.291] | 0.0365 | -0.997 [-1.616, -0.379] |
| R | conf | 0.0189 | -0.035 [-0.159, 0.089] | 0.0217 | -0.192 [-0.445, 0.061] | 0.0271 | -0.486 [-0.535, -0.437] |
| R | rand | 0.0159 | 0.126 [0.073, 0.180] | 0.0166 | 0.090 [-0.151, 0.331] | 0.0204 | -0.122 [-0.566, 0.323] |
Secondary (base design rules; KL differences, negative = VoI better): B1|M8: VoI-Conf -0.0216 [-0.0249, -0.0183], VoI-Rand 0.0578 [0.0495, 0.0660], VoI cut vs P0 -0.363 [-0.510, -0.215]; B1|M16: VoI-Conf -0.0317 [-0.0547, -0.0086], VoI-Rand 0.0429 [0.0273, 0.0584], VoI cut vs P0 -0.033 [-0.236, 0.169]; B2|M8: VoI-Conf 0.0015 [-0.0221, 0.0251], VoI-Rand 0.0540 [0.0099, 0.0982], VoI cut vs P0 -0.706 [-1.171, -0.242]; B2|M16: VoI-Conf -0.0353 [-0.0562, -0.0145], VoI-Rand 0.0420 [0.0133, 0.0708], VoI cut vs P0 -0.347 [-0.570, -0.124]

KL vs M (lw pool; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff | 0.2128 | 0.2131 | 0.2122 | 0.2067 | 0.2047 | 0.2105 | 0.0245 |
| F | voi | 0.2128 | 0.2161 | 0.2123 | 0.2128 | 0.2067 | 0.1833 | 0.0245 |
| F | conf | 0.2128 | 0.2119 | 0.2116 | 0.2066 | 0.2073 | 0.2211 | 0.0245 |
| F | rand | 0.2128 | 0.2031 | 0.1948 | 0.1717 | 0.1276 | 0.0887 | 0.0245 |
| B1 | stiff | 0.1407 | 0.1434 | 0.1449 | 0.1492 | 0.1565 | 0.1191 | 0.0963 |
| B1 | voi | 0.1407 | 0.1370 | 0.1453 | 0.1493 | 0.1313 | 0.0995 | 0.0963 |
| B1 | conf | 0.1407 | 0.1446 | 0.1444 | 0.1508 | 0.1529 | 0.1312 | 0.0963 |
| B1 | rand | 0.1407 | 0.1193 | 0.1089 | 0.0927 | 0.0735 | 0.0566 | 0.0963 |
| B2 | stiff | 0.1091 | 0.1063 | 0.1053 | 0.1058 | 0.1108 | 0.1103 | 0.0685 |
| B2 | voi | 0.1091 | 0.1221 | 0.1255 | 0.1236 | 0.1166 | 0.0921 | 0.0685 |
| B2 | conf | 0.1091 | 0.1077 | 0.1077 | 0.1135 | 0.1151 | 0.1274 | 0.0685 |
| B2 | rand | 0.1091 | 0.0844 | 0.0748 | 0.0736 | 0.0625 | 0.0501 | 0.0685 |
| R | stiff | 0.0529 | 0.0507 | 0.0516 | 0.0524 | 0.0790 | 0.0948 | 0.0183 |
| R | voi | 0.0529 | 0.0539 | 0.0747 | 0.0933 | 0.1128 | 0.1124 | 0.0183 |
| R | conf | 0.0529 | 0.0512 | 0.0537 | 0.0598 | 0.0938 | 0.1039 | 0.0183 |
| R | rand | 0.0529 | 0.0459 | 0.0493 | 0.0564 | 0.0725 | 0.0896 | 0.0183 |

KL vs M, mix aggregator (plain Monte Carlo average of the K = 16 chain claims; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0245 | 0.0231 | 0.0223 | 0.0220 | 0.0247 | 0.0489 |
| F | voi | 0.0245 | 0.0205 | 0.0191 | 0.0252 | 0.0484 | 0.0852 |
| F | conf | 0.0245 | 0.0233 | 0.0227 | 0.0222 | 0.0252 | 0.0533 |
| F | rand | 0.0245 | 0.0188 | 0.0169 | 0.0198 | 0.0326 | 0.0494 |
| B1 | stiff | 0.0963 | 0.0960 | 0.0951 | 0.0951 | 0.0914 | 0.0965 |
| B1 | voi | 0.0963 | 0.0926 | 0.0946 | 0.1002 | 0.1058 | 0.1190 |
| B1 | conf | 0.0963 | 0.0968 | 0.0961 | 0.0951 | 0.0939 | 0.1039 |
| B1 | rand | 0.0963 | 0.0799 | 0.0702 | 0.0563 | 0.0470 | 0.0481 |
| B2 | stiff | 0.0685 | 0.0680 | 0.0662 | 0.0611 | 0.0604 | 0.0723 |
| B2 | voi | 0.0685 | 0.0666 | 0.0700 | 0.0677 | 0.0745 | 0.1122 |
| B2 | conf | 0.0685 | 0.0694 | 0.0668 | 0.0656 | 0.0664 | 0.0899 |
| B2 | rand | 0.0685 | 0.0583 | 0.0509 | 0.0419 | 0.0387 | 0.0399 |
| R | stiff | 0.0183 | 0.0177 | 0.0183 | 0.0188 | 0.0204 | 0.0230 |
| R | voi | 0.0183 | 0.0193 | 0.0223 | 0.0247 | 0.0286 | 0.0365 |
| R | conf | 0.0183 | 0.0178 | 0.0189 | 0.0202 | 0.0217 | 0.0271 |
| R | rand | 0.0183 | 0.0164 | 0.0159 | 0.0155 | 0.0166 | 0.0204 |

Fraction of test queries whose lw pool at M = 0 returns exactly the network prior p0 (beta_int(Y | E) = 0): F 0.009 [-0.005, 0.023]; B1 0.015 [0.007, 0.024]; B2 0.019 [0.009, 0.029]; R 0.062 [0.032, 0.092]

P6 aggregators at M=16 on the declared P2 claims (test KL / log loss, 3-seed mean):

| family | mix (SNIS, unit w) | g0 plain | lw learned-w | dd (P5) | g1 product |
|---|---|---|---|---|---|
| F | 0.0852 / 0.4791 | 0.1426 / 0.5447 | 0.1833 / 0.5412 | 0.1910 / 0.5483 | 2.8300 / 3.5371|
| B1 | 0.1190 / 0.2315 | 0.1400 / 0.2615 | 0.0995 / 0.1931 | 0.1034 / 0.1946 | 0.9038 / 1.2117|
| B2 | 0.1122 / 0.1921 | 0.1669 / 0.2387 | 0.0921 / 0.1751 | 0.0948 / 0.1834 | 1.1189 / 1.1197|
| R | 0.0365 / 0.2011 | 0.0921 / 0.2476 | 0.1124 / 0.2731 | 0.1207 / 0.2809 | 1.6111 / 1.6861|

### r110-T2 (seeds [0, 1, 2]; float32-capped seeds: none)
D0 gate (validation P0 KL, nats; 3-seed mean [95% CI]): F 0.0196 [0.0032, 0.0361]; B1 0.0769 [0.0509, 0.1028]; B2 0.0478 [0.0330, 0.0626]; R 0.0192 [0.0137, 0.0248]; S1 0.0077 [0.0056, 0.0099] -> **FAIL**
D1: Spearman 0.9454 [0.9316, 0.9593], AUC 0.8692 [0.8592, 0.8792] -> **PASS** (strict PASS); expected-entropy Spearman 0.9503 [0.9381, 0.9626]; AUC parents-only 0.8031 [0.7634, 0.8428]; oracle AUC of the exact information gain 0.8705 [0.8602, 0.8808]; Spearman(-s_hat) 0.9454 [0.9313, 0.9595]; Spearman(ln2 - h(q)) 0.9459 [0.9417, 0.9502]; frac beta = 0 0.0238 [-0.0313, 0.0790]
D2 (declared P2 = voi, val log loss {"stiff": 0.3034, "voi": 0.2852}) -> **FAIL**
| family | KL P0 | KL P2 M=0 | KL P2 M=16 | KL P4-Rand M=16 | cut vs P0 | cut vs P4 | cut vs P2 M=0 | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.0186 [0.0030, 0.0342] | 0.2004 [0.1426, 0.2582] | 0.1683 [0.1560, 0.1806] | 0.0914 [0.0781, 0.1046] | -8.760 [-17.063, -0.457] | -0.846 [-1.096, -0.596] | 0.152 [-0.111, 0.416] | FAIL | FAIL |
| B1 | 0.0881 [0.0509, 0.1252] | 0.1436 [0.0993, 0.1879] | 0.1051 [0.0822, 0.1280] | 0.0667 [0.0334, 0.0999] | -0.206 [-0.497, 0.085] | -0.606 [-1.171, -0.040] | 0.263 [0.069, 0.458] | FAIL | FAIL |
| B2 | 0.0576 [0.0465, 0.0686] | 0.0960 [0.0836, 0.1085] | 0.0837 [0.0821, 0.0853] | 0.0573 [0.0465, 0.0680] | -0.459 [-0.751, -0.168] | -0.467 [-0.750, -0.184] | 0.127 [-0.004, 0.257] | FAIL | FAIL |
D3a: drift ratio m=2 0.1534 [0.1121, 0.1947], m=5 0.0492 [0.0418, 0.0567] (scalar drift m=5 105.9231 [97.4192, 114.4270], dedup 5.2079 [4.6710, 5.7448]; mean cos between redundant keys 0.9643 [0.8617, 1.0668]) -> **PASS**
D3b B1: KL both 0.1045 [0.0655, 0.1434] vs fwd 0.1387 [0.0870, 0.1905] / bwd 0.1254 [0.1052, 0.1457]; log loss both 0.1926 [0.1476, 0.2377] vs fwd 0.2312 [0.1622, 0.3002] / bwd 0.2098 [0.1855, 0.2342]; cos(k_fwd, k_bwd) 0.8716 [0.8276, 0.9155] -> PASS
D3b B2: KL both 0.1174 [0.1081, 0.1268] vs fwd 0.1349 [0.1287, 0.1410] / bwd 0.1604 [0.1542, 0.1667]; log loss both 0.1972 [0.1944, 0.1999] vs fwd 0.2193 [0.2087, 0.2298] / bwd 0.2303 [0.2087, 0.2519]; cos(k_fwd, k_bwd) 0.8340 [0.8208, 0.8472] -> PASS
D3 overall -> **PASS**

P1 arm (random-order chains + plain mixture) and the other selectors under the mixture: test KL and cut vs P0 (3-seed mean [95% CI]; a positive cut = better than P0):

| family | selector | KL M=2 | cut M=2 | KL M=8 | cut M=8 | KL M=16 | cut M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0174 | 0.051 [-0.172, 0.275] | 0.0186 | -0.030 [-0.413, 0.353] | 0.0409 | -1.321 [-2.694, 0.053] |
| F | voi | 0.0167 | 0.074 [-0.235, 0.384] | 0.0390 | -1.243 [-2.894, 0.407] | 0.0683 | -2.969 [-6.372, 0.434] |
| F | conf | 0.0174 | 0.047 [-0.208, 0.302] | 0.0201 | -0.115 [-0.550, 0.319] | 0.0418 | -1.394 [-3.047, 0.259] |
| F | rand | 0.0145 | 0.210 [0.058, 0.362] | 0.0283 | -0.626 [-1.825, 0.572] | 0.0437 | -1.537 [-3.677, 0.602] |
| B1 | stiff | 0.0883 | -0.004 [-0.057, 0.049] | 0.0833 | 0.054 [-0.130, 0.238] | 0.0806 | 0.083 [-0.119, 0.285] |
| B1 | voi | 0.0861 | 0.025 [-0.032, 0.081] | 0.0877 | 0.002 [-0.037, 0.042] | 0.0947 | -0.080 [-0.295, 0.135] |
| B1 | conf | 0.0890 | -0.012 [-0.085, 0.061] | 0.0862 | 0.015 [-0.250, 0.280] | 0.0930 | -0.059 [-0.473, 0.355] |
| B1 | rand | 0.0685 | 0.223 [0.200, 0.245] | 0.0506 | 0.424 [0.382, 0.465] | 0.0491 | 0.441 [0.392, 0.490] |
| B2 | stiff | 0.0544 | 0.054 [-0.023, 0.131] | 0.0497 | 0.140 [-0.033, 0.313] | 0.0478 | 0.171 [0.074, 0.268] |
| B2 | voi | 0.0559 | 0.030 [-0.060, 0.119] | 0.0565 | 0.016 [-0.171, 0.203] | 0.0738 | -0.287 [-0.533, -0.041] |
| B2 | conf | 0.0547 | 0.050 [-0.027, 0.126] | 0.0541 | 0.065 [-0.228, 0.359] | 0.0564 | 0.030 [-0.417, 0.478] |
| B2 | rand | 0.0434 | 0.248 [0.169, 0.326] | 0.0389 | 0.324 [0.233, 0.416] | 0.0414 | 0.280 [0.213, 0.348] |
| R | stiff | 0.0166 | 0.020 [-0.051, 0.091] | 0.0184 | -0.093 [-0.413, 0.226] | 0.0234 | -0.377 [-0.534, -0.220] |
| R | voi | 0.0192 | -0.136 [-0.269, -0.004] | 0.0249 | -0.481 [-0.944, -0.018] | 0.0287 | -0.703 [-1.145, -0.262] |
| R | conf | 0.0170 | -0.006 [-0.067, 0.054] | 0.0203 | -0.204 [-0.433, 0.026] | 0.0246 | -0.454 [-0.631, -0.278] |
| R | rand | 0.0153 | 0.092 [-0.073, 0.256] | 0.0183 | -0.086 [-0.423, 0.252] | 0.0209 | -0.239 [-0.623, 0.146] |
Secondary (base design rules; KL differences, negative = VoI better): B1|M8: VoI-Conf -0.0492 [-0.0774, -0.0211], VoI-Rand 0.0477 [0.0297, 0.0658], VoI cut vs P0 -0.553 [-0.618, -0.489]; B1|M16: VoI-Conf -0.0395 [-0.0831, 0.0042], VoI-Rand 0.0384 [0.0216, 0.0552], VoI cut vs P0 -0.206 [-0.497, 0.085]; B2|M8: VoI-Conf -0.0160 [-0.0845, 0.0525], VoI-Rand 0.0248 [-0.0058, 0.0555], VoI cut vs P0 -0.745 [-1.244, -0.247]; B2|M16: VoI-Conf -0.0201 [-0.0741, 0.0339], VoI-Rand 0.0264 [0.0157, 0.0371], VoI cut vs P0 -0.459 [-0.751, -0.168]

KL vs M (lw pool; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff | 0.2004 | 0.2013 | 0.2067 | 0.2108 | 0.2109 | 0.2068 | 0.0186 |
| F | voi | 0.2004 | 0.2124 | 0.2123 | 0.2133 | 0.2013 | 0.1683 | 0.0186 |
| F | conf | 0.2004 | 0.2008 | 0.2083 | 0.2094 | 0.2095 | 0.2146 | 0.0186 |
| F | rand | 0.2004 | 0.1921 | 0.1859 | 0.1660 | 0.1292 | 0.0914 | 0.0186 |
| B1 | stiff | 0.1436 | 0.1494 | 0.1532 | 0.1648 | 0.1762 | 0.1273 | 0.0881 |
| B1 | voi | 0.1436 | 0.1418 | 0.1514 | 0.1580 | 0.1370 | 0.1051 | 0.0881 |
| B1 | conf | 0.1436 | 0.1481 | 0.1508 | 0.1713 | 0.1862 | 0.1445 | 0.0881 |
| B1 | rand | 0.1436 | 0.1291 | 0.1251 | 0.1132 | 0.0892 | 0.0667 | 0.0881 |
| B2 | stiff | 0.0960 | 0.0943 | 0.0957 | 0.1027 | 0.1094 | 0.0916 | 0.0576 |
| B2 | voi | 0.0960 | 0.1017 | 0.1125 | 0.1207 | 0.0999 | 0.0837 | 0.0576 |
| B2 | conf | 0.0960 | 0.0948 | 0.0952 | 0.1075 | 0.1159 | 0.1038 | 0.0576 |
| B2 | rand | 0.0960 | 0.0839 | 0.0826 | 0.0859 | 0.0751 | 0.0573 | 0.0576 |
| R | stiff | 0.0474 | 0.0453 | 0.0485 | 0.0478 | 0.0662 | 0.0949 | 0.0169 |
| R | voi | 0.0474 | 0.0508 | 0.0547 | 0.0790 | 0.0959 | 0.1084 | 0.0169 |
| R | conf | 0.0474 | 0.0464 | 0.0522 | 0.0529 | 0.0727 | 0.1027 | 0.0169 |
| R | rand | 0.0474 | 0.0430 | 0.0445 | 0.0524 | 0.0715 | 0.0895 | 0.0169 |

KL vs M, mix aggregator (plain Monte Carlo average of the K = 16 chain claims; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0186 | 0.0178 | 0.0174 | 0.0170 | 0.0186 | 0.0409 |
| F | voi | 0.0186 | 0.0172 | 0.0167 | 0.0206 | 0.0390 | 0.0683 |
| F | conf | 0.0186 | 0.0179 | 0.0174 | 0.0174 | 0.0201 | 0.0418 |
| F | rand | 0.0186 | 0.0153 | 0.0145 | 0.0182 | 0.0283 | 0.0437 |
| B1 | stiff | 0.0881 | 0.0883 | 0.0883 | 0.0899 | 0.0833 | 0.0806 |
| B1 | voi | 0.0881 | 0.0875 | 0.0861 | 0.0867 | 0.0877 | 0.0947 |
| B1 | conf | 0.0881 | 0.0891 | 0.0890 | 0.0903 | 0.0862 | 0.0930 |
| B1 | rand | 0.0881 | 0.0758 | 0.0685 | 0.0589 | 0.0506 | 0.0491 |
| B2 | stiff | 0.0576 | 0.0551 | 0.0544 | 0.0526 | 0.0497 | 0.0478 |
| B2 | voi | 0.0576 | 0.0553 | 0.0559 | 0.0548 | 0.0565 | 0.0738 |
| B2 | conf | 0.0576 | 0.0559 | 0.0547 | 0.0543 | 0.0541 | 0.0564 |
| B2 | rand | 0.0576 | 0.0498 | 0.0434 | 0.0393 | 0.0389 | 0.0414 |
| R | stiff | 0.0169 | 0.0164 | 0.0166 | 0.0167 | 0.0184 | 0.0234 |
| R | voi | 0.0169 | 0.0182 | 0.0192 | 0.0224 | 0.0249 | 0.0287 |
| R | conf | 0.0169 | 0.0167 | 0.0170 | 0.0174 | 0.0203 | 0.0246 |
| R | rand | 0.0169 | 0.0158 | 0.0153 | 0.0156 | 0.0183 | 0.0209 |

Fraction of test queries whose lw pool at M = 0 returns exactly the network prior p0 (beta_int(Y | E) = 0): F 0.006 [-0.011, 0.024]; B1 0.013 [-0.008, 0.034]; B2 0.026 [0.009, 0.043]; R 0.040 [-0.031, 0.111]

P6 aggregators at M=16 on the declared P2 claims (test KL / log loss, 3-seed mean):

| family | mix (SNIS, unit w) | g0 plain | lw learned-w | dd (P5) | g1 product |
|---|---|---|---|---|---|
| F | 0.0683 / 0.4635 | 0.1304 / 0.5382 | 0.1683 / 0.5315 | 0.2306 / 0.5883 | 2.8191 / 3.5448|
| B1 | 0.0947 / 0.1963 | 0.1192 / 0.2244 | 0.1051 / 0.1978 | 0.1283 / 0.2166 | 0.8957 / 1.1227|
| B2 | 0.0738 / 0.1595 | 0.1193 / 0.1937 | 0.0837 / 0.1650 | 0.1023 / 0.2027 | 1.0682 / 1.0650|
| R | 0.0287 / 0.1837 | 0.0753 / 0.2225 | 0.1084 / 0.2664 | 0.1389 / 0.3044 | 1.5589 / 1.5885|

### r90-T1 (seeds [0, 1, 2]; float32-capped seeds: none)
D0 gate (validation P0 KL, nats; 3-seed mean [95% CI]): F 0.0060 [0.0009, 0.0110]; B1 0.0100 [-0.0030, 0.0231]; B2 0.0072 [0.0040, 0.0104]; R 0.0050 [0.0025, 0.0075]; S1 0.0084 [0.0005, 0.0163] -> **FAIL**
D1: Spearman 0.9232 [0.8954, 0.9511], AUC 0.8476 [0.7906, 0.9046] -> **FAIL** (strict FAIL); expected-entropy Spearman 0.9363 [0.9210, 0.9516]; AUC parents-only 0.7557 [0.6988, 0.8126]; oracle AUC of the exact information gain 0.8866 [0.8613, 0.9120]; Spearman(-s_hat) 0.9224 [0.8943, 0.9506]; Spearman(ln2 - h(q)) 0.9338 [0.9285, 0.9390]; frac beta = 0 0.0892 [-0.0881, 0.2664]
D2 (declared P2 = voi, val log loss {"stiff": 0.2409, "voi": 0.2099}) -> **FAIL**
| family | KL P0 | KL P2 M=0 | KL P2 M=16 | KL P4-Rand M=16 | cut vs P0 | cut vs P4 | cut vs P2 M=0 | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.0060 [0.0006, 0.0113] | 0.0289 [0.0239, 0.0339] | 0.0331 [-0.0022, 0.0684] | 0.0294 [0.0198, 0.0391] | -4.759 [-10.577, 1.058] | -0.095 [-0.943, 0.753] | -0.154 [-1.481, 1.173] | FAIL | FAIL |
| B1 | 0.0100 [-0.0036, 0.0237] | 0.0348 [0.0059, 0.0637] | 0.0604 [-0.0025, 0.1233] | 0.0310 [0.0298, 0.0321] | -5.414 [-8.439, -2.390] | -0.956 [-3.034, 1.122] | -0.871 [-3.617, 1.876] | FAIL | FAIL |
| B2 | 0.0085 [0.0022, 0.0148] | 0.0213 [0.0093, 0.0334] | 0.0511 [0.0437, 0.0584] | 0.0363 [0.0281, 0.0445] | -5.360 [-10.003, -0.717] | -0.410 [-0.543, -0.278] | -1.462 [-2.566, -0.357] | FAIL | FAIL |
D3a: drift ratio m=2 0.0506 [0.0274, 0.0738], m=5 0.0169 [0.0079, 0.0259] (scalar drift m=5 63.5964 [38.1686, 89.0243], dedup 1.1015 [0.0472, 2.1557]; mean cos between redundant keys 0.9911 [0.9873, 0.9948]) -> **PASS**
D3b B1: KL both 0.0494 [0.0374, 0.0614] vs fwd 0.1461 [0.0562, 0.2359] / bwd 0.0824 [0.0505, 0.1143]; log loss both 0.2258 [0.2199, 0.2318] vs fwd 0.3347 [0.2170, 0.4524] / bwd 0.2518 [0.2382, 0.2654]; cos(k_fwd, k_bwd) 0.8363 [0.7373, 0.9352] -> PASS
D3b B2: KL both 0.0329 [0.0177, 0.0481] vs fwd 0.0349 [0.0241, 0.0457] / bwd 0.0434 [0.0088, 0.0780]; log loss both 0.0722 [0.0570, 0.0873] vs fwd 0.0764 [0.0606, 0.0921] / bwd 0.0846 [0.0520, 0.1173]; cos(k_fwd, k_bwd) 0.9656 [0.9256, 1.0056] -> PASS
D3 overall -> **PASS**

P1 arm (random-order chains + plain mixture) and the other selectors under the mixture: test KL and cut vs P0 (3-seed mean [95% CI]; a positive cut = better than P0):

| family | selector | KL M=2 | cut M=2 | KL M=8 | cut M=8 | KL M=16 | cut M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0065 | -0.108 [-0.457, 0.242] | 0.0097 | -0.738 [-2.512, 1.036] | 0.0177 | -2.199 [-5.083, 0.684] |
| F | voi | 0.0054 | 0.040 [-0.739, 0.820] | 0.0192 | -2.397 [-4.618, -0.175] | 0.0363 | -5.208 [-7.636, -2.779] |
| F | conf | 0.0065 | -0.097 [-0.368, 0.173] | 0.0108 | -0.912 [-3.083, 1.259] | 0.0199 | -2.493 [-4.899, -0.086] |
| F | rand | 0.0072 | -0.285 [-1.151, 0.581] | 0.0177 | -2.187 [-4.855, 0.480] | 0.0391 | -6.031 [-11.210, -0.852] |
| B1 | stiff | 0.0113 | -0.216 [-1.006, 0.573] | 0.0205 | -1.669 [-6.014, 2.675] | 0.0361 | -3.517 [-9.878, 2.844] |
| B1 | voi | 0.0211 | -1.326 [-3.098, 0.445] | 0.0633 | -6.961 [-18.606, 4.685] | 0.0721 | -7.897 [-19.910, 4.117] |
| B1 | conf | 0.0120 | -0.293 [-1.187, 0.601] | 0.0284 | -2.749 [-9.420, 3.922] | 0.0497 | -4.959 [-12.301, 2.383] |
| B1 | rand | 0.0129 | -0.510 [-2.048, 1.028] | 0.0384 | -3.760 [-10.208, 2.688] | 0.0793 | -9.063 [-24.073, 5.947] |
| B2 | stiff | 0.0083 | 0.030 [-0.081, 0.142] | 0.0085 | 0.015 [-0.057, 0.086] | 0.0084 | 0.022 [-0.000, 0.044] |
| B2 | voi | 0.0085 | 0.015 [-0.092, 0.121] | 0.0088 | -0.031 [-0.101, 0.039] | 0.0126 | -0.469 [-0.949, 0.011] |
| B2 | conf | 0.0084 | 0.027 [-0.091, 0.144] | 0.0085 | 0.014 [-0.073, 0.101] | 0.0093 | -0.099 [-0.187, -0.011] |
| B2 | rand | 0.0087 | -0.018 [-0.073, 0.037] | 0.0098 | -0.165 [-0.362, 0.033] | 0.0112 | -0.323 [-0.460, -0.186] |
| R | stiff | 0.0033 | -0.002 [-0.172, 0.168] | 0.0049 | -0.511 [-0.951, -0.071] | 0.0096 | -2.147 [-4.723, 0.429] |
| R | voi | 0.0048 | -0.524 [-1.310, 0.262] | 0.0090 | -1.929 [-4.086, 0.229] | 0.0150 | -3.997 [-8.568, 0.574] |
| R | conf | 0.0033 | -0.005 [-0.127, 0.117] | 0.0054 | -0.657 [-1.018, -0.295] | 0.0121 | -3.054 [-6.898, 0.790] |
| R | rand | 0.0043 | -0.363 [-0.969, 0.244] | 0.0093 | -2.044 [-4.417, 0.330] | 0.0151 | -4.114 [-9.629, 1.401] |
Secondary (base design rules; KL differences, negative = VoI better): B1|M8: VoI-Conf -0.0724 [-0.2346, 0.0897], VoI-Rand 0.0624 [-0.0301, 0.1549], VoI cut vs P0 -12.815 [-24.851, -0.780]; B1|M16: VoI-Conf -0.1363 [-0.2142, -0.0584], VoI-Rand 0.0294 [-0.0344, 0.0932], VoI cut vs P0 -5.414 [-8.439, -2.390]; B2|M8: VoI-Conf 0.0282 [0.0175, 0.0389], VoI-Rand 0.0328 [0.0296, 0.0360], VoI cut vs P0 -6.648 [-12.077, -1.219]; B2|M16: VoI-Conf 0.0070 [-0.0036, 0.0176], VoI-Rand 0.0148 [0.0129, 0.0167], VoI cut vs P0 -5.360 [-10.003, -0.717]

KL vs M (lw pool; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff | 0.0289 | 0.0280 | 0.0299 | 0.0408 | 0.0751 | 0.0547 | 0.0060 |
| F | voi | 0.0289 | 0.0290 | 0.0527 | 0.0825 | 0.0903 | 0.0331 | 0.0060 |
| F | conf | 0.0289 | 0.0290 | 0.0307 | 0.0400 | 0.0732 | 0.0784 | 0.0060 |
| F | rand | 0.0289 | 0.0612 | 0.0664 | 0.0557 | 0.0239 | 0.0294 | 0.0060 |
| B1 | stiff | 0.0348 | 0.0356 | 0.0375 | 0.0686 | 0.1455 | 0.1204 | 0.0100 |
| B1 | voi | 0.0348 | 0.0461 | 0.0921 | 0.1516 | 0.1210 | 0.0604 | 0.0100 |
| B1 | conf | 0.0348 | 0.0367 | 0.0393 | 0.0746 | 0.1934 | 0.1967 | 0.0100 |
| B1 | rand | 0.0348 | 0.0590 | 0.0638 | 0.0694 | 0.0586 | 0.0310 | 0.0100 |
| B2 | stiff | 0.0213 | 0.0212 | 0.0214 | 0.0217 | 0.0342 | 0.0454 | 0.0085 |
| B2 | voi | 0.0213 | 0.0219 | 0.0236 | 0.0353 | 0.0615 | 0.0511 | 0.0085 |
| B2 | conf | 0.0213 | 0.0213 | 0.0213 | 0.0214 | 0.0333 | 0.0441 | 0.0085 |
| B2 | rand | 0.0213 | 0.0223 | 0.0234 | 0.0255 | 0.0288 | 0.0363 | 0.0085 |
| R | stiff | 0.0174 | 0.0179 | 0.0189 | 0.0278 | 0.0426 | 0.0506 | 0.0033 |
| R | voi | 0.0174 | 0.0241 | 0.0330 | 0.0512 | 0.0503 | 0.0537 | 0.0033 |
| R | conf | 0.0174 | 0.0180 | 0.0178 | 0.0233 | 0.0415 | 0.0512 | 0.0033 |
| R | rand | 0.0174 | 0.0405 | 0.0440 | 0.0479 | 0.0525 | 0.0609 | 0.0033 |

KL vs M, mix aggregator (plain Monte Carlo average of the K = 16 chain claims; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0060 | 0.0067 | 0.0065 | 0.0061 | 0.0097 | 0.0177 |
| F | voi | 0.0060 | 0.0067 | 0.0054 | 0.0094 | 0.0192 | 0.0363 |
| F | conf | 0.0060 | 0.0067 | 0.0065 | 0.0066 | 0.0108 | 0.0199 |
| F | rand | 0.0060 | 0.0063 | 0.0072 | 0.0097 | 0.0177 | 0.0391 |
| B1 | stiff | 0.0100 | 0.0112 | 0.0113 | 0.0128 | 0.0205 | 0.0361 |
| B1 | voi | 0.0100 | 0.0148 | 0.0211 | 0.0352 | 0.0633 | 0.0721 |
| B1 | conf | 0.0100 | 0.0114 | 0.0120 | 0.0153 | 0.0284 | 0.0497 |
| B1 | rand | 0.0100 | 0.0112 | 0.0129 | 0.0187 | 0.0384 | 0.0793 |
| B2 | stiff | 0.0085 | 0.0083 | 0.0083 | 0.0084 | 0.0085 | 0.0084 |
| B2 | voi | 0.0085 | 0.0084 | 0.0085 | 0.0086 | 0.0088 | 0.0126 |
| B2 | conf | 0.0085 | 0.0083 | 0.0084 | 0.0085 | 0.0085 | 0.0093 |
| B2 | rand | 0.0085 | 0.0087 | 0.0087 | 0.0090 | 0.0098 | 0.0112 |
| R | stiff | 0.0033 | 0.0034 | 0.0033 | 0.0037 | 0.0049 | 0.0096 |
| R | voi | 0.0033 | 0.0042 | 0.0048 | 0.0060 | 0.0090 | 0.0150 |
| R | conf | 0.0033 | 0.0034 | 0.0033 | 0.0037 | 0.0054 | 0.0121 |
| R | rand | 0.0033 | 0.0037 | 0.0043 | 0.0059 | 0.0093 | 0.0151 |

Fraction of test queries whose lw pool at M = 0 returns exactly the network prior p0 (beta_int(Y | E) = 0): F 0.005 [-0.018, 0.028]; B1 0.019 [-0.013, 0.052]; B2 0.013 [0.002, 0.024]; R 0.180 [0.175, 0.186]

P6 aggregators at M=16 on the declared P2 claims (test KL / log loss, 3-seed mean):

| family | mix (SNIS, unit w) | g0 plain | lw learned-w | dd (P5) | g1 product |
|---|---|---|---|---|---|
| F | 0.0363 / 0.3410 | 0.1651 / 0.4459 | 0.0331 / 0.3343 | 0.0496 / 0.3579 | 2.1828 / 2.2291|
| B1 | 0.0721 / 0.2485 | 0.1072 / 0.2917 | 0.0604 / 0.2381 | 0.0783 / 0.2584 | 0.9109 / 1.1345|
| B2 | 0.0126 / 0.0486 | 0.0176 / 0.0575 | 0.0511 / 0.0846 | 0.0605 / 0.0938 | 0.2355 / 0.2894|
| R | 0.0150 / 0.1907 | 0.0454 / 0.2304 | 0.0537 / 0.2476 | 0.0728 / 0.2665 | 0.8977 / 1.0030|

### r90-T2 (seeds [0, 1, 2]; float32-capped seeds: none)
D0 gate (validation P0 KL, nats; 3-seed mean [95% CI]): F 0.0061 [-0.0031, 0.0153]; B1 0.0525 [-0.1246, 0.2296]; B2 0.0094 [-0.0042, 0.0229]; R 0.0062 [-0.0017, 0.0141]; S1 0.0209 [-0.0164, 0.0583] -> **FAIL**
D1: Spearman 0.9249 [0.9021, 0.9477], AUC 0.8567 [0.8269, 0.8866] -> **PASS** (strict FAIL); expected-entropy Spearman 0.9290 [0.9157, 0.9423]; AUC parents-only 0.7631 [0.7316, 0.7947]; oracle AUC of the exact information gain 0.8866 [0.8613, 0.9120]; Spearman(-s_hat) 0.9244 [0.8998, 0.9490]; Spearman(ln2 - h(q)) 0.9321 [0.9099, 0.9542]; frac beta = 0 0.0775 [-0.0982, 0.2532]
D2 (declared P2 = voi, val log loss {"stiff": 0.2413, "voi": 0.2188}) -> **FAIL**
| family | KL P0 | KL P2 M=0 | KL P2 M=16 | KL P4-Rand M=16 | cut vs P0 | cut vs P4 | cut vs P2 M=0 | (a) | (b) |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.0060 [-0.0033, 0.0152] | 0.0268 [-0.0063, 0.0599] | 0.0420 [-0.0227, 0.1067] | 0.0329 [0.0121, 0.0537] | -14.273 [-65.240, 36.694] | -0.206 [-1.267, 0.856] | -0.535 [-1.281, 0.211] | FAIL | FAIL |
| B1 | 0.0536 [-0.1264, 0.2336] | 0.0686 [-0.1242, 0.2613] | 0.0763 [-0.0822, 0.2348] | 0.0451 [-0.0145, 0.1047] | -1.714 [-5.679, 2.251] | -0.528 [-1.676, 0.620] | -0.610 [-3.014, 1.795] | FAIL | FAIL |
| B2 | 0.0104 [-0.0046, 0.0255] | 0.0254 [0.0081, 0.0426] | 0.0591 [0.0414, 0.0767] | 0.0397 [0.0136, 0.0659] | -6.929 [-21.577, 7.719] | -0.536 [-1.380, 0.309] | -1.413 [-2.610, -0.216] | FAIL | FAIL |
D3a: drift ratio m=2 0.0734 [-0.0134, 0.1602], m=5 0.0243 [-0.0081, 0.0567] (scalar drift m=5 49.1414 [20.9900, 77.2927], dedup 1.1212 [0.1409, 2.1015]; mean cos between redundant keys 0.9742 [0.9058, 1.0426]) -> **PASS**
D3b B1: KL both 0.0978 [-0.1327, 0.3283] vs fwd 0.2023 [-0.0430, 0.4475] / bwd 0.1460 [-0.0045, 0.2965]; log loss both 0.2775 [0.0198, 0.5352] vs fwd 0.3925 [0.1224, 0.6626] / bwd 0.3121 [0.1342, 0.4900]; cos(k_fwd, k_bwd) 0.8274 [0.6444, 1.0103] -> PASS
D3b B2: KL both 0.0579 [-0.0136, 0.1295] vs fwd 0.0412 [0.0148, 0.0676] / bwd 0.0743 [-0.0080, 0.1566]; log loss both 0.1014 [0.0421, 0.1608] vs fwd 0.0768 [0.0607, 0.0930] / bwd 0.1263 [0.0582, 0.1944]; cos(k_fwd, k_bwd) 0.9615 [0.9473, 0.9757] -> FAIL
D3 overall -> **FAIL**

P1 arm (random-order chains + plain mixture) and the other selectors under the mixture: test KL and cut vs P0 (3-seed mean [95% CI]; a positive cut = better than P0):

| family | selector | KL M=2 | cut M=2 | KL M=8 | cut M=8 | KL M=16 | cut M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0062 | -0.255 [-1.649, 1.139] | 0.0100 | -1.744 [-8.202, 4.714] | 0.0177 | -3.576 [-13.284, 6.131] |
| F | voi | 0.0067 | -0.674 [-4.222, 2.874] | 0.0208 | -4.653 [-17.796, 8.490] | 0.0353 | -8.133 [-27.684, 11.417] |
| F | conf | 0.0066 | -0.351 [-1.917, 1.216] | 0.0112 | -2.273 [-10.703, 6.158] | 0.0200 | -4.586 [-18.163, 8.990] |
| F | rand | 0.0098 | -1.929 [-10.001, 6.143] | 0.0329 | -11.702 [-56.390, 32.986] | 0.0544 | -17.680 [-76.952, 41.592] |
| B1 | stiff | 0.0550 | -0.081 [-0.479, 0.316] | 0.0691 | -0.958 [-2.907, 0.991] | 0.0805 | -1.562 [-4.515, 1.391] |
| B1 | voi | 0.0679 | -0.540 [-1.305, 0.225] | 0.1043 | -3.206 [-9.595, 3.183] | 0.1109 | -3.572 [-10.644, 3.499] |
| B1 | conf | 0.0570 | -0.150 [-0.573, 0.273] | 0.0780 | -1.464 [-4.261, 1.332] | 0.0884 | -2.030 [-5.867, 1.807] |
| B1 | rand | 0.0527 | -0.096 [-0.547, 0.356] | 0.0742 | -1.638 [-5.204, 1.928] | 0.1015 | -3.573 [-11.273, 4.127] |
| B2 | stiff | 0.0104 | 0.000 [-0.010, 0.011] | 0.0104 | 0.011 [-0.073, 0.096] | 0.0104 | 0.032 [-0.176, 0.240] |
| B2 | voi | 0.0104 | 0.010 [-0.019, 0.039] | 0.0106 | -0.065 [-0.370, 0.239] | 0.0139 | -0.532 [-1.804, 0.740] |
| B2 | conf | 0.0104 | 0.006 [-0.024, 0.036] | 0.0104 | 0.003 [-0.006, 0.013] | 0.0106 | -0.016 [-0.057, 0.025] |
| B2 | rand | 0.0101 | 0.000 [-0.260, 0.260] | 0.0115 | -0.161 [-0.602, 0.279] | 0.0130 | -0.356 [-1.243, 0.530] |
| R | stiff | 0.0063 | -0.023 [-0.118, 0.073] | 0.0076 | -0.290 [-0.838, 0.259] | 0.0116 | -1.081 [-2.633, 0.470] |
| R | voi | 0.0073 | -0.253 [-0.882, 0.376] | 0.0112 | -1.002 [-2.466, 0.461] | 0.0157 | -1.866 [-4.383, 0.652] |
| R | conf | 0.0063 | -0.030 [-0.131, 0.072] | 0.0084 | -0.404 [-0.761, -0.046] | 0.0146 | -1.629 [-3.720, 0.462] |
| R | rand | 0.0077 | -0.287 [-0.686, 0.112] | 0.0125 | -1.228 [-3.010, 0.555] | 0.0179 | -2.246 [-5.067, 0.575] |
Secondary (base design rules; KL differences, negative = VoI better): B1|M8: VoI-Conf -0.1441 [-0.2654, -0.0228], VoI-Rand 0.0532 [-0.0182, 0.1246], VoI cut vs P0 -5.866 [-20.503, 8.770]; B1|M16: VoI-Conf -0.1424 [-0.3163, 0.0314], VoI-Rand 0.0312 [-0.0678, 0.1301], VoI cut vs P0 -1.714 [-5.679, 2.251]; B2|M8: VoI-Conf 0.0326 [0.0165, 0.0486], VoI-Rand 0.0467 [0.0064, 0.0869], VoI cut vs P0 -9.466 [-26.212, 7.279]; B2|M16: VoI-Conf 0.0004 [-0.0419, 0.0426], VoI-Rand 0.0193 [-0.0013, 0.0399], VoI cut vs P0 -6.929 [-21.577, 7.719]

KL vs M (lw pool; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff | 0.0268 | 0.0290 | 0.0318 | 0.0423 | 0.0701 | 0.0449 | 0.0060 |
| F | voi | 0.0268 | 0.0380 | 0.0635 | 0.0944 | 0.0707 | 0.0420 | 0.0060 |
| F | conf | 0.0268 | 0.0271 | 0.0323 | 0.0414 | 0.0822 | 0.0731 | 0.0060 |
| F | rand | 0.0268 | 0.0430 | 0.0535 | 0.0540 | 0.0290 | 0.0329 | 0.0060 |
| B1 | stiff | 0.0686 | 0.0813 | 0.0889 | 0.1230 | 0.1989 | 0.1540 | 0.0536 |
| B1 | voi | 0.0686 | 0.1286 | 0.1524 | 0.1693 | 0.1294 | 0.0763 | 0.0536 |
| B1 | conf | 0.0686 | 0.0973 | 0.1083 | 0.1472 | 0.2735 | 0.2187 | 0.0536 |
| B1 | rand | 0.0686 | 0.0831 | 0.0936 | 0.0943 | 0.0762 | 0.0451 | 0.0536 |
| B2 | stiff | 0.0254 | 0.0261 | 0.0268 | 0.0299 | 0.0478 | 0.0562 | 0.0104 |
| B2 | voi | 0.0254 | 0.0295 | 0.0352 | 0.0526 | 0.0820 | 0.0591 | 0.0104 |
| B2 | conf | 0.0254 | 0.0256 | 0.0263 | 0.0281 | 0.0494 | 0.0587 | 0.0104 |
| B2 | rand | 0.0254 | 0.0245 | 0.0261 | 0.0301 | 0.0353 | 0.0397 | 0.0104 |
| R | stiff | 0.0212 | 0.0218 | 0.0224 | 0.0289 | 0.0431 | 0.0536 | 0.0062 |
| R | voi | 0.0212 | 0.0254 | 0.0342 | 0.0513 | 0.0561 | 0.0554 | 0.0062 |
| R | conf | 0.0212 | 0.0218 | 0.0216 | 0.0272 | 0.0467 | 0.0566 | 0.0062 |
| R | rand | 0.0212 | 0.0506 | 0.0512 | 0.0543 | 0.0569 | 0.0631 | 0.0062 |

KL vs M, mix aggregator (plain Monte Carlo average of the K = 16 chain claims; 3-seed mean):

| family | selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff | 0.0060 | 0.0062 | 0.0062 | 0.0067 | 0.0100 | 0.0177 |
| F | voi | 0.0060 | 0.0071 | 0.0067 | 0.0094 | 0.0208 | 0.0353 |
| F | conf | 0.0060 | 0.0065 | 0.0066 | 0.0070 | 0.0112 | 0.0200 |
| F | rand | 0.0060 | 0.0070 | 0.0098 | 0.0172 | 0.0329 | 0.0544 |
| B1 | stiff | 0.0536 | 0.0542 | 0.0550 | 0.0574 | 0.0691 | 0.0805 |
| B1 | voi | 0.0536 | 0.0623 | 0.0679 | 0.0825 | 0.1043 | 0.1109 |
| B1 | conf | 0.0536 | 0.0557 | 0.0570 | 0.0611 | 0.0780 | 0.0884 |
| B1 | rand | 0.0536 | 0.0517 | 0.0527 | 0.0576 | 0.0742 | 0.1015 |
| B2 | stiff | 0.0104 | 0.0104 | 0.0104 | 0.0105 | 0.0104 | 0.0104 |
| B2 | voi | 0.0104 | 0.0108 | 0.0104 | 0.0103 | 0.0106 | 0.0139 |
| B2 | conf | 0.0104 | 0.0104 | 0.0104 | 0.0104 | 0.0104 | 0.0106 |
| B2 | rand | 0.0104 | 0.0101 | 0.0101 | 0.0108 | 0.0115 | 0.0130 |
| R | stiff | 0.0062 | 0.0063 | 0.0063 | 0.0065 | 0.0076 | 0.0116 |
| R | voi | 0.0062 | 0.0066 | 0.0073 | 0.0089 | 0.0112 | 0.0157 |
| R | conf | 0.0062 | 0.0064 | 0.0063 | 0.0068 | 0.0084 | 0.0146 |
| R | rand | 0.0062 | 0.0069 | 0.0077 | 0.0088 | 0.0125 | 0.0179 |

Fraction of test queries whose lw pool at M = 0 returns exactly the network prior p0 (beta_int(Y | E) = 0): F 0.003 [-0.009, 0.014]; B1 0.102 [-0.218, 0.422]; B2 0.010 [0.001, 0.019]; R 0.180 [0.176, 0.183]

P6 aggregators at M=16 on the declared P2 claims (test KL / log loss, 3-seed mean):

| family | mix (SNIS, unit w) | g0 plain | lw learned-w | dd (P5) | g1 product |
|---|---|---|---|---|---|
| F | 0.0353 / 0.3449 | 0.1743 / 0.4526 | 0.0420 / 0.3366 | 0.0511 / 0.3450 | 2.2147 / 2.2953|
| B1 | 0.1109 / 0.2900 | 0.1399 / 0.3296 | 0.0763 / 0.2590 | 0.0837 / 0.2653 | 0.9761 / 1.2280|
| B2 | 0.0139 / 0.0465 | 0.0162 / 0.0549 | 0.0591 / 0.0924 | 0.0665 / 0.1012 | 0.2624 / 0.3267|
| R | 0.0157 / 0.1928 | 0.0397 / 0.2223 | 0.0554 / 0.2485 | 0.0683 / 0.2573 | 0.8351 / 0.9431|

### Compute per query (forward passes; FLOPs = passes x 5.51e+07 per forward pass; K = 16 chains)

| selector | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|
| P0 | 1 | | | | | |
| stiff | 16 | 32 | 48 | 80 | 144 | 272 |
| voi | 16 | 224 | 432 | 848 | 1680 | 3344 |
| conf | 16 | 32 | 48 | 80 | 144 | 272 |
| rand | 16 | 32 | 48 | 80 | 144 | 272 |
