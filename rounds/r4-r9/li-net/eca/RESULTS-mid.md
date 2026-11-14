## 3. Decision rules (pre-registered in PREREG.md Sec. 7; v2's D0-D3 replace the base design's D1-D4)

Primary runs (eca-b, float64 evaluation), 3 seeds per row; numbers are 3-seed means, 95% intervals in Sec. 13 and
`results/tables.md`. "cut vs P0" = 1 - KL(P2)/KL(P0); negative = worse than one pass.

VERDICTS_TABLE

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
