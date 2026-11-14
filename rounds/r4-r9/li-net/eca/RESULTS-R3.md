# E-CA round 3: a zero-overlap cellular-automaton grid, a depth gate, and stiff-first versus random search: results

eurisko study, 2026-10-15 (round 3, 09:27-13:xxZ). Written by an Opus 5.5 subagent for eurisko. Asked by Kelvin ("Please run
new experiments with gemini; we didn't like the previous results"); spec by gemini
(`/home/agent/workspaces/mats/outbox/overnight-followups-round-3.md`, Rank 2), dispatched by opus-mats (mail #3649).
Pre-registration: `PREREG-R3.md` (drafted 09:46Z; its final version, after a header-line edit, was saved at 09:47:03.48Z, 82 ms
before the smoke's source copy and 18 minutes before the main launches; a copy, `/tmp/eca/PREREG-R3-prelaunch.md`, md5
da122736, was taken in between; unchanged since. The 09:46Z draft time cannot be verified: the edit created a new file).
Numbers: `results/r3/summary.json`, `results/r3/tables.md`, re-derivable from `results/raw/r3/`. Audit: `AUDIT-R3.md` (no
high-severity finding; every verdict reproduced; its wording findings are applied below, Sec. 11).

## 1. TL;DR

- **The overlap is gone, and there is a real gap.** At W = 12, T = 10 no validation or test grid shares its row 0 with
  training (0.000), and the one-pass network has a real multi-step gap (round 3 changed the overlap, W, T and the depth at
  once, so the gap cannot be attributed to the overlap removal alone): rule 110 one-pass KL on validation is F 0.13, B1 0.29, B2 0.24 nats at 4 layers,
  while its one-step check is 0.006. **R3-D0 PASSES for rule 110 (L = 4 selected).**
- **Rule 90 fails the gate at every depth.** Its seed-0 run at L = 4 misses the one-step bar by 0.0003 (S1 KL 0.0153 vs
  0.015; that seed stopped early on patience); L = 2 and 3 miss it by more. Per the pre-registration, rule 90's search rules
  are reported as descriptive only. (Its 3-seed mean at L = 4 would have passed, 0.0108: a near miss, verdict unchanged.)
- **R3-D2a FAILS for rule 110.** On forward queries (rows 6-9 from row 0) under the anchored pool at M = 16, stiffness-first,
  confidence-first and hybrid search each cut KL by 13-15% (bar 25%), though each beats random order by 15-17% with CIs above
  0 (random order gains nothing on F). **R3-D2b FAILS:** the hybrid cuts F, B1, B2 by 12-14% and almost never branched (rule
  110: one branch step in all test chains; some unrevealed cell always had H < 0.20 nats), so it behaved like
  confidence-first.
- **R3-D3 FAILS for rule 110** (and rule 90, descriptive): the stiffness head (Spearman 0.700 with -KL(P* || q)), plain
  confidence (0.697) and 1-step self-consistency (0.697) rank the network's own errors equally well; best minus confidence
  +0.002 (bar 0.05). The comparison is close to degenerate: the head ranks pairs almost exactly like plain confidence
  (Spearman 0.97-0.99 between them), and plain confidence even tracks the exact information gain slightly better (0.937 vs
  0.926).
- **Secondary (not a rule):** VoI among the 6 stiffest cells is the strongest forward search (rule 110 F -31%, paired CI
  above 0; rule 90 F -73%, descriptive, CI includes 0); random order is the strongest on rule-110 bridges (B1 -24%, B2 -23%, CIs above 0);
  VoI among the most uncertain cells is harmful under the plain mixture and neutral under the anchored pool (0%). The arms the
  rules name (stiff, conf, hyb) improve on one pass, but by less than the pre-registered bars; VoI over stiff cells (31%) is
  above the 25% bar but is not one of them.
- **Cost:** $13.66 for round 3 (cap $40); night total for E-CA $33.56; zero eca VMs left.

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

## 3. Decision rules (pre-registered in PREREG-R3.md Sec. 5; 3-seed means, paired 95% t-intervals)

| rule | rule 110 (L = 4, selected) | rule 90 (gate failed; L = 4 shown, descriptive) |
|---|---|---|
| R3-D0 gate | **PASS**: validation one-pass KL F 0.129, B1 0.295, B2 0.244 (bar >= 0.08 on 2 of 3); S1 0.0058 (bar <= 0.015) | **FAIL**: no depth passes on seed 0 (L = 4: S1 0.0153; L = 3: 0.0248; L = 2: 0.0512) |
| R3-D2a forward, M = 16, sw | **FAIL**: cut vs one pass stiff 0.130 [0.052, 0.209], conf 0.146 [0.096, 0.197], hyb 0.136 [0.057, 0.214] (bar 0.25); vs random 0.153 / 0.168 / 0.158, KL(rand) - KL(arm) CIs above 0 (bar 0.15: met) | descriptive: cuts vs one pass 0.52 / 0.50 / 0.50; vs random 0.14 / 0.11 / 0.11, CIs include 0 |
| R3-D2b hybrid, M = 16, sw | **FAIL**: cut vs one pass F 0.136, B1 0.119 [-0.090, 0.329], B2 0.137; KL(stiff) - KL(hyb) on B1 -0.010, B2 -0.001 (CIs include 0); beats random on F (0.021 [0.009, 0.033]) | descriptive: F 0.50, B1 0.18, B2 0.03 |
| R3-D3 epistemic ranking | **FAIL**: Spearman with -KL(P* or q): beta 0.700, confidence 0.697, self-consistency 0.697; best minus confidence +0.002 (bar 0.05) | descriptive: 0.385 / 0.379 / 0.342; +0.006 |

How the rules were applied (`analyze_r3.py`, written before any result existed): the depth L per rule is the largest L
whose seed-0 validation one-pass KL passes the gate; R3-D2a needs each of stiff, conf, hyb to cut KL vs one pass by at least
25% (mean of per-seed cuts, paired CI of the KL difference above 0) and to beat random order by at least 15% (paired CI
above 0); R3-D2b needs the hybrid to cut F, B1 and B2 by at least 25% each, to beat stiff on B1 and B2 and random on F;
R3-D3 needs the stiffness head or self-consistency to beat plain confidence by at least 0.05 in Spearman with -KL.

## 4. The depth gate

| rule | L | F | B1 | B2 | S1 | gate (seed 0) |
|---|---|---|---|---|---|---|
| 110 | 2 | 0.131 | 0.365 | 0.372 | 0.0174 | fail (S1) |
| 110 | 3 | 0.131 | 0.377 | 0.315 | 0.0072 | pass |
| 110 | 4 | 0.130 | 0.353 | 0.250 | 0.0064 | pass, selected |
| 90 | 2 | 0.151 | 0.223 | 0.160 | 0.0512 | fail (S1) |
| 90 | 3 | 0.064 | 0.062 | 0.052 | 0.0248 | fail (S1) |
| 90 | 4 | 0.059 | 0.057 | 0.045 | 0.0153 | fail (S1, by 0.0003) |

(Seed-0 validation one-pass KL, nats.) With memorisation removed, the one-pass network has a large multi-step gap on rule
110 (validation KL 0.13-0.38 nats on F, B1, B2 at every depth) while still learning the local rule at L >= 3. Rule 90's
XOR rule is learned to the bar only by some seeds: at L = 4, seeds 1 and 2 reach S1 0.0070 and 0.0100 (both trained the full
70 minutes) and seed 0, which stopped on patience after 43 minutes, 0.0153. Most rule-110 runs stopped at the 70-minute
cap rather than on patience (validation loss still improving), so the networks are not fully converged.

## 5. Search (rule 110 at L = 4; rule 90 descriptive)

IMG: results/r3/kl-vs-M-r3-r110.png

IMG: results/r3/kl-vs-M-r3-r90.png

Each selector against one pass at M = 16 (cut = 1 - KL/KL(P0), 3-seed mean; * = paired CI of the KL difference above 0;
secondary readout, computed 11:58Z after the results were read; not a decision rule):

| rule 110, L = 4 | one-pass KL | stiff sw | conf sw | hyb sw | voi sw | voi mix | voiu sw | rand sw | rand mix |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.128 | +0.13* | +0.15* | +0.14* | +0.31* | +0.26* | +0.00 | -0.03 | +0.09 |
| B1 | 0.301 | +0.15 | +0.15 | +0.12 | +0.23 | +0.05 | -0.00 | +0.24* | +0.16 |
| B2 | 0.256 | +0.14* | +0.15* | +0.14* | +0.20 | +0.16 | +0.00 | +0.23* | +0.22* |

- On **forward** queries, cells chosen by stiffness or confidence help (13-15%) and random cells do not; VoI among the
  stiffest cells helps most (31%). On **bridges**, random order helps most (23-24%), as in rounds 1-2. VoI among the most
  uncertain cells is harmful under the plain mixture (B1 0.30 -> 0.43 at M = 16) and exactly neutral under the anchored pool,
  whose fitted weights then put everything on the one-pass belief.
- The **hybrid almost never branched** (rule 110: one branch step over all test chains; rule 90 at L = 4: a few dozen on F):
  some unrevealed cell anywhere in the grid nearly always had predictive entropy below 0.20 nats, so it took the
  lowest-entropy cell, which is what confidence-first does (the auditor's replay found identical picks in 99.8-100% of
  steps). These cells are the network's most confident ones anywhere, often far from the query, not a frontier next to it.
  The two arms' 3-seed mean KLs differ by at most 0.007 under sw (single seeds by up to 0.021), through sampling.
- The **Gram-dedup pool** (dd) is worse than one pass on bridges even before any search (at M = 0: B1 +19%, B2 +59%),
  because it is anchored on the network's no-evidence prior with one fitted scale; relative to its own M = 0, VoI chains
  lower its KL on B2. So the dd pool, not the chains, is what is worse than one pass; see Sec. 12.
- **Rule 90 (descriptive):** forward KL falls from 0.067 to 0.032 (stiff) and 0.017 (VoI), whose CIs include 0 (seed 2
  has a larger one-pass error and a larger gain); conf, hyb and random have CIs above 0 (-50%, -50%, -42%); random order is
  best on B2 (-34%).
- Compute per query at M = 16 (K = 16 chains): 272 forward passes for stiff, conf, hyb and rand; 3,344 for voi and voiu.

## 6. Epistemic versus aleatoric stiffness (R3-D3)

| rule | beta_int | confidence ln 2 - H(q) | self-consistency | -s_hat | entropy-matched: beta / confidence / self-consistency |
|---|---|---|---|---|---|
| 110 | 0.700 [0.648, 0.752] | 0.697 [0.649, 0.746] | 0.697 [0.650, 0.743] | 0.700 | 0.172 / 0.201 / 0.132 |
| 90 | 0.385 [0.278, 0.492] | 0.379 [0.275, 0.483] | 0.342 [0.279, 0.405] | 0.387 | 0.175 / 0.166 / -0.008 |

(Spearman with -KL(P*(X_i | x_S) || q(X_i | x_S)) over 2,000 held-out (cell, evidence) pairs per seed.) The stiffness head
ranks these pairs almost exactly like plain confidence (Spearman 0.97-0.99 between the two scores per seed), so part (a) of
the rule is nearly the same ranking as part (b) and the null is close to built in; only self-consistency is a different
score (0.89 with confidence), and it does worse. Within matched entropy bins the correlations drop (rule 110: 0.13-0.20;
rule 90: 0.17 for beta and confidence, -0.008 for self-consistency) and no score beats plain confidence by the bar
(entropy-matched gaps -0.030 and +0.010). Against the exact information gain ln 2 - H*, plain confidence tracks slightly
better than the head at L = 4 (rule 110: 0.937 vs 0.926, higher in all three seeds; self-consistency 0.839).

## 7. Other readouts

- D3 (keys, rule 110): redundant views keep key cosine 0.986 and the dedup drift ratio is 0.027 at m = 5; forward and
  backward chains have key cosine 0.25 (B1) and 0.21 (B2) with wide intervals (round 2 at T = 6: -0.10 and 0.07).
- Training: 18 runs, 43-71 minutes each; validation outcome loss at L = 4 (best per seed): rule 110 0.164-0.167, rule 90 0.152-0.157.

## 8. What this means

- In the round-3 setup (zero overlap, a wider and deeper grid, a depth gate, all changed together) a 4-layer network learns
  the local rule for rule 110 (one-step KL 0.006) while one pass misses forward questions 6-9 steps out by 0.13 nats (exact
  entropy 0.56). No arm keeps W = 12, T = 10 with overlapping splits, so the gap cannot be attributed to removing the
  overlap alone.
- In that regime search through the network's own conditionals does help: up to 31% on forward queries (VoI over stiff
  cells) and 24% on bridges (random order), with the anchored pool. The pre-registered claims were stronger: stiff-first and
  the hybrid at >= 25% on forward queries, the hybrid winning on bridges, and a stiffness signal that beats plain confidence.
  None of those held.
- The hybrid as specified collapses to confidence-first at L = 4, because some nearly determined cell is almost always
  available somewhere in the grid; a different threshold, a candidate set restricted near the query, or a budget split would
  be needed to make it branch (not tested).

## 9. Deviations

- All 18 runs (both rules, L = 2, 3, 4, seeds 0-2) were launched together rather than seed 0 first; the depth selection
  used only the seed-0 validation numbers, as pre-registered (Sec. 3 of PREREG-R3).
- Chains ran on F, B1 and B2 only (R was kept for P0 but no rule uses it), with 500 validation and 500 test queries per
  family (the round-1 chain size was 1,000), for time.
- Most runs hit the 70-minute training cap (`max_minutes`) instead of patience.
- Rule 90's gate failed on seed 0 by 0.0003 in S1 KL; its search numbers are descriptive only, as pre-registered.
- The W = 14 stretch was not run (time); the W = 14/16 generator and inference are ready (`ecadata3.py`).
- **Spec version.** The spec file was revised at 09:35:51Z (W = 14 primary, a fixed 4-layer net, 100k-150k training
  grids, a KL-form TD self-consistency, R3-D3 reported overall and stratified, R3-D0 as a 3-seed mean). This round follows
  the version read and dispatched just before that (opus-mats mail #3649, 09:34:47Z: W = 12 primary, depth gate over L in
  {2, 3, 4} on seed 0, W = 14 as a stretch). Where that matters, per the audit: under a 3-seed gate rule 90 would pass at
  L = 4 (its 3-seed S1 is 0.0108), but its R3-D2a would still fail part (ii) and its R3-D3 the 0.05 bar; with the KL form of
  self-consistency, R3-D3's self-consistency score is lower (0.642 vs 0.696); stratified R3-D3 gaps are -0.030 / +0.010.
  The hybrid samples its chosen cell from q (the spec says clamp); with that cell's entropy at most 0.114 nats the two
  rarely differ.
- Documentation slips (no effect on results): the docstrings of `ecadata3.py` and a comment in `ecarun.py` still describe
  the earlier W in {14, 16} plan and 80/5/7.5/7.5 split (the code uses W = 12 and 3072/384/320/320); result.json's
  `D0_gate_val` field applies the round-1 gate, not R3-D0, and is not used by `analyze_r3.py`.

## 10. Cost and clean-up

- Round 3: **$13.66** at the $0.87/h ceiling for 10 on-demand L4 VMs (1 smoke + 9 main; cap $40); night total for E-CA **$33.56** (`GPU-LOG.md`, rebuilt by `gpu/ledger.py`).
- Live eca VMs at the final check: none. MATS cluster: not used.

## 11. Audit and post-audit corrections

The independent audit (`AUDIT-R3.md`, Opus 5.5, 12:00-12:18Z) re-derived every verdict and table line from the raw files
(2,154 summary numbers within 3.9e-7), regenerated the data (0 overlap, shipped data identical), checked the exact posteriors
against its own dense 4,096-state code (9.7e-15), confirmed the code diff and that nothing leaks, and the GPU total. It found
no high-severity issue. Corrections applied after it (12:20Z): the R3-D3 null is close to degenerate and plain confidence
tracks the information gain better than the head (M1); the gap is not attributed to the overlap removal alone (M2); the dd
pool's offset, not the VoI chains, is what is worse than one pass (M3); and L1-L12 (hybrid branching counts and the
"frontier" wording, rounding, the rule-90 outlier seed, the VMs count, the round-2 one-layer wording, the spec revision at
09:35:51Z, the unverifiable draft time, stale docstrings, the unused round-1 gate field).

## 12. Full tables

### Depth gate (seed 0, validation one-pass KL, nats)

| rule | L | F | B1 | B2 | R | S1 | gate |
|---|---|---|---|---|---|---|---|
| 110 | 2 | 0.1312 | 0.3649 | 0.3718 | 0.0770 | 0.0174 | fail |
| 110 | 3 | 0.1309 | 0.3767 | 0.3148 | 0.0538 | 0.0072 | pass |
| 110 | 4 | 0.1302 | 0.3526 | 0.2501 | 0.0382 | 0.0064 | pass |

Rule 110: selected L = 4 (largest passing; bar S1 <= 0.015 and >= 0.08 on 2 of F/B1/B2)

| 90 | 2 | 0.1509 | 0.2231 | 0.1596 | 0.0558 | 0.0512 | fail |
| 90 | 3 | 0.0644 | 0.0619 | 0.0520 | 0.0260 | 0.0248 | fail |
| 90 | 4 | 0.0594 | 0.0572 | 0.0446 | 0.0166 | 0.0153 | fail |

Rule 90: selected L = None (largest passing; bar S1 <= 0.015 and >= 0.05 on 2 of F/B1/B2)


### Rule 110 at L = 4 (selected by the gate; seeds [0, 1, 2])

R3-D0 (3-seed validation one-pass KL): F 0.1285 [0.1219, 0.1351]; B1 0.2946 [0.1686, 0.4206]; B2 0.2440 [0.2307, 0.2574]; R 0.0370 [0.0343, 0.0396]; S1 0.0058 [0.0042, 0.0073] -> **PASS**

Test one-pass KL (500 chain instances): F 0.1282 [0.1212, 0.1352]; B1 0.3013 [0.1752, 0.4273]; B2 0.2559 [0.2376, 0.2742]

R3-D2a (F, M = 16, sw) -> **FAIL**

| arm | KL | cut vs P0 | cut vs rand | KL(rand) - KL(arm) | pass |
|---|---|---|---|---|---|
| stiff | 0.1115 [0.1062, 0.1167] | 0.130 [0.052, 0.209] | 0.153 [0.080, 0.226] | 0.0202 [0.0096, 0.0308] | no |
| conf | 0.1094 [0.1041, 0.1148] | 0.146 [0.096, 0.197] | 0.168 [0.084, 0.252] | 0.0222 [0.0100, 0.0344] | no |
| hyb | 0.1108 [0.1047, 0.1169] | 0.136 [0.057, 0.214] | 0.158 [0.076, 0.240] | 0.0208 [0.0090, 0.0327] | no |

R3-D2b (hyb, sw, M = 16) -> **FAIL**

| family | KL hyb | cut vs P0 | comparison | pass |
|---|---|---|---|---|
| F | 0.1108 [0.1047, 0.1169] | 0.136 [0.057, 0.214] | rand - hyb 0.0208 [0.0090, 0.0327] | no |
| B1 | 0.2626 [0.2131, 0.3121] | 0.119 [-0.090, 0.329] | stiff - hyb -0.0099 [-0.0580, 0.0383] | no |
| B2 | 0.2206 [0.2075, 0.2337] | 0.137 [0.028, 0.246] | stiff - hyb -0.0014 [-0.0093, 0.0065] | no |

R3-D3 (Spearman with -KL(P* || q) over 2,000 pairs): beta 0.6998 [0.6477, 0.7519], confidence 0.6973 [0.6489, 0.7457], self-consistency 0.6965 [0.6502, 0.7427] (-s_hat 0.6998 [0.6476, 0.7520]); best minus confidence +0.002 -> **FAIL**. Entropy-matched: beta 0.1715 [0.0806, 0.2625], confidence 0.2010 [0.1484, 0.2536], self-consistency 0.1322 [0.0804, 0.1841]

KL vs M (3-seed means; P0 is M = 0 for sw):

| family | selector, pool | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff, sw | 0.1282 | 0.1280 | 0.1280 | 0.1272 | 0.1238 | 0.1115 |
| F | stiff, mix | 0.1282 | 0.1276 | 0.1274 | 0.1271 | 0.1231 | 0.1124 |
| F | stiff, dd | 0.1302 | 0.1302 | 0.1302 | 0.1300 | 0.1291 | 0.1242 |
| F | conf, sw | 0.1282 | 0.1277 | 0.1269 | 0.1251 | 0.1218 | 0.1094 |
| F | conf, mix | 0.1282 | 0.1270 | 0.1264 | 0.1251 | 0.1211 | 0.1106 |
| F | conf, dd | 0.1302 | 0.1301 | 0.1300 | 0.1300 | 0.1286 | 0.1227 |
| F | voi, sw | 0.1282 | 0.1265 | 0.1244 | 0.1209 | 0.1096 | 0.0885 |
| F | voi, mix | 0.1282 | 0.1260 | 0.1241 | 0.1207 | 0.1089 | 0.0952 |
| F | voi, dd | 0.1302 | 0.1301 | 0.1300 | 0.1293 | 0.1252 | 0.1195 |
| F | voiu, sw | 0.1282 | 0.1281 | 0.1303 | 0.1289 | 0.1284 | 0.1282 |
| F | voiu, mix | 0.1282 | 0.1301 | 0.1339 | 0.1401 | 0.1515 | 0.1572 |
| F | voiu, dd | 0.1302 | 0.1296 | 0.1527 | 0.1488 | 0.1479 | 0.1453 |
| F | hyb, sw | 0.1282 | 0.1277 | 0.1270 | 0.1252 | 0.1217 | 0.1108 |
| F | hyb, mix | 0.1282 | 0.1270 | 0.1264 | 0.1251 | 0.1211 | 0.1104 |
| F | hyb, dd | 0.1302 | 0.1301 | 0.1301 | 0.1299 | 0.1286 | 0.1223 |
| F | rand, sw | 0.1282 | 0.1277 | 0.1287 | 0.1328 | 0.1354 | 0.1316 |
| F | rand, mix | 0.1282 | 0.1270 | 0.1266 | 0.1253 | 0.1233 | 0.1165 |
| F | rand, dd | 0.1302 | 0.1298 | 0.1364 | 0.1394 | 0.1352 | 0.1322 |
| B1 | stiff, sw | 0.3013 | 0.2982 | 0.2935 | 0.2851 | 0.2739 | 0.2528 |
| B1 | stiff, mix | 0.3013 | 0.2989 | 0.2947 | 0.2872 | 0.2752 | 0.2562 |
| B1 | stiff, dd | 0.3595 | 0.3585 | 0.3592 | 0.3610 | 0.3807 | 0.3948 |
| B1 | conf, sw | 0.3013 | 0.2982 | 0.2937 | 0.2844 | 0.2736 | 0.2556 |
| B1 | conf, mix | 0.3013 | 0.2982 | 0.2930 | 0.2844 | 0.2732 | 0.2668 |
| B1 | conf, dd | 0.3595 | 0.3606 | 0.3572 | 0.3517 | 0.3562 | 0.3696 |
| B1 | voi, sw | 0.3013 | 0.2892 | 0.2837 | 0.2735 | 0.2578 | 0.2290 |
| B1 | voi, mix | 0.3013 | 0.2894 | 0.2833 | 0.2752 | 0.2683 | 0.2821 |
| B1 | voi, dd | 0.3595 | 0.3556 | 0.3557 | 0.3532 | 0.3668 | 0.3875 |
| B1 | voiu, sw | 0.3013 | 0.2942 | 0.2991 | 0.3014 | 0.3015 | 0.3013 |
| B1 | voiu, mix | 0.3013 | 0.3159 | 0.3317 | 0.3633 | 0.4069 | 0.4260 |
| B1 | voiu, dd | 0.3595 | 0.3773 | 0.4474 | 0.4442 | 0.4477 | 0.4508 |
| B1 | hyb, sw | 0.3013 | 0.2980 | 0.2926 | 0.2837 | 0.2728 | 0.2626 |
| B1 | hyb, mix | 0.3013 | 0.2982 | 0.2929 | 0.2841 | 0.2727 | 0.2709 |
| B1 | hyb, dd | 0.3595 | 0.3596 | 0.3580 | 0.3508 | 0.3587 | 0.3783 |
| B1 | rand, sw | 0.3013 | 0.2836 | 0.2733 | 0.2616 | 0.2497 | 0.2276 |
| B1 | rand, mix | 0.3013 | 0.2942 | 0.2863 | 0.2760 | 0.2627 | 0.2527 |
| B1 | rand, dd | 0.3595 | 0.3440 | 0.3456 | 0.3536 | 0.3460 | 0.2768 |
| B2 | stiff, sw | 0.2559 | 0.2552 | 0.2542 | 0.2484 | 0.2396 | 0.2192 |
| B2 | stiff, mix | 0.2559 | 0.2550 | 0.2540 | 0.2485 | 0.2383 | 0.2193 |
| B2 | stiff, dd | 0.4060 | 0.4090 | 0.4122 | 0.3862 | 0.3596 | 0.3143 |
| B2 | conf, sw | 0.2559 | 0.2549 | 0.2531 | 0.2483 | 0.2410 | 0.2177 |
| B2 | conf, mix | 0.2559 | 0.2547 | 0.2532 | 0.2484 | 0.2403 | 0.2188 |
| B2 | conf, dd | 0.4060 | 0.4086 | 0.4064 | 0.3933 | 0.3420 | 0.3127 |
| B2 | voi, sw | 0.2559 | 0.2477 | 0.2421 | 0.2372 | 0.2300 | 0.2053 |
| B2 | voi, mix | 0.2559 | 0.2477 | 0.2419 | 0.2372 | 0.2318 | 0.2147 |
| B2 | voi, dd | 0.4060 | 0.3857 | 0.3632 | 0.3489 | 0.3386 | 0.3319 |
| B2 | voiu, sw | 0.2559 | 0.2560 | 0.2538 | 0.2540 | 0.2555 | 0.2559 |
| B2 | voiu, mix | 0.2559 | 0.2598 | 0.2629 | 0.2838 | 0.3208 | 0.3486 |
| B2 | voiu, dd | 0.4060 | 0.3808 | 0.3351 | 0.3392 | 0.3535 | 0.3625 |
| B2 | hyb, sw | 0.2559 | 0.2549 | 0.2532 | 0.2485 | 0.2410 | 0.2206 |
| B2 | hyb, mix | 0.2559 | 0.2547 | 0.2531 | 0.2484 | 0.2405 | 0.2190 |
| B2 | hyb, dd | 0.4060 | 0.4088 | 0.4040 | 0.3960 | 0.3434 | 0.3144 |
| B2 | rand, sw | 0.2559 | 0.2478 | 0.2377 | 0.2276 | 0.2148 | 0.1981 |
| B2 | rand, mix | 0.2559 | 0.2472 | 0.2396 | 0.2262 | 0.2089 | 0.1982 |
| B2 | rand, dd | 0.4060 | 0.4223 | 0.3906 | 0.3525 | 0.3007 | 0.2231 |

hyb branch share (test): test|F 0.000 [0.000, 0.000]; test|B1 0.000 [0.000, 0.000]; test|B2 0.000 [-0.000, 0.000]

Forward passes per query (K = 16) at M = 16: stiff 272; conf 272; voi 3344; voiu 3344; hyb 272; rand 272

D1: Spearman(beta, ln2 - H*) 0.9254 [0.8885, 0.9622], AUC 0.8455 [0.8246, 0.8663]; D3a ratio m=5 0.0274 [0.0143, 0.0404], redundant cos 0.9856 [0.9638, 1.0074]; cos(k_fwd, k_bwd) B1 0.2529 [-0.3229, 0.8288], B2 0.2070 [-0.5530, 0.9669]

### Rule 90 at L = 4 (GATE FAILED: L = 4 shown, descriptive; seeds [0, 1, 2])

R3-D0 (3-seed validation one-pass KL): F 0.0693 [0.0164, 0.1222]; B1 0.0754 [-0.0137, 0.1644]; B2 0.0370 [0.0195, 0.0545]; R 0.0156 [0.0127, 0.0185]; S1 0.0108 [0.0004, 0.0212] -> **GATE FAILED (descriptive)**

Test one-pass KL (500 chain instances): F 0.0671 [0.0129, 0.1213]; B1 0.0705 [-0.0163, 0.1572]; B2 0.0391 [0.0187, 0.0595]

R3-D2a (F, M = 16, sw) -> **GATE FAILED (descriptive)**

| arm | KL | cut vs P0 | cut vs rand | KL(rand) - KL(arm) | pass |
|---|---|---|---|---|---|
| stiff | 0.0317 [0.0081, 0.0554] | 0.522 [0.306, 0.737] | 0.135 [-0.593, 0.864] | 0.0098 [-0.0353, 0.0549] | no |
| conf | 0.0341 [0.0014, 0.0667] | 0.497 [0.317, 0.677] | 0.105 [-0.461, 0.672] | 0.0075 [-0.0268, 0.0418] | no |
| hyb | 0.0340 [0.0006, 0.0674] | 0.499 [0.303, 0.695] | 0.109 [-0.464, 0.682] | 0.0075 [-0.0265, 0.0415] | no |

R3-D2b (hyb, sw, M = 16) -> **GATE FAILED (descriptive)**

| family | KL hyb | cut vs P0 | comparison | pass |
|---|---|---|---|---|
| F | 0.0340 [0.0006, 0.0674] | 0.499 [0.303, 0.695] | rand - hyb 0.0075 [-0.0265, 0.0415] | no |
| B1 | 0.0554 [0.0056, 0.1052] | 0.183 [-0.071, 0.437] | stiff - hyb -0.0003 [-0.0024, 0.0018] | no |
| B2 | 0.0379 [0.0202, 0.0556] | 0.027 [-0.035, 0.089] | stiff - hyb 0.0003 [-0.0002, 0.0008] | no |

R3-D3 (Spearman with -KL(P* || q) over 2,000 pairs): beta 0.3851 [0.2784, 0.4918], confidence 0.3789 [0.2747, 0.4830], self-consistency 0.3420 [0.2787, 0.4053] (-s_hat 0.3874 [0.2809, 0.4940]); best minus confidence +0.006 -> **GATE FAILED (descriptive)**. Entropy-matched: beta 0.1748 [0.0235, 0.3261], confidence 0.1659 [0.0695, 0.2624], self-consistency -0.0083 [-0.0423, 0.0256]

KL vs M (3-seed means; P0 is M = 0 for sw):

| family | selector, pool | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|---|---|
| F | stiff, sw | 0.0671 | 0.0653 | 0.0630 | 0.0568 | 0.0450 | 0.0317 |
| F | stiff, mix | 0.0671 | 0.0656 | 0.0631 | 0.0566 | 0.0453 | 0.0388 |
| F | stiff, dd | 0.1340 | 0.1354 | 0.1378 | 0.1379 | 0.1339 | 0.1466 |
| F | conf, sw | 0.0671 | 0.0655 | 0.0636 | 0.0580 | 0.0457 | 0.0341 |
| F | conf, mix | 0.0671 | 0.0662 | 0.0640 | 0.0589 | 0.0489 | 0.0437 |
| F | conf, dd | 0.1340 | 0.1355 | 0.1372 | 0.1384 | 0.1354 | 0.1484 |
| F | voi, sw | 0.0671 | 0.0588 | 0.0502 | 0.0327 | 0.0178 | 0.0166 |
| F | voi, mix | 0.0671 | 0.0594 | 0.0506 | 0.0364 | 0.0298 | 0.0332 |
| F | voi, dd | 0.1340 | 0.1352 | 0.1349 | 0.1346 | 0.1393 | 0.1530 |
| F | voiu, sw | 0.0671 | 0.0636 | 0.0629 | 0.0607 | 0.0611 | 0.0589 |
| F | voiu, mix | 0.0671 | 0.0760 | 0.0754 | 0.0826 | 0.0916 | 0.1041 |
| F | voiu, dd | 0.1340 | 0.1419 | 0.1501 | 0.1541 | 0.1576 | 0.1377 |
| F | hyb, sw | 0.0671 | 0.0655 | 0.0637 | 0.0580 | 0.0457 | 0.0340 |
| F | hyb, mix | 0.0671 | 0.0661 | 0.0640 | 0.0589 | 0.0488 | 0.0440 |
| F | hyb, dd | 0.1340 | 0.1356 | 0.1373 | 0.1383 | 0.1352 | 0.1492 |
| F | rand, sw | 0.0671 | 0.0586 | 0.0526 | 0.0446 | 0.0395 | 0.0415 |
| F | rand, mix | 0.0671 | 0.0588 | 0.0526 | 0.0446 | 0.0393 | 0.0437 |
| F | rand, dd | 0.1340 | 0.1373 | 0.1402 | 0.1414 | 0.1322 | 0.1054 |
| B1 | stiff, sw | 0.0705 | 0.0679 | 0.0657 | 0.0622 | 0.0577 | 0.0551 |
| B1 | stiff, mix | 0.0705 | 0.0679 | 0.0651 | 0.0621 | 0.0573 | 0.0588 |
| B1 | stiff, dd | 0.1071 | 0.1064 | 0.1058 | 0.1059 | 0.1114 | 0.1381 |
| B1 | conf, sw | 0.0705 | 0.0668 | 0.0648 | 0.0637 | 0.0587 | 0.0549 |
| B1 | conf, mix | 0.0705 | 0.0662 | 0.0642 | 0.0638 | 0.0584 | 0.0586 |
| B1 | conf, dd | 0.1071 | 0.1052 | 0.1051 | 0.1070 | 0.1122 | 0.1415 |
| B1 | voi, sw | 0.0705 | 0.0635 | 0.0599 | 0.0557 | 0.0508 | 0.0508 |
| B1 | voi, mix | 0.0705 | 0.0637 | 0.0602 | 0.0566 | 0.0555 | 0.0680 |
| B1 | voi, dd | 0.1071 | 0.1032 | 0.1018 | 0.1032 | 0.1153 | 0.1267 |
| B1 | voiu, sw | 0.0705 | 0.0636 | 0.0643 | 0.0607 | 0.0593 | 0.0619 |
| B1 | voiu, mix | 0.0705 | 0.0642 | 0.0651 | 0.0721 | 0.0827 | 0.0960 |
| B1 | voiu, dd | 0.1071 | 0.1120 | 0.1248 | 0.1412 | 0.1483 | 0.1239 |
| B1 | hyb, sw | 0.0705 | 0.0671 | 0.0648 | 0.0637 | 0.0583 | 0.0554 |
| B1 | hyb, mix | 0.0705 | 0.0662 | 0.0642 | 0.0636 | 0.0587 | 0.0596 |
| B1 | hyb, dd | 0.1071 | 0.1053 | 0.1052 | 0.1070 | 0.1123 | 0.1414 |
| B1 | rand, sw | 0.0705 | 0.0668 | 0.0608 | 0.0560 | 0.0542 | 0.0561 |
| B1 | rand, mix | 0.0705 | 0.0645 | 0.0608 | 0.0560 | 0.0543 | 0.0614 |
| B1 | rand, dd | 0.1071 | 0.1034 | 0.1028 | 0.0998 | 0.0957 | 0.0872 |
| B2 | stiff, sw | 0.0391 | 0.0385 | 0.0385 | 0.0384 | 0.0383 | 0.0382 |
| B2 | stiff, mix | 0.0391 | 0.0386 | 0.0382 | 0.0385 | 0.0384 | 0.0402 |
| B2 | stiff, dd | 0.0723 | 0.0714 | 0.0720 | 0.0733 | 0.0756 | 0.1014 |
| B2 | conf, sw | 0.0391 | 0.0391 | 0.0389 | 0.0383 | 0.0378 | 0.0379 |
| B2 | conf, mix | 0.0391 | 0.0387 | 0.0383 | 0.0384 | 0.0382 | 0.0404 |
| B2 | conf, dd | 0.0723 | 0.0706 | 0.0717 | 0.0737 | 0.0756 | 0.1037 |
| B2 | voi, sw | 0.0391 | 0.0381 | 0.0386 | 0.0382 | 0.0376 | 0.0371 |
| B2 | voi, mix | 0.0391 | 0.0384 | 0.0392 | 0.0389 | 0.0401 | 0.0420 |
| B2 | voi, dd | 0.0723 | 0.0717 | 0.0721 | 0.0765 | 0.0952 | 0.1291 |
| B2 | voiu, sw | 0.0391 | 0.0373 | 0.0362 | 0.0362 | 0.0343 | 0.0353 |
| B2 | voiu, mix | 0.0391 | 0.0405 | 0.0484 | 0.0557 | 0.0649 | 0.0731 |
| B2 | voiu, dd | 0.0723 | 0.0727 | 0.0774 | 0.0665 | 0.0647 | 0.0636 |
| B2 | hyb, sw | 0.0391 | 0.0391 | 0.0387 | 0.0383 | 0.0379 | 0.0379 |
| B2 | hyb, mix | 0.0391 | 0.0387 | 0.0383 | 0.0384 | 0.0382 | 0.0400 |
| B2 | hyb, dd | 0.0723 | 0.0705 | 0.0714 | 0.0739 | 0.0760 | 0.1053 |
| B2 | rand, sw | 0.0391 | 0.0376 | 0.0340 | 0.0306 | 0.0283 | 0.0259 |
| B2 | rand, mix | 0.0391 | 0.0364 | 0.0340 | 0.0306 | 0.0282 | 0.0253 |
| B2 | rand, dd | 0.0723 | 0.0685 | 0.0655 | 0.0639 | 0.0557 | 0.0522 |

hyb branch share (test): test|F 0.000 [-0.000, 0.001]; test|B1 0.000 [-0.000, 0.000]; test|B2 0.000 [0.000, 0.000]

Forward passes per query (K = 16) at M = 16: stiff 272; conf 272; voi 3344; voiu 3344; hyb 272; rand 272

D1: Spearman(beta, ln2 - H*) 0.8897 [0.8740, 0.9053], AUC 0.8165 [0.7917, 0.8413]; D3a ratio m=5 0.0565 [0.0477, 0.0653], redundant cos 0.9615 [0.9235, 0.9996]; cos(k_fwd, k_bwd) B1 -0.3998 [-0.6877, -0.1119], B2 -0.2970 [-0.5991, 0.0051]

### All depths (secondary): 3-seed validation one-pass KL and test KL under sw at M = 16

| rule | L | seeds | val F | val B1 | val B2 | val S1 | test P0 F/B1/B2 | stiff | conf | hyb | rand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 110 | 2 | [0, 1, 2] | 0.1297 | 0.3532 | 0.3673 | 0.0188 | 0.129/0.382/0.365 | 0.129/0.360/0.327 | 0.129/0.358/0.331 | 0.129/0.358/0.331 | 0.136/0.337/0.335 |
| 110 | 3 | [0, 1, 2] | 0.1282 | 0.3583 | 0.3077 | 0.0073 | 0.128/0.387/0.315 | 0.123/0.314/0.287 | 0.123/0.313/0.284 | 0.122/0.314/0.285 | 0.146/0.331/0.267 |
| 110 | 4 | [0, 1, 2] | 0.1285 | 0.2946 | 0.2440 | 0.0058 | 0.128/0.301/0.256 | 0.111/0.253/0.219 | 0.109/0.256/0.218 | 0.111/0.263/0.221 | 0.132/0.228/0.198 |
| 90 | 2 | [0, 1, 2] | 0.1518 | 0.2071 | 0.1348 | 0.0461 | 0.150/0.216/0.147 | 0.101/0.201/0.130 | 0.106/0.198/0.137 | 0.106/0.197/0.136 | 0.149/0.206/0.108 |
| 90 | 3 | [0, 1, 2] | 0.0825 | 0.1093 | 0.0617 | 0.0159 | 0.083/0.098/0.060 | 0.038/0.090/0.055 | 0.041/0.086/0.055 | 0.040/0.086/0.055 | 0.060/0.074/0.039 |
| 90 | 4 | [0, 1, 2] | 0.0693 | 0.0754 | 0.0370 | 0.0108 | 0.067/0.070/0.039 | 0.032/0.055/0.038 | 0.034/0.055/0.038 | 0.034/0.055/0.038 | 0.042/0.056/0.026 |
