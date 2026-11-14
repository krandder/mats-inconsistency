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

R3_COST
