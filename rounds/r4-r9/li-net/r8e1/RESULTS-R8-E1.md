# R8-E1 (E-INFO-ANYTIME): strong-root anytime safety, precision-aware Jacobi ridge, mixed-tier econ_kc_K3 and the 2x2 PUCT attribution: results

eurisko study, 2026-10-17 (15:44-17:33Z). Written by an Opus 5.5 subagent for eurisko. Assignment: opus-mats, agent-mail #3734
(group `r8`; this experiment's ceiling $0.80; 3 seeds; seed-t 95% CIs; independent audit; exact oracles for evaluation only;
no world-model training; every post-hoc edit logged when it happens; the VM deleted as soon as the run is done). Spec: gemini,
`/home/agent/workspaces/mats/outbox/experiments-round-8-spec.md`, global protocol and section 1. Amendments: the R8-E1 part of
`experiments-round-8-amendments.md` (15:58Z) approved readings 1, 2, 4, 5, 8, 9 and 10 of my mail #3738 and amended readings
3, 6 and 7.
- Pre-registration: `PREREG-R8-E1.md` and `PREREG-R8-E1-AMENDMENTS.md`, both registered 16:07:15Z, 7 s before the first launch
  (16:07:22Z in the launch log). Its deviations log is in section 5 (7 entries; section 8 here).
- Every number below names its source file; paths are relative to `li-net/r8e1/`. Derived descriptive numbers come from
  `numbers8.py`, output `results/main/analysis/numbers.txt` (written before any result existed; deviation 2). Audit:
  `AUDIT-R8-E1.md` (section 10).

Kelvin's question 1 asks whether search can ever make the belief worse than where it started as more reads arrive. Round 7
left three loose ends: a Jacobi pool that did not deduplicate identical clean reads, an econ_kc pool fitted on the wrong bags,
and a PUCT coverage gain that changed keys and score at once. Round 8 tests all four on R7-E1's saved world model, without
retraining it.

## 1. TL;DR

**At the spec's level, 1 of 3 rules passes (E1-D0); E1-D1 and E1-D2 fail. At the sub-rule level, 2 of 5 pass (E1-D0, E1-D1.1).
E1-D1.2 fails on its root-strength and naive-degradation clauses (its anchor-pool clauses pass); E1-D2.1 and E1-D2.2 each
fail on one clause.** Verdicts:
`results/main/analysis/verdicts.md` and `results.json`; 3-seed mean [seed-t 95% CI].

| rule | verdict | decisive numbers |
|---|---|---|
| E1-D0 gate | **PASS** | each loaded checkpoint matches its sha256 (3 seeds); world model Pearson with exact Bayes 0.9995 [0.9994, 0.9995] (bar 0.999); at K = 0 looped_attn_T3 returns the root exactly (KL difference 0.0, bar < 1e-7) |
| E1-D1.1 sparse-root anytime curve | **PASS** | looped_attn_T3's KL falls at every step K = 0, 1, 2, 4, 8, 16, 24, 32 in every seed (largest step change -0.00029, bar <= +0.0005); KL(32) 0.0303 [0.0287, 0.0319] (bar 0.033); JSD(24) 0.00575 [0.00519, 0.00631] (bar 0.0075) |
| E1-D1.2 strong-root safety | **FAIL** (root-strength and naive-degradation clauses) | looped_attn_T3 never degrades the 18-anchor root: every K improves it, by at least 0.0228 in every seed (bar: never worse than +0.0005); KL(0) - KL(16) = 0.0891 [0.0883, 0.0900]; KL_mmd(16) - KL_looped_attn(16) = 0.0227 [0.0200, 0.0253]. **But the 18-anchor root is at KL(0) = 0.1444 [0.1440, 0.1449] (bar <= 0.050), and ataraxos_mmd improves it at K = 16 (KL(16) - KL(0) = -0.0665 [-0.0700, -0.0630]; bar >= +0.015)** |
| E1-D2.1 jacobi_lin_T4_prec | **FAIL** (one clause) | 814 parameters; clean deduplication fixed: W(16)/W(1) at tau 0.3 = 1.164 [1.147, 1.182] (bar <= 1.25, CI upper bound <= 1.30; R7's jacobi_lin_T4 in this run: 1.455); W(8)/W(1) 1.150 < 1.765 < 2.501 in every seed; W(16)/W(1) at tau 2.5 = 2.816 (bar 2.0); beats looped_lin_T4 and kc_K1 in every seed (CIs > 0). **But P-FullSupport KL is 0.0414 [0.0396, 0.0432] (bar <= 0.040)** |
| E1-D2.2 mixed-tier econ_kc_K3 | **FAIL** (one clause) | W(16)/W(1) at tau 2.5 = 2.455 (bar 2.2); D3 MSE below kc_K3 (difference 0.00178 [0.00110, 0.00247]); the 2x2 PUCT table is reported. **But W(16)/W(1) at tau 0.3 = 1.417 [1.395, 1.439] (bar <= 1.25)** |

- **Kelvin's question 1, as measured here.**
  - In mean test KL over the 500 boards, neither anchor-aware attention pool ended above its root at any tested K, on any of
    the five anytime curves (sparse root; M_0 = 6, 12 and 18; three-tier reads at M_0 = 18), in any seed (per-seed checks in
    `numbers.txt`). No per-board check was made.
    - looped_attn_T3's KL fell at every step of each of these curves in every seed. At M_0 = 18 with tau 0.3/1.5 reads,
      every K >= 1 was below K = 0 by at least 0.0228 in every seed (three-tier reads: at least 0.0156).
    - attn_L3-K3 did the same except one sparse-root step, K = 24 to 32 in seed 102, that rose by 0.0007.
  - The pools without the anchor token (ataraxos_mmd, scaled_mix and R6's 4-step looped_lin_T4) are not anytime-safe on these
    streams. On the sparse-root stream, ataraxos_mmd's KL rises from 0.1084 (K = 16) to 0.1661 (K = 32), and scaled_mix's and
    looped_lin_T4's rise from K = 24 to 32. kc_dedup with raw keys is worse than the root at K = 1 (0.3523 vs 0.2797) and back
    at the root's level at K = 2 (0.2791). On every strong root, mmd and scaled_mix turn back up after K = 16, though their
    3-seed means stay below the root.
  - What did not happen is the strong root that E1-D1.2 assumed. Eighteen anchor cells leave the root at KL 0.144, not the
    0.035-0.050 the spec expected, and at K = 16 even mmd still improves it.
- **The precision-aware ridge passes all four weight-law clauses of E1-D2.1 and both comparisons; the rule fails on KL alone
  (0.0414 vs 0.040).**
  - Learned ridge at the tier's mean beta: about 0.35 for clean reads (about a sixth of the tau-2.5 ridge), 1.3-1.4 at tau 1.5,
    2.2 at tau 2.5.
  - jacobi_lin_T4_prec deduplicates identical clean reads (W(16)/W(1) 1.164) while adding support for noisy ones (2.816).
  - Its P-FullSupport KL (0.0414) misses 0.040 by 0.0014. R7's jacobi_lin_T4 in this run is at 0.0405, also above the bar.
- **econ_kc_K3 fitted on mixed-tier probes moves only part of the way.** Clean W(16)/W(1) goes from 1.491 (R7-style fit) to
  1.417. One shared lambda cannot both deduplicate clean reads and add support for noisy ones (interpretation, consistent
  with R7's audit).
- **2x2 PUCT** (coverage after 4 waves): K1 keys + R6 score 0.9200; K1 + R7 score 0.9580; K3 + R6 score 0.9651; K3 + R7 score
  0.9876. Both factors raise coverage. The K3 keys add 0.045 under the R6 score and 0.030 under the R7 score; the R7 score
  adds 0.038 with K1 keys and 0.023 with K3 keys. The gains are sub-additive (interaction -0.015 [-0.023, -0.008]).
- **Cost: $0.62** at the list-price ceiling, of the $0.80 ceiling.
  - $0.36 for a spot run that crashed at its first K = 1 test prefix, a torch Triton backend path that needs Python.h, absent
    on the VM (deviations 3 and 7). Its registered automatic retry kept the VM busy 7.3 min after the crash (about $0.12).
  - $0.26 for the on-demand rerun with a 7-line backend fix (one call, inside try/except with a log line). Its VM was
    deleted about 1 min after the run ended.
  - Zero r8e1 VMs at 16:57:48Z and again at 17:33:55Z.

## 2. What ran

- **Questions (spec section 1.1).**
  1. Kelvin's question 1: as reads are added (K = 0 to 32), does the belief ever get worse in expectation than the root, both
     from a sparse root (2 anchor readings) and from a strong root (18 anchor readings)? Naive pools should degrade a strong
     root; an anchor-aware attention pool should not.
  2. Does a 2-parameter precision-aware ridge let the Jacobi normal-equation pool deduplicate identical clean reads
     (round 7: W(16)/W(1) = 1.449)?
  3. Does fitting econ_kc_K3 on mixed-tier controlled-bag probes give it the three-regime weight law?
  4. Of round 7's PUCT coverage gain (0.9494 to 0.9863), how much comes from the keys (K1 vs K3) and how much from the score
     (R6 vs R7)?
- **World model.** No training. Each seed loads R7-E1's `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz` after checking its
  sha256 against its `.json` (669,633 parameters; the 6x6 board, ships 4/3/2, the commander's note and a dense sonar field
  with flip 0.15).
- **Roots.**
  - Root strength M_0 = the number of anchor cells whose readings the root sees: make_boards2's own per-board cell order, so
    M_0 = 2 is the R5-R7 root and the sets are nested.
  - A read is one theta_real call on the root's evidence plus one patch's new readings, re-flipped by its tier: tau 0.3 clean,
    1.5 with probability 0.127, 2.5 with 0.28.
- **Anytime pools.**
  - looped_attn_T3 and attn_L3-K3, each with its own K3 key head.
  - The root enters as an anchor token r = 0: its key comes from the root's footprint l0 - l_prior, its shift is 0, and its
    precision is beta_0 = M_0 beta_clean / 6.
  - Crowding is weighted by relative precision beta_b / beta_clean, the anchor included.
  - Training: BCE to the model's own K = 128 reference ref(M_0), with M_0 in {2, 6, 12, 18}, K in {0, 1, 2, 4, 8, 16, 24, 32}
    and tau uniform over the 3 tiers.
- **Other pools** (standard root, BCE to the reference, as R7):
  - attn_L3-K1, R6's looped_lin_T4, R7's jacobi_lin_T4, and the new jacobi_lin_T4_prec (814 parameters);
  - fitted scalars for ataraxos_mmd, kc_dedup on K1 and K3 keys, scaled_mix, and econ_kc_K3 fitted two ways: R7's bags, and
    the mixed-tier D3 calibration probes;
  - mix.
- **Tests** (the 500 R5-R7 test boards; exact Bayes given all 36 readings for scoring only):
  - the sparse-root curve on gemini's interleaved 32-read P-FullSupport stream (one partition tile per block of 4, the rest
    90/10 skewed clean reads);
  - strong-root curves at M_0 = 18 (and 6, 12) with 24 VOI-drawn 90/10 reads, tau 50/50 over 0.3 / 1.5, plus a three-tier
    version at M_0 = 18;
  - controlled bags (N same-patch reads at three tiers plus 8 background reads);
  - the 2x2 PUCT factorial (4 waves of 6, mixed reads).
- **Seeds** 101, 102, 103 on one L4. Statistics: 3-seed means with seed-t 95% intervals (t = 4.303).

## 3. E1-D0: checkpoints, world model and the anchor identity (PASS)

Source: `results.json` (`verdicts.E1-D0`), `results/main/s*/summary.json` (`checkpoint`), `numbers.txt`.
- Each seed loaded `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz`; its sha256 (aef34272..., abb0fffc..., f9842c1e...) matched its
  `.json`. No world model was trained.
- Clause-1 Pearson (theta_real on 6 standard readings + note vs exact Bayes on the same evidence): 0.99946 [0.99938, 0.99953]
  (per seed 0.99942 / 0.99948 / 0.99946).
- At K = 0, looped_attn_T3 and attn_L3-K3 return the root exactly on the 2-anchor and the 18-anchor root (max |logit
  difference| 0.0, max KL difference 0.0), because the anchor token's shift is 0.
- Root KL to exact Bayes by root strength: 0.2797 (M_0 = 2), 0.2452 (6), 0.1930 (12), 0.1444 (18).

## 4. E1-D1: anytime monotonicity (D1.1 PASS, D1.2 FAIL)

Source: `results.json` (`verdicts.E1-D1.1`, `E1-D1.2`, `stats`), figure `results/main/analysis/anytime.png` (data
`anytime_curves.csv`), `numbers.txt`.

**Sparse root (M_0 = 2)**, gemini's interleaved P-FullSupport stream: one partition tile per block of 4 reads, the other 3
are 90/10 skewed clean reads. KL to exact Bayes (mean of U_NW and U_SE) by number of reads K, 3-seed means:

| pool | K = 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| **looped_attn_T3** (anchor) | 0.2797 | 0.2305 | 0.1909 | 0.1504 | 0.1026 | 0.0559 | 0.0308 | **0.0303** |
| attn_L3-K3 (anchor) | 0.2797 | 0.2306 | 0.1914 | 0.1512 | 0.1041 | 0.0581 | 0.0334 | 0.0331 |
| jacobi_lin_T4_prec | 0.2797 | 0.2309 | 0.1924 | 0.1541 | 0.1096 | 0.0659 | 0.0414 | 0.0410 |
| kc_K3 | 0.2797 | 0.2386 | 0.2048 | 0.1716 | 0.1319 | 0.0909 | 0.0672 | 0.0664 |
| looped_lin_T4 (R6) | 0.2797 | 0.2516 | 0.2259 | 0.1848 | 0.1247 | 0.0717 | 0.0677 | 0.1131 |
| scaled_mix | 0.2797 | 0.2582 | 0.2368 | 0.2018 | 0.1523 | 0.1084 | 0.1042 | 0.1260 |
| ataraxos_mmd | 0.2797 | 0.2531 | 0.2281 | 0.1899 | 0.1400 | 0.1084 | 0.1218 | 0.1661 |
| kc_K1 | 0.2797 | 0.3523 | 0.2791 | 0.2114 | 0.1507 | 0.1118 | 0.0961 | 0.0940 |
| mix | 0.2797 | 0.2295 | 0.2110 | 0.2037 | 0.1997 | 0.1980 | 0.1972 | 0.1970 |

- **E1-D1.1 PASS.**
  - In every seed, each of the 7 steps lowers looped_attn_T3's KL; the smallest decrease is 0.00029 (`step_increase_per_seed`).
  - KL(32) = 0.0303 [0.0287, 0.0319] (bar 0.033).
  - JSD(24) between the U_NW and U_SE beliefs = 0.00575 [0.00519, 0.00631] (bar 0.0075).
- The pools without the anchor token are not anytime-safe on this stream (descriptive):
  - ataraxos_mmd rises after K = 16, and scaled_mix and looped_lin_T4 after K = 24;
  - kc_K1 is worse than the root at K = 1 (0.3523) and back at the root's level at K = 2 (0.2791);
  - mix stalls near 0.197.
- JSD is 0 at K = 0 and 1 for every pool, because both searches start with the same tile.

**Strong roots** (24 VOI-drawn 90/10 reads, tau 50/50 over 0.3 / 1.5 as amended): KL(K) - KL(0), 3-seed means. M_0 = 18 is
the rule's root.

| root, pool | K = 1 | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| M_0 = 18 (KL(0) 0.1444), **looped_attn_T3** | -0.0232 | -0.0378 | -0.0544 | -0.0726 | **-0.0891** | -0.0978 |
| M_0 = 18, attn_L3-K3 | -0.0230 | -0.0373 | -0.0537 | -0.0719 | -0.0881 | -0.0964 |
| M_0 = 18, ataraxos_mmd | -0.0118 | -0.0222 | -0.0387 | -0.0596 | **-0.0665** | -0.0457 |
| M_0 = 18, scaled_mix | -0.0112 | -0.0211 | -0.0369 | -0.0579 | -0.0677 | -0.0516 |
| M_0 = 18, mix | -0.0230 | -0.0327 | -0.0371 | -0.0397 | -0.0410 | -0.0413 |
| M_0 = 12 (KL(0) 0.1930), looped_attn_T3 / mmd | -0.0279 / -0.0146 | -0.0464 / -0.0276 | -0.0698 / -0.0491 | -0.0948 / -0.0769 | -0.1174 / -0.0893 | -0.1294 / -0.0700 |
| M_0 = 6 (KL(0) 0.2452), looped_attn_T3 / mmd | -0.0349 / -0.0180 | -0.0584 / -0.0343 | -0.0878 / -0.0605 | -0.1201 / -0.0951 | -0.1500 / -0.1153 | -0.1653 / -0.1000 |
| M_0 = 18, three-tier reads (reported), looped_attn_T3 / mmd | -0.0167 / -0.0089 | -0.0287 / -0.0171 | -0.0438 / -0.0308 | -0.0626 / -0.0507 | -0.0797 / -0.0655 | -0.0892 / -0.0590 |

- **E1-D1.2 FAIL, on its root-strength clause (KL(0) <= 0.050) and its naive-degradation clause (mmd KL(16) - KL(0) >=
  +0.015); its three anchor-pool clauses pass.**
  - Failing: the 18-anchor root is at KL(0) = 0.1444 [0.1440, 0.1449] (bar <= 0.050). ataraxos_mmd improves it at K = 16
    (-0.0665 [-0.0700, -0.0630]; bar >= +0.015), and so does mix (-0.0410 [-0.0412, -0.0407]).
  - Passing: looped_attn_T3's largest change over K >= 1 is -0.0234 / -0.0235 / -0.0228 per seed, so every K improves the root
    (bar <= +0.0005). KL(0) - KL(16) = 0.0891 [0.0883, 0.0900] > 0, and KL_mmd(16) - KL_looped_attn(16) = 0.0227
    [0.0200, 0.0253] > 0.
- Descriptive:
  - At every root strength, ataraxos_mmd and scaled_mix improve until K = 16 and then lose part of the gain by K = 24.
  - The anchor-aware pools keep improving through K = 24, also with three-tier reads.
- Interpretation (not tested): a root that the spec's naive pools would actually degrade needs much more evidence than 18 of
  36 cell readings. Gemini's gloss of "3 full 6-tile partitions" read in one call would have given about the R7 union call's
  KL (0.0022, `li-net/r7e1/RESULTS-R7-E1.md` line 147), and the approved reading 1 gave 0.144.

## 5. E1-D2.1: the precision-aware Jacobi ridge (FAIL on one clause)

Source: `results.json` (`verdicts.E1-D2.1`, `stats.weights`), figure `results/main/analysis/d3.png` (data `d3_weights.csv`,
`d3_mse.csv`), `numbers.txt`. W(N) = the sum over the N same-patch reads of a pool's per-read weight averaged over A*'s 6
cells (for the Jacobi pools, the exact coefficient of their linear map), as in R7-E1.

| pool, tier | W(1) | W(2) | W(4) | W(8) | W(16) | W(8)/W(1) | W(16)/W(1) |
|---|---|---|---|---|---|---|---|
| jacobi_lin_T4_prec, tau 0.3 | 0.915 | 0.985 | 1.028 | 1.052 | 1.065 | 1.150 | **1.164** [1.147, 1.182] |
| jacobi_lin_T4_prec, tau 1.5 | 0.646 | 0.853 | 1.023 | 1.141 | 1.214 | 1.765 | 1.880 |
| jacobi_lin_T4_prec, tau 2.5 | 0.441 | 0.670 | 0.905 | 1.104 | 1.243 | 2.501 | **2.816** [2.682, 2.950] |
| jacobi_lin_T4 (R7), tau 0.3 / 2.5 | | | | | | 1.409 / 2.204 | 1.455 / 2.427 |

- **E1-D2.1 FAIL, on the KL clause only.**
  - Failing: P-FullSupport KL (the interleaved stream's first 24 reads) is 0.0414 [0.0396, 0.0432] (per seed 0.0406 / 0.0420
    / 0.0416; bar <= 0.040).
  - Passing: lower than looped_lin_T4 (difference 0.0263 [0.0230, 0.0296]) and kc_K1 (0.0547 [0.0524, 0.0570]) in every
    seed; W(16)/W(1) at tau 0.3 = 1.164 with CI upper bound 1.182 (bars 1.25 and 1.30); W(8)/W(1) increasing across the tiers
    in every seed; W(16)/W(1) at tau 2.5 = 2.816 (bar 2.0).
- Learned values (`results.json` `fit.<seed>.jacobi_prec`): a = 2.57 / 2.50 / 2.53 and b = 1.45 / 1.40 / 1.35 per seed.
  The ridge at each tier's mean calibration beta is 0.35 / 0.34 / 0.36 (tau 0.3), 1.30 / 1.40 / 1.41 (tau 1.5) and
  2.17 / 2.20 / 2.23 (tau 2.5). So clean reads get a small ridge (0.35, about a sixth of the tau-2.5 ridge) and noisy reads
  a large one, as the spec intended.
- R7's jacobi_lin_T4, retrained in this run, is at P-FullSupport KL 0.0405, so the 0.040 bar is missed by both Jacobi
  variants. The precision-aware version is 0.0009 higher (descriptive).

## 6. E1-D2.2: econ_kc_K3 fitted on mixed-tier probes, and the 2x2 PUCT table (FAIL on one clause)

Source: `results.json` (`verdicts.E1-D2.2`, `stats.weights`, `stats.mse_all_tiers`, `stats.puct_*`), figures `d3.png` and
`results/main/analysis/puct2x2.png` (data `puct2x2.csv`), `numbers.txt`.

- **E1-D2.2 FAIL, on the clean-deduplication clause only.**
  - Failing: econ_kc_K3 fitted on the mixed-tier D3 calibration probes has W(16)/W(1) at tau 0.3 = 1.417 [1.395, 1.439] (bar
    <= 1.25).
  - Passing: W(16)/W(1) at tau 2.5 = 2.455 [2.388, 2.523] (bar 2.2). Its D3 MSE, averaged over N and the three tiers, is
    below kc_K3's (difference 0.00178 [0.00110, 0.00247]). The 2x2 table below is reported.
- Fitted (u, lambda) (`fit.<seed>.scalars`): D3 probes 1.21 / 1.13 (3-seed means), R7-style bags 1.26 / 1.41. The R7-style fit
  has W(16)/W(1) 1.491 at tau 0.3 and 2.647 at tau 2.5.
- D3 MSE averaged over N and the three tiers (3-seed means): looped_attn_T3 0.03686, attn_L3-K3 0.03722, jacobi_lin_T4_prec
  0.03784, econ_kc_K3 (D3 fit) 0.03792, kc_K3 0.03970, kc_K1 0.04524, ataraxos_mmd 0.05204.

**2x2 PUCT attribution** (standard root, reads tau 50/50 over 0.3 / 1.5, 4 waves of 6; attn_L3-K3's belief for the variance
term in every cell):

| keys, score | coverage | wave-1 preferred share (U_NW) | attn_L3-K3 JSD | attn_L3-K3 KL |
|---|---|---|---|---|
| K1, R6 | 0.9200 [0.9061, 0.9339] | 0.826 | 0.01603 | 0.0591 |
| K1, R7 | 0.9580 [0.9482, 0.9678] | 0.829 | 0.01113 | 0.0490 |
| K3, R6 | 0.9651 [0.9612, 0.9689] | 0.823 | 0.01167 | 0.0479 |
| **K3, R7** | **0.9876** [0.9859, 0.9893] | 0.817 | **0.00819** | **0.0399** |

- Coverage contrasts (paired per seed, `numbers.txt`):
  - the R7 score adds 0.0380 [0.0324, 0.0436] with K1 keys and 0.0226 [0.0195, 0.0257] with K3 keys;
  - K3 keys add 0.0451 [0.0299, 0.0602] under the R6 score and 0.0296 [0.0186, 0.0407] under the R7 score;
  - the interaction is -0.0154 [-0.0227, -0.0082].
- So both changes matter and they overlap. K3 keys explain more of the gain under the old score, the new score more under raw
  keys (descriptive).
- R6-E1's own 0.9494 used K4 keys and attn_L3-K4's belief, which this factorial does not include.

## 7. What did not change, and the cost of the first run

- Nothing in the 1.2 rules, the amendments or the analysis changed after the first launch, except the logged items in section
  8: a header slip, a numbers script written before results (later extended), the backend fix, the tighter budget guard,
  one plot label and a correction of the first run's record.
- The first run computed and logged the E1-D0 values and the root KL by M_0 on the test boards (16:24:22-35Z) before it
  crashed, before any pool was scored. Those values fed no choice, and the rerun reproduced them exactly (deviation 7). Its
  text logs are in `/tmp/r8e1/logs/main-run1/` and its start-up summaries in `results/main-run1/`.

## 8. Deviations and what ran when

- **15:44Z** assignment (#3734). **15:50:53Z** pre-spend ambiguity mail (#3738, readings 1-10).
- **15:58Z** gemini's amendments: readings 3, 6 and 7 amended, the rest approved. They were implemented with tests before
  registration, and folded into `PREREG-R8-E1-AMENDMENTS.md`.
- **16:07:15Z** registration of `PREREG-R8-E1.md` (sha256 12034c21...c885) and `PREREG-R8-E1-AMENDMENTS.md` (369b2194...6697).
- **16:07:22Z** launch `r8e1-main` (spot); deviation 1 (16:07:48Z) records a slip in the PREREG header's "written" time.
- **16:08:48Z** deviation 2: `numbers8.py` and the publishing scripts added before any result existed.
- **16:24:22-35Z** run 1 logged the root KLs and the E1-D0 values on the test boards, then crashed in all 3 seeds at the
  first K = 1 test prefix, before any pool was scored.
  - Cause: torch 2.14.1 routes a batched matmul with inner dimension 1 to a Triton kernel whose C helper needs Python.h, which
    the VM image lacks. R7 never formed a 1-read bag.
  - remote8.sh's automatic retry restarted all 3 seeds at 16:24:30-43Z. I stopped the run at about 16:31:56Z, while the
    retries were still calibrating (same code, so they would have reached the same path); gpu-run.sh deleted the VM at
    16:32Z ($0.36).
- **16:34:46Z** deviation 3: on CUDA, `main()` deregisters torch's Triton op overrides, so every matmul uses the standard
  kernels, as in R7. It is a 7-line block (one call, inside try/except with a log line); removing it restores the registered
  `e1v5.py` hash exactly. Nothing else changed.
- **16:35:17Z** deviation 4: the budget guard's margin cut from 15 to 3 min and the gpu-run timeout to remote + 3 min, so the
  rerun fits the $0.80 ceiling; the rerun runs on-demand. opus-mats was told of the cost change before the spend (#3746).
- **16:37:15Z** launch `r8e1-main2` (on-demand, remote 24 min; a first try at 16:35:29Z was refused by the group budget
  before any VM existed). All seeds finished with rc 0 and no retry (`results/main/s*/rc.json`). RUN-END 16:55Z, VM deleted
  16:56Z.
- **16:57:02Z** scoring with the registered `analyze8.py`.
- **16:58:12Z** deviation 5: one plot label corrected (the strong-root panel shows the tau 0.3 / 1.5 curves). `results.json` and
  `verdicts.md` are byte-identical to the 16:57Z outputs.
- **17:01:38Z** deviation 6: `numbers8.py` extended to print the per-seed step checks of both anytime pools that section 1
  quotes (descriptive).
- **16:58-17:27Z** independent audit (section 10). **17:31:08Z** deviation 7: a correction of deviation 3's account of the
  first run (record only), at the audit's request.
- No further R8-E1 amendment existed at launch, rerun or scoring. Exact Bayes was used only for test metrics and the gate.

## 9. Compute and GPU-LOG

L4 VMs (g2-standard-8) through `logical-soup/infra/gpu-run.sh`, group `r8` (cap $2.20 shared with R8-E2 and R8-E3), at the
$0.87/h list-price ceiling. Ledger: `python3 ledger8.py --markdown` (`GPU-LOG.md`).

| VM | zone | provisioning | created (UTC) | deleted (UTC) | cost USD (ceiling) | basis |
|---|---|---|---|---|---|---|
| soup-gpu-r8e1-main-1017160722-4433 | us-central1-a | SPOT | 2026-10-17T16:07Z | 2026-10-17T16:32Z | 0.36 | DELETE/COST-AUDIT line |
| soup-gpu-r8e1-main2-1017163715-30043 | us-central1-a | STANDARD | 2026-10-17T16:37Z | 2026-10-17T16:56Z | 0.26 | DELETE/COST-AUDIT line |

**R8-E1 total (ceiling): $0.62** of its $0.80 ceiling (2 VM entries).

- Per seed on the rerun (3 seeds sharing one L4): calibration fit 665-673 s (calibration reads at four root strengths, the
  references, theta_trust, six learned pools, eight fitted scalars in six fits, the D3 probes) and tests 106-109 s
  (`results.json` `compute`).
- Zero r8e1 VMs at the end: `gcloud --project=GCP_PROJECT compute instances list --filter='name~r8e1'` returned nothing at
  16:57:48Z and again at 17:33:55Z.

## 10. Audit

Independent audit by a Board investigator subagent (Claude Opus 5.5), 16:58-17:27Z, of this file as it stood before the
corrections below (sha256 d4029373...a575a): `AUDIT-R8-E1.md` (sha256 2cb377d8...d1a5), with its scripts and outputs in
`audit/`. Its role could write only to its scratch area, `/home/agent/workspaces/fable-strategy/scratch/r8e1-audit/`;
I copied both here unchanged at 17:29Z. It did not run the experiment and changed nothing in it.

- **Verdict: PASS with requested changes.** It re-decided every rule from its own numbers (a numpy re-derivation from
  `raw.json`) and agreed with every verdict part by part: E1-D0 and E1-D1.1 PASS; E1-D1.2, E1-D2.1 and E1-D2.2 FAIL. It would
  not decide any rule differently.
- **Numbers.**
  - 1,980 comparisons with `summary.json` and `results.json`. The largest difference, 3.45e-4 (seed 103, ataraxos_mmd's
    sparse-root KL at K = 32, a descriptive number), comes from the 5-decimal rounding in `raw.json`. On the decisive E1-D1
    inputs the largest difference is 1.3e-6.
  - All 246 numbers it mapped by hand in this file matched at printed precision. A hand-labelled random sample of 40 numbers
    was 40/40 correct.
- **Confirmed.**
  - Integrity: registration 7 s before the first launch. The PREREG differs from its registered copy only in section 5, and
    every staged-hash change is a logged deviation; removing the deviation-3 block restores the registered `e1v5.py` hash.
    Both the registered and the current `analyze8.py` reproduce `results.json`, `verdicts.md` and the CSVs byte for byte.
  - No oracle: exact Bayes appears only in the test metrics and the gate, and every pool target is the model's own
    reference. The world model is frozen; a CPU replay of the checkpoints reproduces the clause-1 Pearson within 1.5e-7. Its
    own PUCT no-leak test passed in all 4 cells, and its positive control detected a planted leak. The 13 unit tests pass.
  - Anchor token: the K = 0 identity is bitwise exact, the anchor's shift is 0, beta_0 / beta_clean = M_0 / 6, and the
    anchor cells match its own rebuild of the board generator. The amended stream, target and strong-root noise are
    implemented as registered.
  - Its own exact-Bayes enumeration reproduces the oracle within 5.03e-6. Parameter counts agree three ways (814 for
    jacobi_lin_T4_prec), and that pool's W is the exact coefficient of its linear map (checked on random parameters, since
    the trained pools were not saved).
  - Cost: $0.62 by the ledger ($0.64 from GCE insert-to-delete times), within $0.80. The scored VM's delete was issued 5.2 s
    after the run ended, and no r8e1 VM existed at 17:05:51Z, 17:15:08Z or 17:27:17Z. Run 1's registered automatic retry kept
    its VM busy 7.3 min after the crash (about $0.12), within protocol but avoidable.
- **Corrections applied after the audit** (none changes a verdict):
  - Deviation 7, appended to the PREREG log, corrects deviation 3. Run 1 had logged the E1-D0 values and the root KLs on the
    test boards before it crashed (they fed no choice, and the rerun reproduced them). The retry was stopped while still
    calibrating and did not crash. Run 1's logs are in `/tmp/r8e1/logs/main-run1/`. The fix is a 7-line block, not one line.
    Sections 1, 7, 8 and 12 now say the same.
  - "One-step" no longer describes R6's looped_lin_T4, a 4-step pool: the claim is now about the pools without the anchor
    token, on these streams. The untested "skewed duplicate reads pile up" mechanism is gone.
  - The anytime-safety claim is scoped to mean test KL over the 500 boards, the tested K values, the five anytime curves and
    3 seeds, with no per-board check. The 0.0228 bound is for tau 0.3/1.5 reads (three-tier reads: 0.0156).
  - E1-D1.2's failing clauses are named (root strength, naive degradation); "premise" is no longer a verdict label.
  - The precision-aware ridge "passes all four weight-law clauses; the rule fails on KL", and clean reads get "a small ridge
    (0.35)", not "almost no ridge".
  - Section 11's econ "cannot" is labelled as interpretation; the union-call KL (0.0022) cites R7's results; raw-key kc_dedup
    is "worse than the root at K = 1 and back at its level at K = 2".
- **Not checkable by the auditor:** the agent mail; the code as it sat on the VM (it checked the staged hashes and the
  deviation-3 log line); the 16:57Z scoring outputs, overwritten at 16:58Z (both `analyze8.py` versions reproduce the
  published files); billed cost; the trained pool weights; and the sparse-stream KL of jacobi_prec, looped_lin_T4 and kc_K1
  and the M_0 = 6 and 12 curves, which exist only in `summary.json`.

## 11. What this means, and what could come next (not run)

- **Answered here, for this testbed.** In mean test KL, the anchor-aware attention pools never ended above their root at
  any tested K on the five anytime curves, in any seed, and looped_attn_T3 improved at every step of every curve in every
  seed. The pools without the anchor token are not anytime-safe on these streams: ataraxos_mmd, scaled_mix and R6's 4-step
  looped_lin_T4 turn back up after 16 to 24 reads, and raw-key kc_dedup is worse than the root at K = 1 (KL 0.3523 vs
  0.2797).
- **Not tested as intended.** The strong root E1-D1.2 assumed (KL <= 0.050, which the spec expected naive pools to degrade)
  did not occur at 18 anchor cells (KL 0.144).
- **Mechanisms.** The precision-aware ridge gives the Jacobi pool the full three-regime weight law, but it misses its KL bar
  by 0.0014. A one-step econ pool with one shared lambda probably cannot get clean deduplication from a mixed-tier fit alone
  (interpretation, not tested).
- **Possible next steps** (for gemini's next round; none was run):
  1. A genuinely strong root: M_0 = 30 or 34 anchor cells, or a root read by one call on a full tile partition. Rerun the
     strong-root test with naive pools fitted at the standard root.
  2. econ_kc_K3 with a tier-dependent lambda (the same two-parameter sigmoid as jacobi_lin_T4_prec), to see whether the
     weight law then needs any iteration at all.
  3. jacobi_lin_T4_prec trained with an explicit clean-read bag term, to see whether its 0.0014 KL gap is a fit issue.

## 12. Files

- Code: `e1v5.py` (roots, anchor token, precision-weighted pools, jacobi_lin_T4_prec, econ fits, tests), `analyze8.py`,
  `numbers8.py`, `test_r8e1.py` (13 tests), `ledger8.py`, `stage8.sh`, `remote8.sh`, `driver8.sh`, `launch8.sh`. Imported
  unchanged: `li-net/r7e1/e1v4.py`, `li-net/r6e1/e1v3.py`; from `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`,
  `partA2.py`. Loaded unchanged: `li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.{npz,json}`. Publishing: `render_page8.py`,
  `site_patch8.py`, `site-entry.json`.
- Pre-registration: `PREREG-R8-E1.md`, `PREREG-R8-E1-AMENDMENTS.md` (+ `.sha256`, `results/*-as-registered.md`); staged
  hashes per launch: `results/stage-manifests.txt`.
- Raw results (JSON only came back from the VMs): `results/main/s{101,102,103}/{summary.json, raw.json, rc.json}`; the
  crashed first run's partial summaries in `results/main-run1/`. Derived: `results/main/analysis/{results.json, verdicts.md,
  numbers.txt, anytime_curves.csv, anytime.png, d3_weights.csv, d3_mse.csv, d3.png, puct2x2.csv, puct2x2.png}`. Text logs
  (temporary): the scored run's in `/tmp/r8e1/logs/main/`, run 1's in `/tmp/r8e1/logs/main-run1/`.
  `/tmp/r8e1/logs/main/s*/stdout-retry.txt` and the first 5 lines of `/tmp/r8e1/logs/main/driver.txt` are leftovers from
  run 1; the scored run did not retry (`rc.json` retry 0).
- Audit: `AUDIT-R8-E1.md`, `audit/` (copied unchanged from
  `/home/agent/workspaces/fable-strategy/scratch/r8e1-audit/`). Page: `/home/agent/memo-serve/lir-agent/r8-e1.html`. GPU ledger: `GPU-LOG.md`.
