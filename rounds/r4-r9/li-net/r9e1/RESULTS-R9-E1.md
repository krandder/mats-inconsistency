# R9-E1 (E-INFO-STRONG-ROOT): genuine strong-root anytime safety and the precision-aware crowding pool econ_prec_K3: results

eurisko study, 2026-10-17 (18:32-20:37Z). Written by an Opus 5.5 subagent for eurisko. Assignment: opus-mats, agent-mail #3753
(group `r9`, GPU_GROUP_BUDGET_USD 1.35 shared with R9-E2; this experiment's cap $0.60; 3 seeds; seed-t 95% CIs; independent
audit; exact oracles for evaluation only; no world-model training; every post-hoc edit logged when it happens; the VM deleted at
DONE). Spec: gemini, `/home/agent/workspaces/mats/outbox/experiments-round-9-spec.md`, section 0, the global protocol, section 1
and section 3. Amendments: Part B of `experiments-round-9-amendments.md` (19:00Z) approved my readings 1-4 and 7-10 (mail
#3758) and amended readings 5 and 6.
- Pre-registration: `PREREG-R9-E1.md` and `PREREG-R9-E1-AMENDMENTS.md`, both registered 19:16:38Z, before any launch. Its
  deviations log is in section 5 (1 entry; section 8 here).
- Every number below names its source file; paths are relative to `li-net/r9e1/`. Derived descriptive numbers come from the
  registered `numbers9.py`, output `results/main/analysis/numbers.txt`. Audit: `AUDIT-R9-E1.md` (section 10).

Round 8 left Kelvin's question 1 untested from a strong root: R8-E1's 18-anchor root sat at KL 0.144, and even ataraxos_mmd
still improved it. Round 9 builds the root from 30-34 of the 36 cell readings, adds reads to it in two ways, and tests whether
the pools stay at or below the root. It also gives the 1-step econ_kc pool a precision-dependent ridge.

## 1. TL;DR

**At the spec's level, 2 of 3 rules pass (E1-D0, E1-D1); E1-D2 fails. At the sub-rule level, 4 of 5 pass (E1-D0, E1-D1.1,
E1-D1.2, E1-D2.2). E1-D2.1 (econ_prec_K3) fails on two clauses: clean deduplication and the D3 MSE comparison with the
2-scalar econ_kc_K3.** Verdicts: `results/main/analysis/verdicts.md` and `results.json`; 3-seed mean [seed-t 95% CI].

| rule | verdict | decisive numbers |
|---|---|---|
| E1-D0 gate | **PASS** | each checkpoint matches its sha256 (3 seeds); world-model Pearson with exact Bayes 0.99946 [0.99938, 0.99953] (bar 0.999); at K = 0 both anchor pools return the root exactly at M_0 = 2 and 32 (KL difference 0.0, bar < 1e-7) |
| E1-D1.1 sparse-root anytime curve | **PASS** | looped_attn_T3's KL never rises by more than +0.00016 between consecutive K in any seed (bar +0.0005); KL(32) 0.0323 [0.0298, 0.0348] (bar 0.033); JSD(24) 0.00699 [0.00636, 0.00761] (bar 0.0075) |
| E1-D1.2 strong-root safety | **PASS** | M_0_strong = 32 in every seed; root KL(0) 0.0322 [0.0317, 0.0326] (bar 0.050). Mode B: mix KL(16) - KL(0) = +0.1761 [0.1755, 0.1767] and ataraxos_mmd +0.0470 [0.0385, 0.0554] (bar +0.015, CI > 0). looped_attn_T3: largest change from the root over K = 1-24, per seed, Mode A -0.00325 / -0.00294 / -0.00298, Mode B -0.00048 / +0.00003 / -0.00041 (bar +0.0005); KL(0) - KL(16) Mode A 0.0170 [0.0159, 0.0181], Mode B 0.0030 [0.0027, 0.0033]; below mmd and scaled_mix at K = 16 and 24 in both modes (all 8 CIs > 0). **The PASS rests on approved reading 1:** under the spec's literal selection order M_0_strong would be 34, where this rule fails (section 3) |
| E1-D2.1 econ_prec_K3 | **FAIL** (two clauses) | W(16)/W(1) at tau 0.3 = 1.267 [1.237, 1.297] (bar 1.25; the CI-upper clause, 1.30, passes); D3 MSE minus econ_kc_K3's: 0.000091 [-0.000090, 0.000273] (CI not > 0). Passing: W(8)/W(1) 1.243 < 1.808 < 2.389 in every seed; W(16)/W(1) at tau 2.5 = 2.659 (bar 2.20); D3 MSE below kc_K3's (0.00164 [0.00136, 0.00192]) |
| E1-D2.2 jacobi_lin_T4_prec | **PASS** | W(16)/W(1) 1.109 at tau 0.3 and 2.515 at tau 2.5; W(8)/W(1) 1.098 < 1.607 < 2.276 in every seed; P-FullSupport KL 0.0349 [0.0343, 0.0355] (bar 0.0420); lower KL than looped_lin_T4 (by 0.0315), kc_K1 (0.0601), kc_K3 (0.0298) and econ_prec_K3 (0.0092), all CIs > 0 |

- **Kelvin's question 1, from a strong root, as measured here** (mean test KL over 500 boards, the tested K, 3 seeds).
  - The root sees 32 of the 36 readings and sits at KL 0.032 from exact Bayes (R8's 18-anchor root: 0.144).
  - **Mode B** (each read is a bounded call on the 2 standard anchors plus a 6-cell patch, and its shift over that small root
    is added to the strong root): every pool without the anchor token ended above the root (K = 24).
    - mix, at once (+0.213 at K = 1); kc_K1, at once (+0.119).
    - The learned pools without the anchor, from the first read: jacobi_lin_T4_prec (+0.012 at K = 1, +0.044 at K = 16),
      econ_prec_K3 (+0.012, +0.046), kc_K3 (+0.021, +0.053), attn_L3-K1 (+0.006, +0.042).
    - The fixed-step sums ataraxos_mmd and scaled_mix and R6's 4-step looped_lin_T4 from K = 4 on (+0.047, +0.053 and
      +0.045 at K = 16).
    - The two anchor-aware attention pools stayed below the root at every K in the 3-seed mean, but gained little: -0.0030 at
      K = 16, -0.0033 at K = 24. In seed 102, looped_attn_T3 was +0.00003 above the root at K = 1.
  - **Mode A** (each read is a call on the strong root's evidence plus the patch, as in R8): in the 3-seed means every pool
    except kc_K1 stayed below the root at every K; the anchor pools are at -0.0170 at K = 16. At K = 24 they end below every
    other learned pool, by 0.0004-0.0018 against econ_prec_K3, both econ_kc_K3 fits, kc_dedup K3 and the two Jacobi pools
    (paired CIs > 0); at K = 16 their margin over econ_prec_K3 and the econ_kc_K3 fits is only 0.0002-0.0003, and 5 of those
    6 paired CIs include 0 (descriptive; `audit/check8_extra.txt`). mmd, scaled_mix and looped_lin_T4 turn back up after
    K = 16, still below the root.
  - Caveat: the anchor pools were trained on both read modes (reading 4), so their Mode B behaviour is learned from Mode B
    training bags, not a zero-shot transfer.
  - At the strongest root, M_0 = 34 (descriptive), the anchor pools stay within 0.0007 of the root in Mode B: looped_attn_T3
    is +0.0004 above it at K = 1 and -0.0004 at K = 24 (means), and in seed 102 it rises to +0.00062, beyond the rule's
    +0.0005 tolerance; attn_L3-K3 ends +0.0002 above it at K = 24 (up to +0.00070 in seed 101). Under the spec's literal
    selection order M_0 = 34 would have been the rule's root, and E1-D1.2 would fail there (section 3; the audit's
    sensitivity check).
- **econ_prec_K3 moves most of the way but not all.** Its learned ridge grows with the read's noise (3-seed means about 0.55
  for clean reads, 1.0 at tau 1.5 and 1.35 at tau 2.5). Clean W(16)/W(1) falls from 1.432 (econ_kc_K3 with one ridge, this
  run) to 1.267, against a 1.25 bar. Its D3 MSE is not distinguishable from econ_kc_K3's.
- **jacobi_lin_T4_prec passes every clause.** With gemini's three-term loss its ridge is 0.11 for clean reads, 0.73 at tau 1.5
  and 1.54 at tau 2.5 (3-seed means). It dedups clean reads (1.109) and its P-FullSupport KL is 0.0349 (R8-E1, with one
  loss on standard bags: 0.0414 against a 0.040 bar). Its KL is 0.0092 below the 1-step econ_prec_K3's.
- **Cost: $0.31** at the list-price ceiling, of the $0.60 cap: one spot L4, created 19:40Z, deleted 20:03Z, about 1 min after
  the run ended. Zero r9e1 VMs at 20:03:25Z and again at 20:38:01Z.

## 2. What ran

- **Questions (spec section 1).**
  1. Kelvin's question 1 from a genuinely strong root: as reads are added, does the belief ever end up worse than a root
     that is already close to the full-evidence belief? R8-E1's 18-anchor root sat at KL 0.144, and with R8's
     cumulative-context reads even ataraxos_mmd improved it. Gemini predicted that mix and ataraxos_mmd degrade a strong root
     when bounded local reads are added to it, and that the anchor-aware attention pools do not. (In this run at M_0 = 18,
     bounded Mode B reads let mix degrade that root at once, +0.0952 at K = 1, and mmd ends above it, +0.0031 at K = 24;
     descriptive.)
  2. Does the 1-step crowding pool econ_kc_K3 get the three-regime weight law (dedup identical clean reads, add support
     for noisy ones) once its ridge depends on each read's precision (econ_prec_K3, 3 scalars)?
  3. Does jacobi_lin_T4_prec, trained jointly on standard bags, full-support bags and the D3 probes, keep that law and beat
     the 1-step pools on partly overlapping reads?
- **World model.** No training. Each seed loads R7-E1's `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz` after checking its
  sha256 against its `.json` (669,633 parameters; the 6x6 board, ships 4/3/2, the commander's note and a dense sonar field
  with flip 0.15).
- **Roots.**
  - M_0 anchor cells = the first M_0 cells of make_boards2's own per-board cell order, so the roots are nested and M_0 = 2 is
    the R5-R8 root; ladder 2, 6, 12, 18, 30, 32, 34.
  - Strong root (reading 1, approved): per seed, on the 4,000 calibration boards only, the smallest M_0 in {30, 32, 34}
    whose root has KL(ref(36) || root) <= 0.035, ref(36) being the model's own belief given all 36 readings.
- **Reads.** A read is one theta_real call on a root's evidence plus one 2x3 patch; patch cells the root has not seen get
  their reading re-flipped by the read's tier (tau 0.3 clean, 1.5 with probability 0.127, 2.5 with 0.28).
  - Mode A (cumulative context, as R8): reads from the strong root's evidence, tau three-tier.
  - Mode B (bounded context): the same patches read from the 2-anchor root (tau 0.3 / 1.5), and their shifts over that root
    added to the strong root. Every pool except the two anchor-aware pools (trained over the 7 root strengths with half
    Mode A and half Mode B bags) uses its standard-root fit; mix averages the reads' beliefs.
- **Pools.**
  - Anytime, anchor-aware: looped_attn_T3 and attn_L3-K3 with R8's anchor token (the root as a zero-shift token with
    precision M_0 beta_clean / 6) and precision-weighted crowding, trained by BCE to the model's own reference ref(M_0)
    over the 7 root strengths, K in {0, 1, 2, 4, 8, 16, 24, 32}, half Mode A and half Mode B bags at M_0 >= 6.
  - R8's standard-root pools unchanged: attn_L3-K1, R6's looped_lin_T4, R7's jacobi_lin_T4, fitted scalars for
    ataraxos_mmd, kc_dedup (K1, K3 keys), econ_kc_K3 (R7-style and on the D3 probes), scaled_mix; mix.
  - econ_prec_K3 (amended): w_r = u beta_r / (softplus(a - b beta_r) + sum_b beta_b relu(<k_r, k_b>)^2), fitted by BCE on
    the mixed-tier D3 calibration probes.
  - jacobi_lin_T4_prec (amended loss): 0.5 x standard-bag BCE + 0.5 x calibration P-FullSupport BCE + 0.25 x D3 MSE.
- **Tests** (the 500 R5-R8 test boards; exact Bayes given all 36 readings, for scoring only):
  - the sparse-root curve on R8's interleaved 32-read P-FullSupport stream (one partition tile per block of 4, the rest
    90/10 skewed clean reads); P-FullSupport KL = KL at K = 24;
  - strong-root curves in Mode A and Mode B at M_0 = 18, 30, 32, 34 (24 VOI-drawn 90/10 reads per half);
  - controlled bags (N same-patch reads at three tiers plus 8 background reads).
- **Seeds** 101, 102, 103 on one L4. Statistics: 3-seed means with seed-t 95% intervals (t = 4.303); ratios are means of
  per-seed ratios; comparisons paired per seed.

## 3. E1-D0: checkpoints, world model, K = 0 identity (PASS)

Source: `results.json` (`verdicts.E1-D0`), `numbers.txt`.

- The three checkpoints load with their sha256 matching their `.json` (aef34272..., abb0fffc..., f9842c1e...).
- Clause-1 Pearson of theta_real with exact Bayes on the standard 6-cell evidence: 0.99946 [0.99938, 0.99953] (per seed
  0.99942 / 0.99948 / 0.99946).
- At K = 0, looped_attn_T3 and attn_L3-K3 return the root exactly: largest |KL difference| 0.0 and largest |logit difference|
  0.0, at M_0 = 2 and M_0 = 32 (the rule) and at every other M_0 (reported).
- Strong-root selection (calibration boards only, `fit.<seed>.strong_selection`): KL(ref(36) || root) at M_0 = 30 / 32 / 34 is
  0.04527 / 0.03005 / 0.01483 (seed 101), 0.04518 / 0.03000 / 0.01476 (102) and 0.04539 / 0.03014 / 0.01482 (103), so
  M_0_strong = 32 in every seed. The other order, KL(root || ref(36)), is 0.0354 / 0.0353 / 0.0355 at M_0 = 32 (reported).
  - **Sensitivity (audit).** Under that order, the spec's literal KL_cal(root || ref(36)), every seed is above 0.035 at
    M_0 = 32, so M_0_strong would be 34. At M_0 = 34, E1-D1.2 fails two parts in Mode B: looped_attn_T3 rises +0.00062 in
    seed 102 (bar +0.0005), and KL(0) - KL(16) = 0.0003 [-0.0004, 0.0010] (per seed 0.00035 / -0.00003 / 0.00056), so its CI
    is not above 0 (`audit/check8_extra.txt`). The PASS therefore rests on reading 1 (reference first, the direction of every
    test KL), which was chosen on calibration boards only and approved by gemini before launch. At M_0 = 30 the rule would
    pass.
- Test root KL to exact Bayes by M_0 (`stats.kl_root`): 2: 0.27966; 6: 0.24520; 12: 0.19299; 18: 0.14444; 30: 0.04618; 32:
  0.03219; 34: 0.01687.

## 4. E1-D1.1: the sparse-root anytime curve (PASS)

Source: `results.json` (`stats.fs_kl_*`, `verdicts.E1-D1.1`), figure `results/main/analysis/anytime.png` (data
`anytime_curves.csv`), `numbers.txt`. R8's interleaved P-FullSupport stream from the standard root (2 anchor readings): KL to
exact Bayes by K, 3-seed means.

| pool | K = 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| looped_attn_T3 (anchor) | 0.2797 | 0.2309 | 0.1918 | 0.1531 | 0.1066 | 0.0583 | 0.0326 | 0.0323 |
| attn_L3-K3 (anchor) | 0.2797 | 0.2310 | 0.1919 | 0.1531 | 0.1065 | 0.0589 | 0.0332 | 0.0332 |
| jacobi_lin_T4_prec | 0.2797 | 0.2309 | 0.1920 | 0.1542 | 0.1092 | 0.0616 | 0.0349 | 0.0339 |
| jacobi_lin_T4 (R7) | 0.2797 | 0.2315 | 0.1929 | 0.1546 | 0.1105 | 0.0653 | 0.0406 | 0.0402 |
| econ_prec_K3 | 0.2797 | 0.2302 | 0.1911 | 0.1544 | 0.1121 | 0.0684 | 0.0441 | 0.0433 |
| econ_kc_K3 (D3 fit) | 0.2797 | 0.2308 | 0.1917 | 0.1545 | 0.1123 | 0.0692 | 0.0452 | 0.0445 |
| kc_dedup K3 | 0.2797 | 0.2365 | 0.2012 | 0.1688 | 0.1306 | 0.0884 | 0.0647 | 0.0642 |
| kc_dedup K1 | 0.2797 | 0.3567 | 0.2796 | 0.2095 | 0.1509 | 0.1108 | 0.0950 | 0.0933 |
| looped_lin_T4 (R6) | 0.2797 | 0.2520 | 0.2269 | 0.1863 | 0.1270 | 0.0721 | 0.0664 | 0.1106 |
| scaled_mix | 0.2797 | 0.2581 | 0.2366 | 0.2018 | 0.1529 | 0.1085 | 0.1042 | 0.1271 |
| ataraxos_mmd | 0.2797 | 0.2538 | 0.2296 | 0.1922 | 0.1428 | 0.1080 | 0.1178 | 0.1591 |
| mix | 0.2797 | 0.2299 | 0.2115 | 0.2042 | 0.2005 | 0.1981 | 0.1973 | 0.1971 |

- **E1-D1.1 PASS.**
  - looped_attn_T3's KL falls at every step except one: from K = 24 to 32 in seed 101 it rises by +0.00016, within the
    +0.0005 tolerance (`step_increase_per_seed`; the other seeds fall by 0.00032 and 0.00072 at that step).
  - KL(32) = 0.0323 [0.0298, 0.0348] (per seed 0.0332 / 0.0325 / 0.0312; bar 0.033 on the mean). Seed 101 alone is above
    0.033; the PASS uses the registered mean reading.
  - JSD(24) between the U_NW and U_SE beliefs = 0.00699 [0.00636, 0.00761] (bar 0.0075 on the mean).
- Compared with R8-E1 (descriptive; `li-net/r8e1/results/main/analysis/results.json`): KL(32) 0.0303 there, 0.0323 here,
  and JSD(24) 0.00575 there, 0.00699 here. Here the anytime pools are trained over 7 root strengths and two read modes
  instead of 4 strengths and one mode, so the sparse root gets about a seventh of the training bags instead of a quarter
  (interpretation, not tested).
- attn_L3-K3 rises by +0.00037 from K = 24 to 32 in seed 102 (`numbers.txt`, per-seed step checks; descriptive).
- Some pools without the anchor token turn back up on this stream (descriptive): ataraxos_mmd after K = 16, scaled_mix and
  R6's looped_lin_T4 after K = 24; kc_dedup with K1 keys is worse than the root at K = 1 (0.3567 vs 0.2797).

## 5. E1-D1.2: genuine strong-root safety (PASS)

Source: `results.json` (`verdicts.E1-D1.2`, `stats.srS*`, `stats.srA*`, `stats.srB*`), figure `anytime.png`, `numbers.txt`.
Each seed at its own M_0_strong (32 in every seed). 24 reads per half, VOI-drawn from the strong root's belief, skewed 90/10;
the same patches in both modes. KL(K) - KL(0), 3-seed means; the root is at KL(0) = 0.0322.

**Mode A** (cumulative context: reads from the strong root's evidence, tau three-tier):

| pool | K = 1 | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| looped_attn_T3 (anchor) | -0.0031 | -0.0057 | -0.0091 | -0.0129 | -0.0170 | -0.0189 |
| attn_L3-K3 (anchor) | -0.0030 | -0.0056 | -0.0091 | -0.0129 | -0.0171 | -0.0190 |
| jacobi_lin_T4_prec | -0.0028 | -0.0054 | -0.0086 | -0.0122 | -0.0160 | -0.0176 |
| econ_prec_K3 | -0.0029 | -0.0056 | -0.0091 | -0.0129 | -0.0168 | -0.0184 |
| kc_dedup K3 | -0.0009 | -0.0032 | -0.0070 | -0.0116 | -0.0165 | -0.0184 |
| attn_L3-K1 | -0.0031 | -0.0053 | -0.0077 | -0.0098 | -0.0118 | -0.0123 |
| looped_lin_T4 (R6) | -0.0022 | -0.0043 | -0.0076 | -0.0118 | -0.0135 | -0.0073 |
| scaled_mix | -0.0019 | -0.0037 | -0.0067 | -0.0108 | -0.0133 | -0.0092 |
| ataraxos_mmd | -0.0021 | -0.0040 | -0.0073 | -0.0112 | -0.0124 | -0.0060 |
| mix | -0.0025 | -0.0055 | -0.0068 | -0.0074 | -0.0078 | -0.0080 |
| kc_dedup K1 | +0.0288 | +0.0149 | +0.0022 | -0.0066 | -0.0119 | -0.0134 |

**Mode B** (bounded context: reads on the 2 standard anchors plus the patch, tau 0.3 / 1.5, their shift over the 2-anchor root
added to the strong root; mix averages the reads' beliefs):

| pool | K = 1 | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| looped_attn_T3 (anchor) | -0.0003 | -0.0008 | -0.0015 | -0.0022 | -0.0030 | -0.0033 |
| attn_L3-K3 (anchor) | -0.0005 | -0.0010 | -0.0017 | -0.0024 | -0.0030 | -0.0033 |
| jacobi_lin_T4_prec | +0.0123 | +0.0198 | +0.0290 | +0.0374 | +0.0437 | +0.0460 |
| econ_prec_K3 | +0.0123 | +0.0201 | +0.0296 | +0.0387 | +0.0458 | +0.0485 |
| kc_dedup K3 | +0.0212 | +0.0318 | +0.0416 | +0.0490 | +0.0530 | +0.0539 |
| attn_L3-K1 | +0.0060 | +0.0138 | +0.0257 | +0.0365 | +0.0416 | +0.0438 |
| looped_lin_T4 (R6) | -0.0005 | -0.0002 | +0.0023 | +0.0125 | +0.0452 | +0.0877 |
| scaled_mix | -0.0005 | -0.0002 | +0.0024 | +0.0139 | +0.0530 | +0.1041 |
| ataraxos_mmd | -0.0005 | -0.0004 | +0.0017 | +0.0118 | +0.0470 | +0.0939 |
| mix | +0.2125 | +0.1931 | +0.1838 | +0.1791 | +0.1761 | +0.1754 |
| kc_dedup K1 | +0.1193 | +0.1036 | +0.0847 | +0.0699 | +0.0606 | +0.0575 |

- **E1-D1.2 PASS, all 15 parts.**
  1. Strong root: KL(0) = 0.0322 [0.0317, 0.0326] (per seed 0.0323 / 0.0320 / 0.0323; bar 0.050).
  2. Mode B, naive pooling degrades it at K = 16: mix +0.1761 [0.1755, 0.1767], ataraxos_mmd +0.0470 [0.0385, 0.0554] (bar
     +0.015 with CI > 0). In Mode A neither degrades it (mix -0.0078 [-0.0083, -0.0073], mmd -0.0124 [-0.0132, -0.0117];
     reported).
  3. looped_attn_T3 in both modes:
     - largest KL(K) - KL(0) over K = 1-24, per seed: Mode A -0.00325 / -0.00294 / -0.00298, Mode B -0.00048 / +0.00003 /
       -0.00041 (bar +0.0005 in every seed);
     - KL(0) - KL(16): Mode A 0.0170 [0.0159, 0.0181], Mode B 0.0030 [0.0027, 0.0033] (CI > 0);
     - KL(mmd) - KL(looped_attn) at K = 16 / 24: Mode A 0.0046 [0.0037, 0.0055] / 0.0130 [0.0114, 0.0145], Mode B 0.0500
       [0.0415, 0.0584] / 0.0972 [0.0822, 0.1123]; KL(scaled_mix) - KL(looped_attn): Mode A 0.0037 [0.0027, 0.0047] / 0.0097
       [0.0080, 0.0114], Mode B 0.0560 [0.0548, 0.0573] / 0.1075 [0.1061, 0.1088] (all CI > 0).
- attn_L3-K3 (reported): largest change from the root per seed, Mode A -0.0031 / -0.0030 / -0.0030, Mode B -0.00056 /
  -0.00020 / -0.00059.
- Other root strengths (descriptive; KL(K) - KL(0) at K = 1 / 16 / 24, 3-seed means, `numbers.txt`):
  - M_0 = 30 (KL(0) 0.0462), Mode B: looped_attn_T3 -0.0012 / -0.0065 / -0.0072; mmd -0.0016 / +0.0380 / +0.0831;
    jacobi_lin_T4_prec +0.0098 / +0.0331 / +0.0347; mix +0.1987 / +0.1616 / +0.1609.
  - M_0 = 34 (KL(0) 0.0169), Mode B: looped_attn_T3 +0.0004 / -0.0003 / -0.0004 (largest per-seed rise +0.00062, seed 102);
    attn_L3-K3 +0.0003 / +0.0000 / +0.0002 (largest +0.00070, seed 101); mmd +0.0006 / +0.0566 / +0.1057.
  - M_0 = 18 (KL(0) 0.1444), Mode B: looped_attn_T3 -0.0130 / -0.0561 / -0.0623; mmd -0.0088 / -0.0286 / +0.0031; mix
    +0.0952 / +0.0608 / +0.0601.
  - In Mode A, looped_attn_T3, attn_L3-K3, mmd, mix, jacobi_lin_T4_prec and econ_prec_K3 are below the root at K = 16 and
    24 at each of these root strengths.
- Interpretation (not tested): a bounded read re-reads cells the strong root has already seen, and its shift over the
  2-anchor root repeats evidence the strong root already holds. Pools that add such shifts with weights fitted at the
  standard root count that evidence twice; the anchor token lets the attention pools discount reads that overlap the root's
  footprint. No ablation separates the anchor token from the Mode B training bags.

## 6. E1-D2.1: econ_prec_K3 on the controlled bags (FAIL on two clauses)

Source: `results.json` (`verdicts.E1-D2.1`, `stats.weights`, `stats.mse_all_tiers`, `fit.<seed>.econ_prec`), figure
`results/main/analysis/d3.png` (data `d3_weights.csv`, `d3_mse.csv`), `numbers.txt`. W(N) = the sum over the N same-patch reads
of a pool's per-read weight averaged over A*'s 6 cells (for the Jacobi pools, the exact coefficient of their linear map), as R7
and R8; 8 background reads in every bag.

| pool, tier | W(1) | W(2) | W(4) | W(8) | W(16) | W(8)/W(1) | W(16)/W(1) |
|---|---|---|---|---|---|---|---|
| econ_prec_K3, tau 0.3 | 0.873 | 0.980 | 1.047 | 1.085 | 1.106 | 1.243 | 1.267 [1.237, 1.297] |
| econ_prec_K3, tau 1.5 | 0.685 | 0.916 | 1.107 | 1.238 | 1.319 | 1.808 | 1.926 [1.888, 1.964] |
| econ_prec_K3, tau 2.5 | 0.526 | 0.786 | 1.047 | 1.256 | 1.398 | 2.389 | 2.659 [2.616, 2.702] |
| jacobi_lin_T4_prec, tau 0.3 | 0.869 | 0.913 | 0.939 | 0.954 | 0.963 | 1.098 | 1.109 [1.083, 1.134] |
| jacobi_lin_T4_prec, tau 1.5 | 0.686 | 0.869 | 1.008 | 1.101 | 1.156 | 1.607 | 1.686 [1.653, 1.720] |
| jacobi_lin_T4_prec, tau 2.5 | 0.486 | 0.713 | 0.934 | 1.106 | 1.222 | 2.276 | 2.515 [2.335, 2.694] |
| econ_kc_K3 (D3 fit), tau 0.3 | 0.801 | 0.953 | 1.054 | 1.115 | 1.148 | 1.391 | 1.432 [1.411, 1.454] |
| econ_kc_K3 (D3 fit), tau 1.5 | 0.680 | 0.926 | 1.135 | 1.283 | 1.375 | 1.888 | 2.023 [1.949, 2.097] |
| econ_kc_K3 (D3 fit), tau 2.5 | 0.570 | 0.844 | 1.113 | 1.328 | 1.473 | 2.332 | 2.586 [2.401, 2.771] |
| kc_dedup K3, tau 0.3 | 1.245 | 1.312 | 1.352 | 1.373 | 1.385 | 1.103 | 1.112 [1.076, 1.148] |
| kc_dedup K3, tau 1.5 | 0.797 | 0.930 | 1.019 | 1.069 | 1.100 | 1.342 | 1.380 [1.323, 1.437] |
| kc_dedup K3, tau 2.5 | 0.550 | 0.677 | 0.767 | 0.821 | 0.852 | 1.492 | 1.550 [1.482, 1.618] |

- **E1-D2.1 FAIL.**
  - Failing: W(16)/W(1) at tau 0.3 = 1.267 [1.237, 1.297] (per seed 1.255 / 1.267 / 1.279; bar 1.25 on the mean). D3 MSE
    (averaged over N and the three tiers) minus econ_kc_K3's (D3 fit): 0.000091 [-0.000090, 0.000273], so its CI is not
    above 0.
  - Passing: CI upper bound of W(16)/W(1) at tau 0.3, 1.297 (bar 1.30); W(8)/W(1) 1.243 < 1.808 < 2.389 across the tiers in
    every seed; W(16)/W(1) at tau 2.5 = 2.659 [2.616, 2.702] (bar 2.20); D3 MSE below kc_K3's by 0.00164 [0.00136,
    0.00192].
- Fitted scalars (`fit.<seed>.econ_prec`): u = 1.113 / 1.134 / 1.135, a = 2.613 / 2.390 / 1.548, b = 0.922 / 0.691 / 0.512 per
  seed. The ridge softplus(a - b beta) at each tier's mean beta is 0.43 / 0.61 / 0.61 (tau 0.3), 0.97 / 1.13 / 0.95 (tau 1.5)
  and 1.38 / 1.49 / 1.17 (tau 2.5). The amendment removed the 2.0 ceiling. At each tier's mean beta the fitted ridge stays
  below 2; it exceeds 2 only for reads with beta below 0.82 (seed 101) or 0.78 (seed 102), where lambda(0) = 2.68 and 2.48.
  How many test reads have such a low beta was not saved.
- For comparison (same run): econ_kc_K3 with one ridge (D3 fit, lambda 1.13 / 1.34 / 1.22) has W(16)/W(1) 1.432 at tau 0.3
  and 2.586 at tau 2.5. The precision-dependent ridge takes clean deduplication from 1.432 to 1.267 and keeps the noisy
  rise (2.659) (same run, same keys and D3 probes; paired difference in W(16)/W(1) at tau 0.3 0.166 [0.155, 0.176]; at
  tau 2.5, econ_prec_K3 minus econ_kc_K3 = +0.073 [-0.152, 0.297]; descriptive, not a registered test;
  `audit/check8_extra.txt`).
- D3 MSE per tier minus econ_prec_K3's (`verdicts.E1-D2.1.stats.per_tier_mse_diff`): econ_kc_K3 0.00016 [0.00003, 0.00028] at
  tau 0.3, 0.00004 [-0.00018, 0.00026] at tau 1.5 and 0.00007 [-0.00013, 0.00028] at tau 2.5 (reported).

## 7. E1-D2.2: jacobi_lin_T4_prec (PASS)

Source: `results.json` (`verdicts.E1-D2.2`, `stats.fs_kl_*_K24`, `fit.<seed>.jacobi_prec`), `d3.png`, `numbers.txt`.

- **E1-D2.2 PASS, all 8 parts.**
  - Weight law: W(16)/W(1) at tau 0.3 = 1.109 [1.083, 1.134] (bar 1.25); W(8)/W(1) 1.098 < 1.607 < 2.276 across the tiers in
    every seed; W(16)/W(1) at tau 2.5 = 2.515 [2.335, 2.694] (bar 2.20).
  - P-FullSupport KL (the interleaved stream's first 24 reads) = 0.0349 [0.0343, 0.0355] (per seed 0.0346 / 0.0350 / 0.0351;
    bar 0.0420).
  - KL minus jacobi_lin_T4_prec's at K = 24: looped_lin_T4 0.0315 [0.0263, 0.0367], kc_K1 0.0601 [0.0555, 0.0647], kc_K3
    0.0298 [0.0282, 0.0314], econ_prec_K3 0.0092 [0.0080, 0.0103] (all CI > 0; lower in every seed).
- Learned ridge (`fit.<seed>.jacobi_prec`): a = 2.70 / 2.55 / 2.19, b = 1.75 / 1.61 / 1.61; at each tier's mean beta 0.112 /
  0.106 / 0.124 (tau 0.3), 0.733 / 0.744 / 0.704 (tau 1.5), 1.576 / 1.576 / 1.460 (tau 2.5).
- Descriptive: R7's jacobi_lin_T4 in this run has P-FullSupport KL 0.0406 and W(16)/W(1) 1.459 at tau 0.3. The anchor pools
  are lower still at K = 24 (looped_attn_T3 0.0326, attn_L3-K3 0.0332).
- R8-E1's jacobi_lin_T4_prec (one loss on R8's standard bags; `li-net/r8e1/results/main/analysis/results.json`) had KL
  0.0414 and W(16)/W(1) 1.164 / 2.816. Here the loss adds calibration P-FullSupport bags and the D3 term (gemini's amendment
  to reading 6); the clean ridge is smaller (0.11 vs R8's 0.35) and so is the noisy one (1.54 vs 2.2) (descriptive).

## 8. Deviations and what ran when

- **18:32:25Z** assignment (#3753). **18:38-18:40Z** CPU check of the calibration-board root KL at M_0 = 18 / 30 / 32 / 34
  (real checkpoints, calibration boards only). **18:42:49Z** pre-spend readings mail (#3758, readings 1-10).
- **18:51:53Z** a CPU smoke on the real checkpoints, stopped at 18:52:55Z while still building its calibration pool, so that no
  test-board number existed before registration (it had logged only the calibration-board selection). **18:53-19:16Z** CPU
  smokes with untrained, randomly initialised full-size checkpoints (`/tmp/r9e1/smoke-ckpt`), `analyze9.py` and `numbers9.py`
  on their summaries; 10 unit tests pass.
- **19:00Z** gemini's amendments (Part B): readings 1-4 and 7-10 approved; readings 5 and 6 amended. **19:07:09Z** folded into
  `PREREG-R9-E1-AMENDMENTS.md`; confirmed to opus-mats with no cost or ETA change (#3765).
- **19:16:38Z** registration of `PREREG-R9-E1.md` (sha256 3a1db6d1...de11) and `PREREG-R9-E1-AMENDMENTS.md` (232df930...f50).
- **19:16:46Z** first launch try: gpu-run.sh refused `driver9.sh` as not executable, before any VM existed. **19:19:05Z**
  deviation 1: `chmod +x` on the four `.sh` files (contents and hashes unchanged).
- **19:19:14Z** second try refused by gpu-up.sh before any VM existed: group r9 held R9-E2's $0.76 reservation, and this VM's
  $0.97 would have passed the $1.35 cap. I waited for R9-E2's VM to be deleted (19:40Z).
- **19:40:18Z** launch `r9e1-main` (spot L4). The staged hashes of all three tries equal the registered ones
  (`results/stage-manifests.txt`). VM ready 19:41Z; seeds started 19:45:30Z after the pip install; all three ended with rc 0
  and no retry (`results/main/s*/rc.json`); remote rc 0 at 20:02:15Z; VM deleted 20:03Z.
- **20:03:25Z** no r9e1 VM listed; no new amendment file. **20:03:28Z** scoring with the registered `analyze9.py`; then the
  registered `numbers9.py`. Nothing in the rules, the code or the analysis changed after registration (deviation 1 changed
  only file modes).
- Exact Bayes was used only for test metrics and the gate.

## 9. Compute and GPU-LOG

L4 VM (g2-standard-8) through `logical-soup/infra/gpu-run.sh`, group `r9`, at the $0.87/h list-price ceiling. Ledger:
`python3 ledger9.py --markdown` (`GPU-LOG.md`).

| VM | zone | provisioning | created (UTC) | deleted (UTC) | cost USD (ceiling) | basis |
|---|---|---|---|---|---|---|
| soup-gpu-r9e1-main-1017194018-32101 | us-central1-a | SPOT | 2026-10-17T19:40Z | 2026-10-17T20:03Z | 0.31 | DELETE/COST-AUDIT line |

**R9-E1 total (ceiling): $0.31** of its $0.60 cap (1 VM). The spot VM's real price is lower.

- Per seed (3 seeds sharing one L4, `results.json` `compute`): calibration fit 901-918 s (the strong-root selection, the
  standard pool and reference, read pools at six root strengths, theta_trust, the anytime pools, R8's pools and scalars, the
  D3 probes, econ_prec_K3, the calibration P-FullSupport bags and jacobi_lin_T4_prec) and tests 72-74 s.
- Zero r9e1 VMs at the end: `gcloud --project=GCP_PROJECT compute instances list --filter='name~r9e1'` returned nothing at
  20:03:25Z and again at 20:38:01Z.

## 10. Audit

Independent audit by a Board investigator subagent (Claude Opus 5.5), 20:04-20:34Z, of this file as it stood before the
corrections below (sha256 12008db5...ceb6): `AUDIT-R9-E1.md` (sha256 b0a5e882...16fb), with its scripts and outputs in `audit/`.
Its role could write only to its scratch area, `/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/`; I copied
its final file and folder here unchanged at 20:36Z. It did not run the experiment and changed nothing in it.

- **Verdict: CONFIRMED, with changes to RESULTS.** It re-decided every rule from its own numbers (a numpy re-derivation from
  `raw.json`) and got the same verdicts: E1-D0, E1-D1.1, E1-D1.2 and E1-D2.2 PASS; E1-D2.1 FAIL on two clauses (E1-D1 PASS,
  E1-D2 FAIL). It would decide no rule differently.
- **Numbers.**
  - The largest gap between its re-derivation and `summary.json` is 2.7e-4 (seed 103's sparse-root mmd KL at K = 32, a number
    no rule uses); on the headline numbers it is 8.7e-5. All 348 per-seed KLs lie inside the intervals that the 5-decimal
    rounding of `raw.json` allows.
  - All 312 table cells in this file match `results.json` at printed precision. One prose number was rounded twice
    (jacobi_lin_T4_prec's a in seed 103: 2.19, printed 2.20); fixed.
- **Confirmed.**
  - Integrity: registration (19:16:38Z) precedes the first launch-log line (19:16:46Z) and the first r9e1 GPU-LOG line
    (19:40Z); all registered and staged hashes match; the only change after registration is deviation 1 (file modes).
    The registered `analyze9.py` and `numbers9.py` reproduce `verdicts.md`, `numbers.txt`, the CSVs and `results.json`
    exactly.
  - No oracle and no world-model training: exact Bayes appears only in the test protocol's scoring and the gate; the
    strong-root selection uses calibration boards and the model's own ref(36). The amended items (softplus ridge,
    three-term loss) and the approved readings (Mode A, Mode B, anytime training with both modes) are implemented as the
    PREREG states.
  - Parameters: 814 (jacobi_lin_T4_prec with its head), 3 (econ_prec_K3, which also uses attn_L3-K3's 592-parameter K3 key
    head, uncounted as for econ_kc_K3), 39,428 and 106,180 (the anchor pools); econ_prec_K3's closed form on identical reads.
  - Cost: $0.31 by GPU-LOG, at most $0.33 from the GCE operation times, within $0.60; the VM's deletion began 7 s after the run
    ended and finished 57 s after it; no r9e1 VM at 20:09:08Z, 20:28:41Z or 20:33:14Z.
- **Sensitivities it reported** (no rule decided differently): E1-D1.2 fails if M_0_strong is chosen by the spec's literal KL
  order (M_0 = 34); E1-D1.1 would fail if KL(32) <= 0.033 had to hold in every seed (seed 101: 0.0332); E1-D2.1's FAIL holds
  against either 2-scalar econ_kc_K3 fit.
- **Corrections applied after the audit** (none changes a verdict):
  - The E1-D1.2 PASS is now stated as resting on approved reading 1, with the M_0 = 34 failure (sections 1, 3 and 11).
  - Mode A's "anchor pools gained the most" is replaced by the paired margins (clear at K = 24, mostly within noise at K = 16).
  - Section 2 no longer says the 18-anchor root was "too weak to degrade"; it says R8's cumulative-context reads improved it,
    and that bounded reads do degrade it here.
  - Mode B's description says the anchor pools are trained, not standard-root fits.
  - The ridge-ceiling sentence is scoped to each tier's mean beta; the econ ridge comparison gives its paired CIs and is
    labelled descriptive, here and in section 11.
  - Section 11's Mode B summary is scoped to the 3-seed means; the per-seed KL(32) of seed 101 is flagged; the auditor's
    background-read calculation is added to next step 3; seed 103's a is corrected.
- **Not checkable by the auditor:** real billing; the pre-registration CPU check and the stopped real-checkpoint smoke (no logs
  left in `/tmp/r9e1`); the mail; the code as it ran on the VM (it checked the staged hashes); the clause-1 Pearson and the
  logit-level K = 0 identity (not in `raw.json`); the trained pools, which were not saved.

## 11. What this means, and what could come next (not run)

- **Answered here, for this testbed** (mean test KL, 3 seeds). From a root that already holds 32 of 36 readings, bounded reads
  left every pool without the anchor token above the root by K = 24: mix, the fixed-step sums, R6's looped_lin_T4, kc_dedup,
  and also the learned precision-aware pools (jacobi_lin_T4_prec, econ_prec_K3). The anchor-aware attention pools, trained on
  such bags, stayed just below the root in the 3-seed means (seed 102: looped_attn_T3 +0.00003 above it at K = 1). This
  PASS depends on reading 1: at M_0 = 34, the root the spec's literal selection order picks, the rule fails (section 3).
  With cumulative-context reads every pool except kc_K1 stayed below the root at every
  K.
- **How safe is "safe".** In Mode B the anchor pools gain only 0.003 nats at M_0 = 32 and are within +0.0007 of the root at
  M_0 = 34. That is consistent with discounting redundant reads, not with extracting much from them (interpretation).
- **Mechanisms** (descriptive: a same-run comparison with econ_kc_K3, not an ablation). A precision-dependent ridge brings
  the 1-step econ pool most of the way to clean deduplication (1.267 vs 1.25) without losing the noisy rise; the 4-step jacobi_lin_T4_prec gets the whole weight law and a lower P-FullSupport KL.
- **Possible next steps** (for gemini's next round; none was run):
  1. An ablation of the anchor pools' Mode B safety: the same pools trained on Mode A bags only, to separate the anchor token
     from the Mode B training bags.
  2. A bounded-read pool that knows the root's footprint without attention: econ_prec_K3 or jacobi_lin_T4_prec with the
     anchor token's crowding term (beta_0 relu(<k_r, k_0>)^2) added to each read's denominator.
  3. econ_prec_K3 fitted with a clean-read bag term, to see whether its 0.017 gap to the 1.25 bar is a fit issue. The
     auditor's calculation points elsewhere: on identical clean reads with no background reads, the fitted scalars give
     W(16)/W(1) = 1.116 / 1.151 / 1.171, and the D3 bags' 8 background reads raise each board's intercept
     (lambda + C_bg) / (u beta) 1.6-2.2-fold, which lifts the measured ratio to 1.255-1.279. The gap may come from background
     crowding rather than the fit (`AUDIT-R9-E1.md`, requested change 9).

## 12. Files

- Code: `e1v6.py` (strong-root ladder and selection, Mode A / Mode B tests, anytime training with both modes, econ_prec_K3,
  jacobi_lin_T4_prec's three-term loss), `analyze9.py`, `numbers9.py`, `test_r9e1.py` (10 tests), `ledger9.py`, `stage9.sh`,
  `remote9.sh`, `driver9.sh`, `launch9.sh`. Imported unchanged: `li-net/r8e1/e1v5.py`, `li-net/r7e1/e1v4.py`,
  `li-net/r6e1/e1v3.py`; from `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`, `partA2.py`. Loaded unchanged:
  `li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.{npz,json}`. Publishing: `render_page9.py`, `site_patch9.py`,
  `site-entry.json`.
- Pre-registration: `PREREG-R9-E1.md`, `PREREG-R9-E1-AMENDMENTS.md` (+ `.sha256`, `results/*-as-registered.md`); staged hashes
  per launch: `results/stage-manifests.txt`.
- Raw results (JSON only came back from the VM): `results/main/s{101,102,103}/{summary.json, raw.json, rc.json}`. Derived:
  `results/main/analysis/{results.json, verdicts.md, numbers.txt, anytime_curves.csv, anytime.png, d3_weights.csv, d3_mse.csv,
  d3.png}`. Text logs (temporary): `/tmp/r9e1/logs/main/` and the launch log `/tmp/r9e1/logs/launch-r9e1-main.log`.
- Audit: `AUDIT-R9-E1.md`, `audit/`. Page: `/home/agent/memo-serve/lir-agent/r9-e1.html`. GPU ledger: `GPU-LOG.md`.

