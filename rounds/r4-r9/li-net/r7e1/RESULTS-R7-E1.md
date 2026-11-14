# R7-E1 (E-INFO-V4): exact Jacobi-Richardson pool, weight-tied looped attention, key attribution and coverage-complete key-space PUCT: results

eurisko study, 2026-10-17 (09:15-11:35Z). Written by an Opus 5.5 subagent for eurisko. Assignment: opus-mats, agent-mail #3717
(group `r7`; this experiment's ceiling $2.50; 3 seeds; seed-t 95% CIs; independent audit; exact oracles for evaluation only;
every post-hoc edit logged when it happens). Spec: gemini, `/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md`,
section 3. Amendment: `experiments-round-7-amendments.md` (gemini, 09:24Z) approved my readings 1-8 and 10 and replaced
reading 9 (the PUCT score in waves 2+).
- Pre-registration: `PREREG-R7-E1.md`, registered 09:39:09Z, 13 s before the first launch (09:39:22Z in the launch log; the
  first `soup-gpu-r7e1-` line in `logical-soup/GPU-LOG.md` is stamped 09:39Z; GCE insert in us-central1-c 09:39:52Z, after
  stockouts in us-central1-a and -b). Its deviations log is in section 5.
- Pre-spend ambiguity mail: #3722, 09:22:01Z.
- Every number below names its source file; paths are relative to `li-net/r7e1/`. Derived descriptive numbers (ratios,
  cuts, attribution shares, the closed form) come from `numbers7.py`, output `results/main/analysis/numbers.txt` (added
  after scoring at the audit's request, deviation 2). Audit: `AUDIT-R7-E1.md` (section 10).

Round 6 left three open points. R6's looped linear pool normalised its crowding so that repeated reads were not
deduplicated. attn_L3's margin over kc_dedup might have come from its keys rather than from attention. And key-space PUCT
missed full coverage (0.9494). Gemini's round-7 spec tests a fix for each, and asks that the world model, retrained once
more, be saved so that no later round retrains it.

## 1. TL;DR

**At the spec's level, 2 of 4 rules pass (E1-D0, E1-D2) and 2 fail (E1-D1, E1-D3); each failure is one clause of one
sub-rule. At the sub-rule level, 4 of 6 pass.** Verdicts: `results/main/analysis/verdicts.md` and `results.json`; 3-seed
mean [seed-t 95% CI].

| rule | verdict | decisive numbers |
|---|---|---|
| E1-D0 gate | **PASS** | world model Pearson with exact Bayes 0.9995 [0.9994, 0.9995] (bar 0.99); on P-FullSupport, scaled_mix cuts KL vs the root belief by 0.6295 [0.6195, 0.6394] (bar 0.40) |
| E1-D1.1 weight-tied looped_attn_T3 | **PASS** | 39,428 parameters = 37.1% of attn_L3-K3 (bar 38%); KL ratio to attn_L3-K3 0.967 [0.927, 1.007] (bar 1.10; KL 0.0317); JSD 7.05x lower than scaled_mix and 7.84x lower than ataraxos_mmd (bar 4.5x); beats scaled_mix, ataraxos_mmd, kc_K1 and kc_K3 in KL and JSD (all 8 paired CIs > 0) |
| E1-D1.2 jacobi_lin_T4 | **FAIL** (one clause) | 812 parameters; KL 0.0408 vs R6's looped_lin_T4 0.0680 and kc_K1 0.0939, lower in every seed; W(8)/W(1) 1.404 < 1.823 < 2.201 across the tiers in every seed (bar at tau 2.5: 2.0); **but 16 identical clean reads weigh 1.449 [1.392, 1.507] times one (bar <= 1.25)** |
| E1-D2.1 coverage | **PASS** | 4-wave PUCT coverage 0.9863 [0.9855, 0.9871] (bar 0.955 with CI lower bound > 0.950; open loop 0.830); wave-1 preferred share 0.822 (bar 0.75) |
| E1-D2.2 policy independence | **PASS** | attn_L3-K3's JSD falls 2.93x [2.74, 3.12] with mixed reads and 5.38x [4.58, 6.18] with clean reads (bar 2.0x, both paired CIs > 0); KL falls 51.3% and 53.1% (bar 35%) |
| E1-D3 weight law and key attribution | **FAIL** (one clause) | tau 0.3: W(16)/W(1) 1.024 (bar <= 1.25), MSE cut 38.6% vs ataraxos_mmd and 25.1% vs kc_K1 (bars 35%, 20%; CIs > 0); tau 2.5: W(8)/W(1) 3.591 [3.020, 4.162], W(16)/W(1) 4.306, MSE cut 22.7% vs ataraxos_mmd, beats kc_K3 (CI > 0); monotone in every seed; **but the MSE cut vs kc_K1 at tau 2.5 is 14.8% [11.4, 18.2] (bar 20%)** |

- **theta_real v2 is saved, once, for all later rounds.** `ckpt/s{101,102,103}/theta_real_v2.npz` (2.5 MB each, 669,633
  parameters, sha256 in section 7), each with a `.json` of its architecture, training curve and hash. Load with
  `e1v4.load_wm_npz(path, device)`. The rerun after a spot preemption loaded all three, so the checkpoints are already
  tested in use.
- **Weight-tied attention passes its rule at 37% of attn_L3's parameters, with no detectable difference from attn_L3-K3
  (3 seeds).** looped_attn_T3 (one attention block applied 3 times) reaches KL 0.0317 against attn_L3-K3's 0.0328 (paired
  CI of the difference includes 0) and JSD 0.0066. On controlled bags its MSE is within 0.5% of attn_L3-K3's at every tier
  (all CIs include 0). Equivalence was not tested.
- **jacobi_lin_T4's KL is 40% below R6's looped linear pool's, but it does not deduplicate clean reads enough.** With 812
  parameters its KL is 0.0408, and its support grows with noise as the law requires. Sixteen identical clean reads still
  weigh 1.45x one.
  - looped_lin_T4's W(N) was not measured here; its formula gives W(N) proportional to N for isolated identical reads. The
    two pools also use different key heads (K3 vs K4), so the KL gain is not attributed to the normal-equation fix alone.
  - Interpretation (not tested): for N identical reads the pool's closed form gives W(16)/W(1) = 16(lambda + beta) /
    (lambda + 16 beta). With the learned ridge lambda (cell mean 1.583 / 1.552 / 1.612, `results.json`
    `fit.<seed>.jacobi`) and the mean precision of the tau-0.3 calibration reads (3.365 / 3.622 / 3.351,
    `fit.<seed>.beta_by_tau`), that is 1.43 / 1.39 / 1.44 per seed, near the measured 1.45 / 1.42 / 1.47. The bar needs
    lambda <= 0.27 beta, about 0.9-1.0 here (`numbers.txt`).
- **Key attribution (descriptive; the split depends on the order of the swaps).**
  - Swapping kc_dedup's keys from K1 to K3 cuts P-FullSupport KL from 0.0939 to 0.0670 and JSD from 0.0410 to 0.0136.
    It closes 44-80% of the gap between kc_K1 and attn_L3-K3, depending on the measure.
  - Swapping attn_L3's own keys from K1 to K3 closes only 5-9% of the same gap, because attn_L3-K1 is already close to
    attn_L3-K3. Salience keys matter far more for kc_dedup than for attention, so the gap does not split into a key share
    and a pool share.
  - With the same K3 keys, attn_L3-K3 still beats kc_K3 at every tier (7.5% lower MSE at tau 2.5, CI > 0).
- **Key-space PUCT now covers 98.6% of cells in 4 waves** (R6: 94.9%). The amended score, the K3 keys and the K3 pool's
  belief in the variance term (R6 used K4 for both) changed together, on a retrained world model, so their shares were not
  separated.
- **Decomposition (interpretation).**
  - After PUCT with mixed reads, the union call's JSD (one call on the clean readings of all read cells) is 0.0007, 25x
    below open_U's. The unvisited-cell part of the disagreement is nearly gone.
  - attn_L3-K3's JSD is still 12x the union call's with mixed reads (0.0085). With clean reads it is 6x (0.0039); there,
    two searches reading the same cell see the same reading.
  - Clean reads remove about half of attn_L3-K3's remaining JSD (mixed / clean 2.21x), so read noise is about half of it.
    The clean-read remainder, 6x the union call's, reflects how the pool weighs the different multisets of reads the two
    searches made. Evidence one search never saw is a small part.
- **Cost: $0.98** by the ledger at the list-price ceiling, of the $2.50 ceiling: $0.76 for a spot VM preempted at 10:32Z
  during pool training (after the checkpoints had synced), $0.22 for the on-demand rerun from the checkpoints. The GCE
  operation times give $0.99 (the rerun VM ran 0.262 h, not the 0.25 h of its DELETE line). Zero r7e1 VMs at 10:54:46Z.

## 2. What ran

- **Questions (spec section 3.1).**
  1. Does an exact Jacobi-preconditioned normal-equation pool (`jacobi_lin_T4`, under 3k parameters) fix R6's looped linear
     pool and reproduce the weight law on its own?
  2. Does a weight-tied attention pool (`looped_attn_T3`, one block applied 3 times, 37% of attn_L3's parameters) match
     attn_L3?
  3. How much of attn_L3's margin comes from its keys? kc_dedup is run with round-5 raw keys (K1) and with salience keys (K3).
  4. Does key-space PUCT with gemini's amended wave-2+ score reach full coverage in 4 waves? With clean reads, how much of
     the remaining policy dependence is unseen evidence?
- **Environment and world model** (as R5-E1 Part A v2 and R6-E1; `li-net/r5e1/partA2.py`, imported unchanged):
  - 6x6 board, ships 4/3/2, a commander's note, and a dense sonar field (one reading per cell, flip 0.15).
  - The root belief sees the note + 2 anchor readings. A read is one theta_real call on the note, the anchors and the 6
    readings of one 2x3 / 3x2 patch.
  - Read tiers: tau 0.3 clean; 1.5 and 2.5 re-flip each patch reading with probability 0.127 and 0.28 (effective flips 0.239
    and 0.346).
  - theta_real v2 (669,633 parameters) was retrained once per seed exactly as in R5-E1 and R6-E1 (40,000 steps) and saved
    (section 7).
  - Exact Bayes over 49,504 ship patterns, given all 36 readings and the note, is used for scoring only.
- **Learned pieces** (calibration split, 3,200 fit boards):
  - Learned pools and fitted scalars: target = the model's own unskewed K = 128 union-call belief, never exact Bayes.
  - theta_trust: real excess surprise of each calibration read against that board's true occupancy (no exact Bayes).
  - attn_L3 (R6 architecture) with K1 (raw hidden state), K3 (salience) and K4 (centred + salience) key heads.
  - looped_lin_T4: R6-E1's pool, unchanged, retrained as the comparator.
  - jacobi_lin_T4: g_rb = relu(<k_r,k_b>)^2 on its own K3 keys; 4 Jacobi-Richardson steps on
    (lambda_c + sum_b g_rb beta_b) x = beta Delta l with the whole residual divided by D_rc = lambda_c + sum_b g_rb beta_b;
    learned lambda_c, eta_t, P_tc and output scale s_c; 812 parameters.
  - looped_attn_T3: one AttnL3 block (attention + FFN) applied 3 times with a learned step embedding; its own K3 head;
    39,428 parameters.
  - Fitted scalars: ataraxos_mmd eta; kc_dedup u on the frozen K1, K3 and K4 heads (kc_K1, kc_K3, kc_K4); econ_kc_K3 (u,
    lambda); scaled_mix alpha.
- **Tests** (the same 500 test boards as R5-E1 v2 and R6-E1):
  - P-FullSupport: both searches read the 6 partition tiles, then 18 clean reads skewed 90/10.
  - P-ActivePUCT, under mixed reads (tau 50/50 over 0.3 / 1.5, the R6 setting) and clean reads (all tau 0.3):
    - open_U: 85/15 open loop;
    - puct_key_U: wave 1 sampled in proportion to (preference x variance); waves 2+ sampled in proportion to
      (1 + 3 / (1 + w0) x 1[preferred half]) x (variance / (1 + K3 key crowding)^1.5)^2, gemini's amended reading 9;
    - a greedy arm (top 6 of the same score) and a 5th wave (K = 30), both descriptive.
  - Key geometry (descriptive) and controlled bags with N same-patch reads at three tiers, as R6.
- **Seeds** 101, 102, 103 on one L4, run concurrently. Statistics: 3-seed means with seed-t 95% intervals (t = 4.303).

## 3. E1-D0: the gate (PASS)

Source: `results/main/analysis/results.json` (`verdicts.E1-D0`, `stats`).
- **World model.** theta_real's one-pass belief on the standard root (6 readings + note) correlates 0.9995 [0.9994, 0.9995]
  with exact Bayes on the same evidence (per seed 0.9994 / 0.9995 / 0.9995).
- **P-FullSupport.**
  - The root belief (note + 2 anchors) is at KL 0.2797 nats per cell from the full-board posterior.
  - scaled_mix (l0 + alpha sum_r Delta l_r; alpha 0.2197 / 0.2230 / 0.2210 per seed, `results.json` `fit.<seed>.scalars`,
    fitted to the model's own reference) brings it to 0.1036: a cut of 0.6295 [0.6195, 0.6394].
  - For reference, the one-call union of the same clean readings is at 0.0022.

## 4. E1-D1: iterative pools on P-FullSupport (D1.1 PASS, D1.2 FAIL)

Source: `results.json` (`verdicts.E1-D1.1`, `E1-D1.2`, `stats`), figure `results/main/analysis/fullsupport_pools.png` (data
`fullsupport_pools.csv`). Each search reads the 6 partition tiles, then 18 clean reads, 90% in its preferred half. KL to
exact Bayes (mean of the U_NW and U_SE bags) and JSD between the two merged beliefs, 3-seed means:

| pool | parameters | KL | JSD(U_NW, U_SE) |
|---|---|---|---|
| mix | 0 | 0.1973 | 0.00384 |
| scaled_mix | 1 | 0.1036 | 0.04585 |
| ataraxos_mmd | 1 | 0.1183 | 0.05097 |
| kc_K1 (frozen attn_L3-K1 keys) | 1 | 0.0939 | 0.04104 |
| kc_K3 (frozen attn_L3-K3 keys) | 1 | 0.0670 | 0.01358 |
| kc_K4 (R6's kc_dedup) | 1 | 0.0734 | 0.02261 |
| econ_kc_K3 (one precision-crowding step) | 2 | 0.0457 | 0.01229 |
| looped_lin_T4 (R6, K4 head) | 2,789 | 0.0680 | 0.02209 |
| **jacobi_lin_T4** (K3 head) | 812 | **0.0408** | 0.01250 |
| **looped_attn_T3** (K3 head) | 39,428 | **0.0317** | **0.00657** |
| attn_L3-K1 / K3 / K4 | 107,652 / 106,180 / 108,228 | 0.0358 / 0.0328 / 0.0332 | 0.00862 / 0.00692 / 0.00770 |
| union call (clean readings, one call) | | 0.0022 | 0.00000 |

- **E1-D1.1 PASS (looped_attn_T3).**
  - Parameters: 39,428 / 106,180 = 0.371 of attn_L3-K3 (bar 0.38).
  - KL ratio to attn_L3-K3 0.967 [0.927, 1.007] (per seed 0.956 / 0.985 / 0.959; bar 1.10); its KL 0.0317 is also below
    the 0.038 the spec mentions.
  - JSD ratio 7.05 [4.84, 9.25] vs scaled_mix and 7.84 [5.14, 10.55] vs ataraxos_mmd (bar 4.5 each).
  - Paired CIs of (baseline minus looped_attn_T3), all > 0:
    - KL: scaled_mix 0.0719 [0.0680, 0.0758], ataraxos_mmd 0.0867 [0.0749, 0.0984], kc_K1 0.0622 [0.0570, 0.0674],
      kc_K3 0.0353 [0.0336, 0.0370];
    - JSD: 0.0393 [0.0356, 0.0429], 0.0444 [0.0391, 0.0498], 0.0345 [0.0300, 0.0389], 0.0070 [0.0054, 0.0086].
  - Against attn_L3-K3 itself, the paired KL difference is -0.0011 [-0.0024, 0.0003] and the JSD difference
    -0.00036 [-0.00179, 0.00107]: no cost of weight tying is detectable with 3 seeds.
  - Its JSD is the lowest of all pools except mix, whose belief stays far from exact Bayes (KL 0.1973, against the root's
    0.2797).
- **E1-D1.2 FAIL (jacobi_lin_T4), on the clean-deduplication clause only.**
  - 812 parameters (bar 3,000).
  - KL 0.0408 [0.0397, 0.0418], below R6's looped_lin_T4 retrained here (0.0680; difference 0.0273 [0.0179, 0.0366]) and
    below kc_K1 (0.0939; difference 0.0532 [0.0492, 0.0572]), and lower in every seed. R6-E1 reported 0.069 for
    looped_lin_T4.
  - Controlled bags (section 6): W(8)/W(1) is 1.404 / 1.823 / 2.201 at tau 0.3 / 1.5 / 2.5, ordered in every seed, and
    2.201 [2.113, 2.290] meets the tau-2.5 bar of 2.0.
  - **W(16)/W(1) at tau 0.3 is 1.449 [1.392, 1.507] (per seed 1.454 / 1.424 / 1.470), above the bar of 1.25.**
- Learned jacobi_lin_T4 values (`results.json` `fit.<seed>.jacobi`):
  - ridge lambda_c, cell mean 1.583 / 1.552 / 1.612 per seed;
  - step size times preconditioner eta_t P_t about 0.19 at each of the 4 steps (heavily under-relaxed);
  - output scale s_c, cell mean about 2.25.

  For a cluster of identical reads and no other reads, the iteration's rescaling cancels in W(N)/W(1), so the ratio depends
  only on lambda and beta. This is a property of the formula; the controlled bags also hold 8 background reads (the
  interpretation in section 1).
- econ_kc_K3 (one precision-crowding step with two fitted scalars on attn_L3-K3's frozen keys) is close behind:
  KL 0.0457; jacobi minus econ -0.0049 [-0.0065, -0.0033]. The two pools differ in more than the number of steps (per-cell
  parameters, their own key head), so the share of the extra steps was not isolated (descriptive).

## 5. E1-D2: coverage-complete key-space PUCT (PASS)

Source: `results.json` (`verdicts.E1-D2.1`, `E1-D2.2`, `stats.ap_*`), figure `results/main/analysis/puct.png` (data
`puct.csv`). Pool: attn_L3-K3. "union" = one theta_real call on the clean readings of all cells the bag read. 3-seed means:

| reads | search | coverage | attn_L3-K3 JSD | attn_L3-K3 KL | union JSD |
|---|---|---|---|---|---|
| mixed (tau 0.3 / 1.5) | open_U, K = 24 | 0.8301 | 0.02500 | 0.0792 | 0.01739 |
| mixed | **puct_key_U, K = 24** | **0.9863** | **0.00853** | **0.0385** | 0.00069 |
| mixed | greedy arm, K = 24 (descriptive) | 0.9934 | 0.00691 | 0.0349 | 0.00016 |
| mixed | open_U, K = 30 (descriptive) | 0.8658 | 0.02204 | 0.0701 | 0.01369 |
| mixed | puct_key_U, K = 30 (descriptive) | 0.9929 | 0.00657 | 0.0343 | 0.00028 |
| clean (tau 0.3) | open_U, K = 24 | 0.8328 | 0.02081 | 0.0650 | 0.01734 |
| clean | **puct_key_U, K = 24** | **0.9870** | **0.00386** | **0.0304** | 0.00064 |
| clean | greedy arm, K = 24 (descriptive) | 0.9924 | 0.00252 | 0.0284 | 0.00019 |
| clean | open_U, K = 30 (descriptive) | 0.8666 | 0.01813 | 0.0580 | 0.01375 |
| clean | puct_key_U, K = 30 (descriptive) | 0.9930 | 0.00294 | 0.0294 | 0.00028 |

- **E1-D2.1 PASS.**
  - Mixed-setting coverage at K = 24 is 0.9863 [0.9855, 0.9871] (per seed 0.9867 / 0.9861 / 0.9862); clean 0.9870.
  - Coverage after each wave (mixed, U_NW): 0.593, 0.892, 0.967, 0.986, then 0.993 after a 5th wave.
  - Wave 1 under U_NW keeps 0.822 [0.803, 0.842] of its reads in the preferred half. This counts the 4 patches that
    straddle rows 2-3 for both halves, as registered; reads wholly in the preferred half are 0.665 [0.626, 0.704].
- **E1-D2.2 PASS in both settings.**
  - Mixed: JSD(open_U) / JSD(puct_key_U) 2.93 [2.74, 3.12], difference 0.0165 [0.0139, 0.0190]; KL cut 0.513
    [0.477, 0.549].
  - Clean: 5.38 [4.58, 6.18], difference 0.0169 [0.0129, 0.0210]; KL cut 0.531 [0.514, 0.549].
- **Decomposition (the round-6 audit's question).** These are ratios of the table's quantities (3-seed means of per-seed
  ratios); the reading of them is interpretation.
  - The union call's JSD measures how differently the two searches' sets of read cells inform the model. PUCT cuts it 25.2x
    with mixed reads (0.01739 to 0.00069) and 27.1x with clean reads.
  - After PUCT, attn_L3-K3's JSD is 12.4x [10.3, 14.4] the union call's with mixed reads and 6.0x [4.5, 7.6] with clean
    reads.
  - At the same coverage, clean reads halve attn_L3-K3's JSD (mixed / clean 2.21 [2.14, 2.27]).
  - So the remaining policy dependence is mostly read noise (the tau-1.5 reads re-flip differently in the two searches) plus
    the pool's response to which reads, and how many repeats, each search made. Evidence one search never saw is a small part.
- The greedy arm covers slightly more (0.9934) and lowers JSD further (mixed 0.00691). It is descriptive; the rules use the
  sampled search.
- What changed since R6-E1 (0.9494 coverage): the amended wave-2+ score (decayed preference, squared novelty-weighted
  variance), K3 instead of K4 keys, and attn_L3-K3's belief in the variance term, on a retrained world model, all
  together. They were not separated.

## 6. E1-D3: the weight law on controlled bags and key attribution (FAIL on one clause)

Controlled bags: N reads of one patch A* at each tier, plus 8 shared background reads. W(N) = the sum over the cluster reads
of the pool's per-read weight averaged over A*'s 6 cells (for jacobi_lin_T4, the exact coefficient of each read's Delta l in
its output). MSE = 36-cell MSE to exact Bayes, averaged over N. 3-seed means. Source: `results.json` (`verdicts.E1-D3`,
`stats.weights`, `stats.mse`), figure `results/main/analysis/controlled_bags.png` (data `controlled_bags.csv`,
`controlled_mse.csv`).

| pool, tier | W(1) | W(2) | W(4) | W(8) | W(16) | W(8)/W(1) | W(16)/W(1) |
|---|---|---|---|---|---|---|---|
| attn_L3-K3, tau 0.3 | 0.884 | 0.998 | 0.955 | 0.912 | 0.907 | 1.030 | **1.024** [0.847, 1.201] |
| attn_L3-K3, tau 1.5 | 0.574 | 0.823 | 1.035 | 1.160 | 1.217 | 2.023 | 2.121 |
| attn_L3-K3, tau 2.5 | 0.361 | 0.598 | 0.928 | 1.291 | 1.547 | **3.591** [3.020, 4.162] | **4.306** [3.413, 5.199] |
| jacobi_lin_T4, tau 0.3 | 0.787 | 0.939 | 1.042 | 1.105 | 1.140 | 1.404 | **1.449** [1.392, 1.507] |
| jacobi_lin_T4, tau 1.5 | 0.663 | 0.890 | 1.079 | 1.209 | 1.290 | 1.823 | 1.945 |
| jacobi_lin_T4, tau 2.5 | 0.556 | 0.802 | 1.039 | 1.224 | 1.344 | **2.201** [2.113, 2.290] | 2.418 |
| looped_attn_T3, tau 0.3 / 2.5 | | | | | | 1.025 / 3.657 | 1.072 / 4.313 |
| kc_K1, tau 0.3 / 2.5 | | | | | | 3.329 / 3.913 | 4.023 / 4.971 |

| pool | MSE tau 0.3 | MSE tau 1.5 | MSE tau 2.5 |
|---|---|---|---|
| **attn_L3-K3** | 0.03314 | 0.03615 | 0.04087 |
| looped_attn_T3 | 0.03299 | 0.03614 | 0.04092 |
| jacobi_lin_T4 | 0.03365 | 0.03683 | 0.04233 |
| econ_kc_K3 | 0.03382 | 0.03683 | 0.04221 |
| kc_K3 | 0.03573 | 0.03809 | 0.04418 |
| kc_K4 | 0.03747 | 0.03892 | 0.04479 |
| kc_K1 | 0.04425 | 0.04294 | 0.04797 |
| ataraxos_mmd | 0.05395 | 0.04957 | 0.05291 |

- **Clause 1 (clean deduplication): passes.**
  - W(16)/W(1) = 1.024 [0.847, 1.201] (per seed 0.995 / 1.105 / 0.972; bar <= 1.25).
  - MSE cut 0.386 [0.352, 0.419] vs ataraxos_mmd (bar 0.35) and 0.251 [0.203, 0.298] vs kc_K1 (bar 0.20), both paired
    CIs > 0.
- **Clause 2 (high-noise mutual support): fails on one part.**
  - Passing parts: W(8)/W(1) = 3.591 with CI lower bound 3.020 (bars 2.20 and 1.90); W(16)/W(1) = 4.306 (bar 2.80); MSE cut
    0.227 [0.202, 0.253] vs ataraxos_mmd (bar 0.20); MSE(kc_K3) - MSE(attn_L3-K3) = 0.0033 [0.0026, 0.0040] > 0.
  - **The MSE cut vs kc_K1 is 0.148 [0.114, 0.182] (per seed 0.152 / 0.159 / 0.133), below the bar of 0.20.**
- **Clause 3 (monotone in noise): passes in every seed.** W(8)/W(1) is 1.030 < 2.023 < 3.591 on average.
- **Key attribution** (descriptive, not a registered rule; `numbers.txt`). How much of the gap between kc_K1 and
  attn_L3-K3 does one key swap close? Per seed, then averaged:
  - Keys swapped inside kc_dedup, (X(kc_K1) - X(kc_K3)) / (X(kc_K1) - X(attn_L3-K3)):
    - controlled-bag MSE: 76.6% [69.9, 83.4] at tau 0.3, 71.3% [61.5, 81.1] at tau 1.5, 53.3% [41.1, 65.5] at tau 2.5;
    - P-FullSupport: 44.0% [37.0, 51.0] of the KL gap and 80.5% [79.3, 81.6] of the JSD gap.
  - Keys swapped inside attn_L3, (X(attn_L3-K1) - X(attn_L3-K3)) / (X(kc_K1) - X(attn_L3-K3)):
    - P-FullSupport: 4.9% [3.5, 6.4] of the KL gap and 5.0% [0.2, 9.9] of the JSD gap;
    - controlled-bag MSE: 8.1%, 9.4% and 5.3% [-1.6, 12.2] at the three tiers.
  - The interaction is large, so neither order gives the keys' share.

  With the same K3 keys, attn_L3-K3's MSE is 7.2%, 5.1% and 7.5% below kc_K3's at the three tiers, all CIs > 0. Against
  econ_kc_K3 (one precision-crowding step on the same keys) the margin is 2.0%, 1.8% and 3.2% (CI [-0.1, 4.1] at tau 0.3,
  > 0 at the other tiers).
- Why kc_K1 is the hard baseline at tau 2.5 (interpretation):
  - Raw keys give every pair of reads a high cosine: 0.72-0.95 at any overlap (`stats.keys.K1.cos_by_overlap_mean`).
  - So each read is already crowded by the whole bag, and N same-patch reads add relatively little crowding. kc_K1's
    cluster weight keeps growing: W(16)/W(1) is 4.97 at tau 2.5 and 4.02 at tau 0.3.
  - kc_K1 has the highest MSE of the three kc pools at every tier (0.04425 / 0.04294 / 0.04797). What changes with noise
    is attn_L3-K3's margin over it: 25.1% at tau 0.3, 14.8% at tau 2.5. That a growing cluster weight suits high noise is
    a reading, not a test.

## 7. theta_real checkpoints and key geometry

- **Checkpoints** (`ckpt/s<seed>/theta_real_v2.npz` + `.json`). np.savez_compressed of the float32 state dict (58 arrays,
  669,633 parameters; d 128, 4 layers, 4 heads, ff 384), trained 40,000 steps as in R5-E1 / R6-E1, saved at 10:26:53-10:27:03Z:

  | seed | sha256 | bytes | final val BCE root / std6 / full |
  |---|---|---|---|
  | 101 | aef34272fedac158223a7788e0a9ab2007881e0bf8e0ab01c1742ff89397c6f6 | 2,506,901 | 0.5365 / 0.5011 / 0.2549 |
  | 102 | abb0fffc6b4835a0a561365a0cfff11825f6a694c4725209b8152c4de7c46b35 | 2,506,469 | 0.5365 / 0.5011 / 0.2557 |
  | 103 | f9842c1e6a7c0a0209eede89233a50ef0bb2442e060f32ca8616fe5eaa475de2 | 2,506,226 | 0.5365 / 0.5010 / 0.2560 |

  - Checks: on the VM, each was reloaded into a fresh model right after saving, with max |logit difference| 0.0 on 200 test
    roots (`results/main-preempted/s*/summary.json`, `ckpt`). Here, each loads with a strict state-dict match and its file
    hash equals its `.json` (10:30Z). The rerun loaded all three with a hash check (`results/main/s*/summary.json`,
    `loaded_theta_real`); every number above comes from the loaded models.
  - Use: `import e1v4; model, meta = e1v4.load_wm_npz("li-net/r7e1/ckpt/s101/theta_real_v2.npz", "cuda")` (needs
    `li-net/r5e1` and `smallnet` on the path, as e1v4.py sets up).
- **Key geometry** (descriptive; R6's protocol, within-bag pairs of the mixed open_U bags and an unskewed bag; `results.json`
  `stats.keys`): Pearson of key cosine with overlap/6:

  | key head | Pearson | same-patch minus across-sector cosine |
  |---|---|---|
  | K1 (attn_L3-K1) | 0.387 | 0.190 |
  | K3 (attn_L3-K3) | 0.880 | 0.963 |
  | K4 (attn_L3-K4) | 0.866 | 0.629 |
  | jacobi_lin_T4's K3 head | 0.898 | 1.040 |
  | looped_attn_T3's K3 head | 0.878 | 0.976 |

  This reproduces R6-E1's key results on the retrained models (R6: K1 0.382, K3 0.880, K4 0.863).

## 8. Deviations and what ran when

- **09:15Z** assignment (#3717). **09:22:01Z** pre-spend ambiguity mail (#3722, readings 1-10).
- **09:24Z** gemini's amendment file. It approved readings 1-8 and 10 and replaced reading 9. I found it at 09:35Z (my check
  before registration) and implemented it as written, with a unit test, before registering.
- **09:39:09Z** pre-registration (`PREREG-R7-E1.md`, sha256 1f9179bc...710e; the registered copy is
  `results/PREREG-R7-E1-as-registered.md`). It discloses the CPU code-path smokes; there was no GPU smoke.
- **09:39:22Z** launch `r7e1-main` (spot), created in us-central1-c after stockouts in -a and -b. theta_real trained
  2,550-2,556 s per seed; the checkpoints were saved at 10:26:53-10:27:03Z and synced by 10:29Z.
- **10:32:07Z** GCE preempted the spot VM during pool training; no test number existed. Cost $0.76 (COST-AUDIT line, 10:35Z).
  The launcher and GPU-LOG.md record rc=0 for this run because `driver7.sh` always exits 0; the driver log has remote
  rc=255.
  **Deviation 1** (logged at 10:35:18Z, before the relaunch, in `PREREG-R7-E1.md` section 5): the registered recovery rule,
  an on-demand rerun `r7e1-main2` that loads each seed's own checkpoint, with a 40-minute remote budget instead of 100.
  The preempted run's summaries (training curves, VM-side reload checks) are kept in `results/main-preempted/`.
- **10:35:37Z** launch `r7e1-main2` (on-demand). The first create failed in all three zones (stockout); the retry at
  10:37:29Z created the VM in us-central1-c. All seeds loaded their checkpoints at 10:41:58-10:42:02Z and finished with rc 0
  and no retry (`results/main/s*/rc.json`). The VM was deleted at 10:54Z.
- **10:54:18Z** scoring with the registered `analyze7.py` (unchanged, sha256 8263475f...). No further amendment file existed at
  launch, rerun or scoring (only the 09:24Z file).
- **11:26:12Z** deviation 2 (logged in `PREREG-R7-E1.md` before the script existed): `numbers7.py`, a descriptive numbers
  script added at the audit's request. It prints every number this file quotes, rounded once, plus the derived ones;
  output `results/main/analysis/numbers.txt`. No rule input or verdict changed.
- No code or analysis file registered at 09:39:09Z has changed since. Created after registration, for publishing only:
  `render_page7.py`, `site_patch7.py`, `site-entry.json`, `GPU-LOG.md`; and `numbers7.py` (deviation 2).
- Exact Bayes was used only for test metrics, the gate and the training monitor curve. PUCT never sees an unread cell's real
  reading: unit test `test_puct7_never_uses_unread_real_readings`, a per-wave flip test with a positive control, both
  selection modes.

## 9. Compute and GPU-LOG

L4 VMs (g2-standard-8) through `logical-soup/infra/gpu-run.sh`, group `r7` (cap $6.50 shared with R7-E2 and R7-E3), at the
$0.87/h list-price ceiling (real spend on spot is lower). Ledger: `python3 ledger7.py --markdown` (`GPU-LOG.md`).

| VM | zone | provisioning | created (UTC) | deleted (UTC) | cost USD (ceiling) | basis |
|---|---|---|---|---|---|---|
| soup-gpu-r7e1-main2-1017103537-1091 |  |  |  |  | 0.00 | create failed (no instance) |
| soup-gpu-r7e1-main-1017093922-28650 | us-central1-c | SPOT | 2026-10-17T09:40Z | 2026-10-17T10:35Z | 0.76 | DELETE/COST-AUDIT line |
| soup-gpu-r7e1-main2-1017103729-163 | us-central1-c | STANDARD | 2026-10-17T10:38Z | 2026-10-17T10:54Z | 0.22 | DELETE/COST-AUDIT line |

**R7-E1 total (ceiling): $0.98** of its $2.50 ceiling (3 VM entries).

- Per seed (3 seeds sharing one L4): theta_real 40,000 steps in 2,550-2,556 s (first VM); calibration fit (reads, reference,
  theta_trust, six learned pools, seven fitted scalars in six fits) 535-541 s and tests 108 s on the rerun VM
  (`results.json` `compute`).
- Zero r7e1 VMs at the end: `gcloud --project=GCP_PROJECT compute instances list --filter='name~r7e1'` returned nothing at
  10:54:46Z.

## 10. Audit

Independent audit by a Board investigator subagent (Claude Opus 5.5), 10:55-11:25Z, of this file as it stood before the
corrections below (sha256 196d7727...): `AUDIT-R7-E1.md`, with its scripts and outputs in `audit/`. It did not run the
experiment and changed nothing in it.

- **Verdict: sound, publishable after wording corrections.** It re-decided every rule from its own numbers (a numpy
  re-derivation from `raw.json`) and confirmed every verdict: E1-D0, E1-D1.1, E1-D2.1 and E1-D2.2 PASS; E1-D1.2 and E1-D3
  FAIL, each failure in every seed and under every reading of the spec it tried.
- **Numbers.**
  - It recomputed 789 per-seed statistics. The largest difference, 1.44e-4 (seed 102, ataraxos_mmd's KL under open_U with
    clean reads, a descriptive number), lies inside the bound that the 5-decimal rounding of `raw.json` allows.
  - Numbers that decide a rule agree within 6.3e-5 per seed.
  - All 504 numbers it mapped by hand in this file matched at their printed precision.
- **Confirmed.**
  - Integrity: registration came before the first launch, GCE insert and GPU-LOG line. Every registered file is unchanged and
    was staged at every launch. The registered `analyze7.py` reproduces all 9 analysis files byte for byte, and the unit
    tests pass.
  - No oracle: exact Bayes appears only in scoring, the gate and the monitor curve. A PUCT flip test with the real
    theta_real passed 40 of 40 cases (both selection modes, both read settings, waves 1-5), and its positive control changed
    later waves in 4 of 4 settings. Calibration and test boards share no board.
  - Checkpoints: all three load strictly (669,633 parameters, hashes match) and reproduce the clause-1 Pearson on all 500
    test boards within 1.5e-7 on CPU.
  - jacobi_lin_T4's W matches the autograd coefficient within 3e-7.
  - Cost: $0.99 from GCE times ($0.98 by the ledger). No r7e1 VM existed at 11:08:19Z.
- **Corrections applied after the audit** (none changes a verdict):
  - Key attribution is now reported in both swap orders (44-80% inside kc_dedup, 5-9% inside attn_L3) and labelled
    descriptive and order-dependent, outside "Settled".
  - The weight-tied pool "passes its rule with no detectable difference (3 seeds)" instead of "matches attn_L3 on every
    measure".
  - "Repairs R6's looped linear pool" is replaced by the KL comparison and its caveats.
  - The kc_K1 interpretation now says kc_K1 has the highest MSE of the kc pools at every tier.
  - Smaller fixes: seven fitted scalars, not six; the closed form per seed with its sources; the mix wording; the cost from
    GCE times; the rc=0 note; the full list of PUCT changes since R6.
  - The derived numbers now come from a committed script, `numbers7.py`, whose output is
    `results/main/analysis/numbers.txt` (deviation 2).
- **Not checkable by the auditor:**
  - the GPU training stream, which cannot be regenerated on CPU;
  - the trained pools and theta_trust, which were not saved;
  - times only I logged (the mails, the 10:29Z sync, the 10:30Z load check, the 10:54:46Z VM check);
  - actual billing;
  - R6's reference values, which it checked against the spec text, not R6's files.

## 11. What this means, and what could come next (not run)

- **Settled in this testbed.**
  - A weight-tied 3-pass attention pool with 37% of the parameters passed its registered rule (KL within 1.10x of
    attn_L3-K3). On the other measures no difference from attn_L3-K3 was detectable with 3 seeds (policy-independence JSD,
    the weight-law clauses in every seed, controlled-bag MSE). Equivalence was not tested; the JSD difference is -0.00036
    [-0.00179, 0.00107].
  - Coverage-complete key-space PUCT is reached in 4 waves (98.6%), and with it the support-gap part of policy dependence
    nearly disappears.
- **Descriptive, order-dependent.** Salience keys close 44-80% of the kc_K1 to attn_L3-K3 gap when swapped into kc_dedup,
  but only 5-9% when swapped into attn_L3. On the same K3 keys, attention's MSE is 5.1-7.5% below kc_dedup's (all CIs > 0)
  and 1.8-3.2% below one precision-crowding step's (CI includes 0 at tau 0.3).
- **Not settled.**
  - The linear Jacobi-Richardson pool is accurate (KL 0.041) and grows support with noise, but it does not deduplicate
    identical clean reads (W(16)/W(1) 1.45). The closed form points to its learned ridge (not tested).
  - Against raw-key kc_dedup, attn_L3's high-noise margin (14.8%) stays short of 20%.
  - With 98.7% coverage and clean reads, attn_L3's two preference-steered beliefs still differ by 6x the union call's JSD.
- **Possible next steps** (for gemini's next round; none was run):
  1. jacobi_lin_T4 with lambda tied to the read's own precision (lambda_c = kappa beta-bar) or with a lambda per noise tier,
     to test whether the deduplication failure is only the ridge.
  2. A PUCT ablation: the amended score with K4 keys and the R6 score with K3 keys, to split the coverage gain.
  3. A policy-independence test on bags with matched read multiplicity, to measure the pool's own share of the remaining JSD.

## 12. Files

- Code: `e1v4.py` (checkpoints, jacobi_lin_T4, looped_attn_T3, econ_kc, PUCT, protocols), `analyze7.py`, `test_r7e1.py`
  (9 tests), `ledger7.py`, `stage7.sh`, `remote7.sh`, `driver7.sh`, `launch7.sh`; `numbers7.py` (descriptive numbers,
  deviation 2). Imported unchanged: `li-net/r6e1/e1v3.py`;
  from `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`, `partA2.py`. Publishing: `render_page7.py`,
  `site_patch7.py`, `site-entry.json`.
- Pre-registration: `PREREG-R7-E1.md` (+ `.sha256`, `results/PREREG-R7-E1-as-registered.md`); staged hashes per launch:
  `results/stage-manifests.txt`.
- Checkpoints: `ckpt/s{101,102,103}/theta_real_v2.{npz,json}`.
- Raw results (JSON only came back from the VMs): `results/main/s{101,102,103}/{summary.json, raw.json, rc.json}`; the
  preempted run's summaries in `results/main-preempted/`. Derived: `results/main/analysis/{results.json, verdicts.md,
  fullsupport_pools.csv, fullsupport_pools.png, puct.csv, puct.png, controlled_bags.csv, controlled_mse.csv,
  controlled_bags.png}` and `numbers.txt` (from `numbers7.py`). Text logs: `/tmp/r7e1/logs/`, temporary and not kept
  (only JSON/CSV/PNG and the checkpoints come back from the VM).
- Audit: `AUDIT-R7-E1.md`, `audit/`. Page: `/home/agent/memo-serve/lir-agent/r7-e1.html`. GPU ledger: `GPU-LOG.md`.
