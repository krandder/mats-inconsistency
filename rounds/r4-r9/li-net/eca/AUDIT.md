# E-CA independent audit: code, bug story, leakage, permutation, stop rules, headline numbers, GPU spend

Auditor: Opus 5.5 (1M context; model ID claude-opus-5-5[1m]), independent pass, 2026-10-15T04:27Z. Checks 0, 2, 3, 4 started 03:52Z on the pre-final tree; checks 1, 5, 6 and a full re-run of every script ran on the final files after /tmp/eca/FINAL-READY appeared (04:10:32Z). Audited versions: RESULTS.md md5 33adc70f (built 04:10:56Z), results/summary.json md5 d2b39b88, GPU-LOG.md md5 4c1bb201, raw tree of 121 files (md5 lists: /tmp/eca/audit/final-checksums.txt, final-raw-checksums.txt; all unchanged at the last check except eca/ecanet.py, see L13 and the end of this file).

**Summary (5 lines)**
1. Every pre-registered verdict re-derives from the primary raw files with independent code: D0 FAIL in all four cells; D1 PASS r110-T1, r110-T2, r90-T2, FAIL r90-T1; D2 FAIL in all four; D3 PASS r110-T1, r110-T2, r90-T1, FAIL r90-T2. summary.json (12,888 numbers) and tables.md / verdicts.md reproduce: 0 differences above 1e-6 relative when KL is computed as analyze.py does; 1,202 differ by at most 6.8e-6 absolute when KL is computed exactly (float64 cancellation in analyze.py; immaterial).
2. The bug story, the code diff, the frozen source copies, the data, the permutation, the stop rules and the exact inference all check out; no exact posterior and no CA-rule function feeds training, a fitted parameter or any choice. eca/ecanet.py was edited at 04:26Z, after FINAL-READY, so the as-run network code is now only in the frozen copies (L13).
3. GPU: $17.02 at the $0.87/h ceiling (sum of 18 DELETE / COST-AUDIT lines), equal to RESULTS.md and eca/GPU-LOG.md; gcloud at 04:12:10Z lists no soup-gpu-eca* instance.
4. Overstated in RESULTS.md: stiffness-first chains do improve rule-110 B2 under the mixture (mean cut 0.09-0.17, paired CIs exclude 0 in both arms); random-order chains lower rule-110 forward KL at M = 1-2, which RESULTS Sec. 4 itself reports, against the TL;DR's "every search arm makes things worse"; the P1-arm headline is a readout added at 03:47Z after the pattern was seen, and RESULTS does not label it post hoc (it does hold in all four rule-110 bridge cells and in the replication).
5. The r90-T1 D1 FAIL (AUC 0.8476 against 0.85) rests on the pre-registered "deterministic local blanket" label, which also counts child triples that do not determine the cell (6.3% of rule-90 positives); with a rule-aware label the AUC is 0.8502. The exact-information-gain AUC called a "ceiling" is not an upper bound (beta exceeds it in 3 of 6 rule-110 runs).

## Verdict table

| check | result | evidence |
|---|---|---|
| 0. Bug story, as-run vs fixed code, primary-source rule | PASS | Original CSVs (results/raw/main, 12 runs x 4,000 rows): 694,465 "+inf" logit cells in 50-61 of the 122 estimate columns per run, 0 "-inf", 0 NaN; all in lw/dd pools at M <= 8 and the g1 product at every M (e.g. voi\|lw\|0: 7,430; P0_logit, voi\|lw\|16, rand\|lw\|16: 0). T = 8 CSVs: 201,621 "+inf". eca-b CSVs (main2, 12 x 4,000 rows): 0 non-finite; eca-b result.json: 0 non-finite numbers (original runs: 496-616 each). `diff /tmp/eca/ecarun-as-run-0135Z.py eca/ecarun.py`: 15 hunks, all on the evaluation side (float64 casts in kl_bern, logloss, ece, logit_np, sig_np, beta_np, pooled; chain claims, q_prior, P0, D1 q and D3a view probabilities stored as float64; logit_np clip 1e-9 -> 1e-12; CSV precision; header note). The training loop and the head fitting have no hunk; the float32 q used to select chain cells is unchanged. Frozen copies: all 6 src-eca-b-* ecarun.py = eca/ecarun.py (md5 db0f4908); all 11 original and T = 8 copies = the as-run file (md5 fefd4845, also = results/raw/ecarun-as-run-0135Z.py); the smoke copy differs from the as-run file only in the --K default (4 vs 16); ecadata.py, ecanet.py, smallnet/netdef.py identical in all 18 copies and in the tree as of 04:10:54Z (eca/ecanet.py was changed at 04:26:06Z, see L13); all 24 data files identical in every copy and equal, array for array, to a fresh regeneration with ecadata.simulate and the split seeds (T = 6 and T = 8). Primary-source rule (PREREG Sec. 12): 12 of 12 eca-b result.json home, the last (r110-T2-s2) synced 04:07:29Z, before 06:15Z; summary.json "sources" are all main2; the replication is analysed from main/ with the [6e-8, 1 - 6e-8] cap (my capped re-derivation reproduces results/replication/summary.json: 12,888 numbers, 0 mismatches); T = 8 likewise (6,348 numbers, 0 mismatches). All 30 jobs exited rc 0 (runs/status/*.jsonl). One presentation exception in the secondary tables: finding L9. |
| 1a. D2 re-derived from test-estimates.csv | PASS | Own KL code (log-sigmoid, no import of analyze.py) on columns P0_logit, voi\|lw\|16, rand\|lw\|16, voi\|lw\|0, 1,000 instances x 3 seeds per cell. Declared P2 from result.json chains.val[fam\|sel\|lw\|16].logloss (mean over F, B1, B2 and seeds): VoI in all four cells (stiff / voi: r110-T1 0.3018 / 0.2878, r110-T2 0.3034 / 0.2852, r90-T1 0.2409 / 0.2099, r90-T2 0.2413 / 0.2188). (a) and (b) FAIL on every family of every cell; D2 FAIL in all four. Tables below. |
| 1b. D1 re-derived from d1-pairs.csv; blanket label | PASS (label caveat, M4) | AUC(beta; blanket) from the CSV equals result.json to <= 1.0e-6 in all 12 runs; Spearman(beta, ln 2 - exact_H) to <= 1.1e-3 (the CSV rounds beta to 5 dp and exact_H to 6 dp, which creates ties: e.g. 100 of 1,986 nonzero betas tied in r90-T1-s0). The grid / target / reveal_count columns regenerate exactly from the seeds in ecarun.py (rng 31000 + seed, ecadata.random_reveal(2000, 32000 + seed)), so S was recovered; blanket and parents labels recomputed with my own geometry code agree 24,000 / 24,000; exact_p and exact_H agree with my independent inference to <= 5e-7 (CSV rounding). ecadata.blankets checked by hand on cells 0, 20, 47 (T = 6, W = 8): {[9,1,2], [8,7,1], [15,6,7]}; {[11,12,13], [27,18,19], [28,19,21], [29,21,22]}; {[38,39,32]}, all as derived by hand. ecarun.py builds the label in grid order from S_g and ecadata.blankets (line 568), parents-only at line 569. |
| 1c. D3a ratios, D0 | PASS | D3a ratio = dd drift / lw drift recomputed from result.json equals the stored ratio exactly in all 12 runs; m = 5: 0.043 / 0.049 / 0.017 / 0.024, m = 2: 0.133 / 0.153 / 0.051 / 0.073 (r110-T1 / r110-T2 / r90-T1 / r90-T2): PASS x4. D0 from P0.val: FAIL x4 (table below). |
| 1d. Comparison with summary.json and RESULTS.md | PASS (numbers); prose items in 6 | summary.json: 12,888 numbers compared; 0 above 1e-6 relative with analyze.py's KL convention (sigmoid then clip to [1e-12, 1 - 1e-12]); with exact KL, 1,202 differ above 1e-6 relative, all of them KL / log-loss values (or their CI bounds) of pools with very confident estimates (logits up to the +-27.63 clip), where analyze.py's 1 - sigmoid(z) loses digits in float64 (max absolute 6.8e-6 on g1-product values of 0.2-3 nats; max relative 9.8e-5 on a CI bound of -0.0012 nats, absolute 1.1e-7). My own rendition of tables.md (356 of 366 lines; the 10-line compute table was checked by formula, 16(M+1) and 16(1 + 13M) passes) is identical at displayed precision except the oracle-AUC field of the 4 D1 lines (0.8705 / 0.8866 published; 0.8703 / 0.8869 with full-precision exact H; finding L11); verdicts.md identical. RESULTS.md Sec. 3, 9 and 13 contain verdicts.md, the replication and T = 8 verdicts.md and tables.md verbatim. Prose numbers: three small mismatches (L2, L3, L5). |
| 1e. Exact inference spot check | PASS | 20 random test instances per family (F, B1, B2, R) per rule: an independent implementation written for the audit (own transition matrix from the Wolfram rule bits, two clamped forward passes for P(E, Y = 1) and P(E, Y = 0); no backward pass) vs ecadata.posterior_marginals: max abs diff 2.1e-15; vs the CSV pstar column: 5.0e-11 (10-dp rounding). All 1,000 CSV rows per family match the cached posterior at the query cell (<= 5e-11) and y equals the grid value; pstar is identical across all 12 primary runs. `python3 ecadata.py` brute-force check (2^16 grids, rules 110/90/30, eps 0.015/0.05): max abs(fb - brute) 5.3e-14 (RESULTS/PREREG quote 3.8e-15; both are float64 round-off). Mean exact H on test reproduces PREREG Sec. 2 to 3 dp. |
| 2. No posterior or rule knowledge in training or selection | PASS | Read ecarun.py end to end (769 lines). (i) T1 loss = BCE of hidden cells vs true values with uniform random masks (lines 369-377); T2 rbtd8 targets = true values or the next level's detached prediction (378-398). (ii) Early stopping on es_logloss = validation outcome BCE on val grids with random masks, EMA model (349-360, 407-423). (iii) Stiffness head regressed on realised BCE of hidden cal cells (436-461); key head and log_u trained by outcome BCE of the crowded pool on cal bags built from random sets (469-516). (iv) fit_u uses validation outcomes y only (612-625; D3b 728-736). (v) the D2 declaration uses chains.val log loss only (analyze.py 163-165, re-derived). (vi) Exact posteriors (fam_arrays, D.posterior_marginals) feed only the P0 / D0 KL (536-549), test KL (641-643, 736-739), D1 (553-593) and the CSV; D.blankets / D.parents only build the D1 labels and the D3a views; D3b orders use grid rows, an evaluation-side construction. Chains read true values only of revealed cells (154) and sample every materialised cell from q (213-217). Splits use disjoint seeds (train 1.0M / cal 4.0M / val 7.0M / test 9.0M + 10,000 x rule). Not a leak, but note L6: 81.5-82.1% of test grids occur verbatim in the training set (small support). |
| 3. Permutation | PASS | make_perm seeds only on 20261015 + 1000 T + W; make_perm(6, 8) equals the PREREG list; all 36 T = 6 result.json files (results/raw/main, results/raw/main2 and the /tmp/eca/runs/main2 copies) carry that one perm and all 6 T = 8 files carry make_perm(8, 8). Every model input passes through tok() (x[:, PERM]) or INV: Xtr, Xcal, Xes/Res, fam_arrays (P0, chains, D3b, CSV), D1 and its expected version, D3a, the D3b orders. ecanet.Net: learned absolute position embedding pos [48, 128], every Block called as blk(x, None) (smallnet Block passes it as attn_mask; the attention re-computed for the key head adds no bias); no row/column index anywhere in the model or training (masks are uniform over tokens, T2 levels follow a random token order). |
| 4. Training and stop rules | PASS | All 12 primary runs present (and 12 replication, 6 T = 8). stop_reason = patience in all 30; replaying the stop rule from train-history.csv (improvement iff val loss < best - 1e-6, patience 10) gives the same best_step, steps_run and best loss in all 30. Primary training time 16.9-75.6 min (cap 140); T = 8 max 93.4 min (cap 130). result.json args equal the Sec. 10 / 11 launch configuration in all 30 runs (eval_every 1000, patience 10, max_steps 400000, max_minutes 140 / 130, K 16, Mmax 16, n_val_inst and n_test_inst 1000, n_pairs 2000, B 256, B2 64, lr 1e-3, ema 0.995, stiff_steps 3000, key_steps 2000, T 6 / 8). Same torch 2.14.1+cu130 and NVIDIA L4 on all 16 rentals. PREREG Sec. 3 says validation every 500 steps; Sec. 10 and every run use 1000 (L8). |
| 5. GPU spend and clean-up | PASS (true billing CANNOT-CHECK) | logical-soup/GPU-LOG.md, lines with vm=soup-gpu-eca*: 18 DELETE / COST-AUDIT lines (16 DELETE, 2 COST-AUDIT at $0.00 for the two stockout create failures), at most one per VM; sum of cost_usd = $17.02 = eca/GPU-LOG.md total = RESULTS.md Sec. 1 and 11. 16 CREATE lines, all in us-central1 (12 in -a, 2 in -b, 2 in -c); every created VM has a DELETE line. `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca'` at 04:12:10Z: empty (the same listing without the filter returned 15 RUNNING instances, so the call worked). guard-eca.sh log: 0 AGE-KILL / CAP-KILL lines (last STATUS 04:13:46Z, total_est 17.02). The cost is ceiling rate x VM age as logged; actual billing could not be checked (Cloud Billing API disabled). |
| 6. Other overstatements | FAIL (wording; no verdict affected) | Three TL;DR sentences are contradicted or overstated by the re-derived numbers (M1-M3), and the label behind the one borderline D1 verdict does not match its description (M4); details and numbers under Findings. CIs are over the 3 seeds as pre-registered, and every paired comparison is on identical instances (pstar identical across runs, instance ids 0-999), so no CI is over the wrong unit and no comparison mixes instance sets. |

## Re-derived numbers

All numbers below come from the audit's own scripts (/tmp/eca/audit/rederive.py, check1b_d1.py, make_tables.py) on results/raw/main2; KL computed exactly (log-sigmoid). "cut" = mean over seeds of 1 - KL(arm) / KL(reference); intervals are paired 95% t-intervals over the 3 seeds (2 df, t = 4.303).

#### D2 per seed: mean test KL over the 1,000 chain instances (s0 / s1 / s2), from test-estimates.csv

| cell | declared P2 (val log loss stiff / voi) | family | P0_logit | P2 lw M=16 | rand lw M=16 | P2 lw M=0 |
|---|---|---|---|---|---|---|
| r110-T1 | voi (0.3018 / 0.2878) | F | 0.0287 / 0.0273 / 0.0174 | 0.1799 / 0.1799 / 0.1901 | 0.0920 / 0.0881 / 0.0861 | 0.2381 / 0.2196 / 0.1807 |
| r110-T1 | voi (0.3018 / 0.2878) | B1 | 0.0949 / 0.0941 / 0.0999 | 0.0899 / 0.1045 / 0.1041 | 0.0538 / 0.0560 / 0.0600 | 0.1264 / 0.1414 / 0.1543 |
| r110-T1 | voi (0.3018 / 0.2878) | B2 | 0.0705 / 0.0586 / 0.0765 | 0.0877 / 0.0827 / 0.1059 | 0.0511 / 0.0486 / 0.0506 | 0.1080 / 0.1039 / 0.1154 |
| r110-T2 | voi (0.3034 / 0.2852) | F | 0.0175 / 0.0254 / 0.0130 | 0.1630 / 0.1692 / 0.1727 | 0.0874 / 0.0974 / 0.0892 | 0.1957 / 0.2257 / 0.1799 |
| r110-T2 | voi (0.3034 / 0.2852) | B1 | 0.0986 / 0.0947 / 0.0709 | 0.1133 / 0.1068 / 0.0951 | 0.0728 / 0.0759 / 0.0513 | 0.1397 / 0.1630 / 0.1280 |
| r110-T2 | voi (0.3034 / 0.2852) | B2 | 0.0627 / 0.0557 / 0.0544 | 0.0832 / 0.0835 / 0.0844 | 0.0606 / 0.0524 / 0.0588 | 0.0973 / 0.1003 / 0.0905 |
| r90-T1 | voi (0.2409 / 0.2099) | F | 0.0040 / 0.0083 / 0.0056 | 0.0324 / 0.0476 / 0.0192 | 0.0299 / 0.0330 / 0.0253 | 0.0312 / 0.0274 / 0.0281 |
| r90-T1 | voi (0.2409 / 0.2099) | B1 | 0.0043 / 0.0153 / 0.0105 | 0.0312 / 0.0765 / 0.0734 | 0.0313 / 0.0304 / 0.0311 | 0.0323 / 0.0247 / 0.0475 |
| r90-T1 | voi (0.2409 / 0.2099) | B2 | 0.0056 / 0.0099 / 0.0101 | 0.0477 / 0.0522 / 0.0533 | 0.0325 / 0.0383 / 0.0381 | 0.0175 / 0.0268 / 0.0197 |
| r90-T2 | voi (0.2413 / 0.2188) | F | 0.0070 / 0.0091 / 0.0019 | 0.0279 / 0.0260 / 0.0721 | 0.0296 / 0.0267 / 0.0424 | 0.0235 / 0.0155 / 0.0415 |
| r90-T2 | voi (0.2413 / 0.2188) | B1 | 0.0090 / 0.0146 / 0.1372 | 0.0385 / 0.0404 / 0.1499 | 0.0316 / 0.0309 / 0.0728 | 0.0333 / 0.0149 / 0.1575 |
| r90-T2 | voi (0.2413 / 0.2188) | B2 | 0.0035 / 0.0147 / 0.0131 | 0.0514 / 0.0654 / 0.0603 | 0.0362 / 0.0516 / 0.0314 | 0.0174 / 0.0294 / 0.0293 |

#### D2 3-seed means: relative cuts (mean of per-seed cuts) and paired 95% t-intervals (2 df, t = 4.303)

| cell | family | cut vs P0 | KL(P0) - KL(P2) | (a) | cut vs P4-Rand | KL(P4) - KL(P2) | (b) | cut vs P2 M=0 |
|---|---|---|---|---|---|---|---|---|
| r110-T1 | F | -6.915 [-13.363, -0.467] | -0.1588 [-0.1887, -0.1289] | FAIL | -1.068 [-1.386, -0.751] | -0.0946 [-0.1154, -0.0737] | FAIL | 0.124 [-0.263, 0.512] |
| r110-T1 | B1 | -0.033 [-0.236, 0.169] | -0.0032 [-0.0224, 0.0160] | FAIL | -0.757 [-1.002, -0.511] | -0.0429 [-0.0584, -0.0273] | FAIL | 0.292 [0.211, 0.372] |
| r110-T1 | B2 | -0.347 [-0.570, -0.124] | -0.0236 [-0.0388, -0.0083] | FAIL | -0.838 [-1.388, -0.287] | -0.0420 [-0.0708, -0.0133] | FAIL | 0.158 [-0.006, 0.322] |
| r110-T1 | D2 overall | | | | | | FAIL | |
| r110-T2 | F | -8.760 [-17.063, -0.457] | -0.1497 [-0.1715, -0.1279] | FAIL | -0.846 [-1.096, -0.596] | -0.0770 [-0.0918, -0.0621] | FAIL | 0.152 [-0.111, 0.416] |
| r110-T2 | B1 | -0.206 [-0.497, 0.085] | -0.0170 [-0.0328, -0.0012] | FAIL | -0.606 [-1.171, -0.040] | -0.0384 [-0.0552, -0.0216] | FAIL | 0.263 [0.069, 0.458] |
| r110-T2 | B2 | -0.459 [-0.751, -0.168] | -0.0261 [-0.0384, -0.0138] | FAIL | -0.467 [-0.750, -0.184] | -0.0264 [-0.0371, -0.0157] | FAIL | 0.127 [-0.004, 0.257] |
| r110-T2 | D2 overall | | | | | | FAIL | |
| r90-T1 | F | -4.759 [-10.577, 1.058] | -0.0271 [-0.0592, 0.0050] | FAIL | -0.095 [-0.943, 0.753] | -0.0037 [-0.0295, 0.0221] | FAIL | -0.154 [-1.481, 1.173] |
| r90-T1 | B1 | -5.414 [-8.439, -2.390] | -0.0503 [-0.1009, 0.0002] | FAIL | -0.956 [-3.034, 1.122] | -0.0294 [-0.0932, 0.0344] | FAIL | -0.871 [-3.617, 1.876] |
| r90-T1 | B2 | -5.360 [-10.003, -0.717] | -0.0426 [-0.0440, -0.0411] | FAIL | -0.410 [-0.543, -0.278] | -0.0148 [-0.0167, -0.0129] | FAIL | -1.462 [-2.566, -0.357] |
| r90-T1 | D2 overall | | | | | | FAIL | |
| r90-T2 | F | -14.273 [-65.240, 36.694] | -0.0360 [-0.1098, 0.0377] | FAIL | -0.206 [-1.267, 0.856] | -0.0091 [-0.0533, 0.0351] | FAIL | -0.535 [-1.281, 0.211] |
| r90-T2 | B1 | -1.714 [-5.679, 2.251] | -0.0227 [-0.0445, -0.0008] | FAIL | -0.528 [-1.676, 0.620] | -0.0312 [-0.1301, 0.0678] | FAIL | -0.610 [-3.014, 1.795] |
| r90-T2 | B2 | -6.929 [-21.577, 7.719] | -0.0486 [-0.0533, -0.0439] | FAIL | -0.536 [-1.380, 0.309] | -0.0193 [-0.0399, 0.0013] | FAIL | -1.413 [-2.610, -0.216] |
| r90-T2 | D2 overall | | | | | | FAIL | |

#### D0 (validation P0 KL from result.json P0.val, 3-seed means) and D3a (ratios recomputed from the stored drifts)

| cell | F | B1 | B2 | S1 | D0 | D3a ratio m=5 | D3a ratio m=2 | D3a | max abs(recomputed - stored ratio) |
|---|---|---|---|---|---|---|---|---|---|
| r110-T1 | 0.0252 | 0.0804 | 0.0585 | 0.0077 | FAIL | 0.043 [0.031, 0.056] | 0.133 [0.101, 0.165] | PASS | 0.0e+00 |
| r110-T2 | 0.0196 | 0.0769 | 0.0478 | 0.0077 | FAIL | 0.049 [0.042, 0.057] | 0.153 [0.112, 0.195] | PASS | 0.0e+00 |
| r90-T1 | 0.0060 | 0.0100 | 0.0072 | 0.0084 | FAIL | 0.017 [0.008, 0.026] | 0.051 [0.027, 0.074] | PASS | 0.0e+00 |
| r90-T2 | 0.0061 | 0.0525 | 0.0094 | 0.0209 | FAIL | 0.024 [-0.008, 0.057] | 0.073 [-0.013, 0.160] | PASS | 0.0e+00 |

#### D3b (dd pool, result.json; 3-seed means)

| cell | family | KL both | KL fwd | KL bwd | log loss both | log loss fwd | log loss bwd | verdict |
|---|---|---|---|---|---|---|---|---|
| r110-T1 | B1 | 0.0933 | 0.1250 | 0.1345 | 0.1771 | 0.2101 | 0.2151 | PASS |
| r110-T1 | B2 | 0.1028 | 0.1341 | 0.1605 | 0.1932 | 0.2256 | 0.2498 | PASS |
| r110-T1 | D3 overall (a and b) | | | | | | | PASS |
| r110-T2 | B1 | 0.1045 | 0.1387 | 0.1254 | 0.1926 | 0.2312 | 0.2098 | PASS |
| r110-T2 | B2 | 0.1174 | 0.1349 | 0.1604 | 0.1972 | 0.2193 | 0.2303 | PASS |
| r110-T2 | D3 overall (a and b) | | | | | | | PASS |
| r90-T1 | B1 | 0.0494 | 0.1461 | 0.0824 | 0.2258 | 0.3347 | 0.2518 | PASS |
| r90-T1 | B2 | 0.0329 | 0.0349 | 0.0434 | 0.0722 | 0.0764 | 0.0846 | PASS |
| r90-T1 | D3 overall (a and b) | | | | | | | PASS |
| r90-T2 | B1 | 0.0978 | 0.2023 | 0.1460 | 0.2775 | 0.3925 | 0.3121 | PASS |
| r90-T2 | B2 | 0.0579 | 0.0412 | 0.0743 | 0.1014 | 0.0768 | 0.1263 | FAIL |
| r90-T2 | D3 overall (a and b) | | | | | | | FAIL |

#### D1 per run: from d1-pairs.csv (own rank code) vs result.json

| run | Spearman csv | Spearman json | AUC csv | AUC json | label agree (own geometry) | max abs(exact_H csv - indep) | oracle AUC (exact IG) |
|---|---|---|---|---|---|---|---|
| r110-T1-s0 | 0.9497 | 0.9495 | 0.8688 | 0.8688 | 2000/2000 | 5.0e-07 | 0.8669 |
| r110-T1-s1 | 0.9456 | 0.9452 | 0.8723 | 0.8723 | 2000/2000 | 5.0e-07 | 0.8687 |
| r110-T1-s2 | 0.9508 | 0.9506 | 0.8712 | 0.8712 | 2000/2000 | 5.0e-07 | 0.8752 |
| r110-T2-s0 | 0.9464 | 0.9462 | 0.8738 | 0.8738 | 2000/2000 | 5.0e-07 | 0.8669 |
| r110-T2-s1 | 0.9396 | 0.9395 | 0.8672 | 0.8672 | 2000/2000 | 5.0e-07 | 0.8687 |
| r110-T2-s2 | 0.9507 | 0.9506 | 0.8665 | 0.8665 | 2000/2000 | 5.0e-07 | 0.8752 |
| r90-T1-s0 | 0.9197 | 0.9209 | 0.8582 | 0.8582 | 2000/2000 | 5.0e-07 | 0.8907 |
| r90-T1-s1 | 0.9139 | 0.9134 | 0.8213 | 0.8213 | 2000/2000 | 5.0e-07 | 0.8753 |
| r90-T1-s2 | 0.9358 | 0.9354 | 0.8633 | 0.8633 | 2000/2000 | 5.0e-07 | 0.8947 |
| r90-T2-s0 | 0.9280 | 0.9291 | 0.8657 | 0.8657 | 2000/2000 | 5.0e-07 | 0.8907 |
| r90-T2-s1 | 0.9323 | 0.9313 | 0.8431 | 0.8431 | 2000/2000 | 5.0e-07 | 0.8753 |
| r90-T2-s2 | 0.9140 | 0.9144 | 0.8614 | 0.8614 | 2000/2000 | 5.0e-07 | 0.8947 |

#### D1 3-seed means (result.json values)

| cell | Spearman(beta, ln2 - H) | AUC(beta; blanket) | verdict | strict (CI lower bounds) | Spearman(ln2 - h(q)) |
|---|---|---|---|---|---|
| r110-T1 | 0.9484 [0.9413, 0.9556] | 0.8708 [0.8663, 0.8752] | PASS | PASS | 0.9487 [0.9380, 0.9595] |
| r110-T2 | 0.9454 [0.9316, 0.9593] | 0.8692 [0.8592, 0.8792] | PASS | PASS | 0.9459 [0.9417, 0.9502] |
| r90-T1 | 0.9232 [0.8954, 0.9511] | 0.8476 [0.7906, 0.9046] | FAIL | FAIL | 0.9338 [0.9285, 0.9390] |
| r90-T2 | 0.9249 [0.9021, 0.9477] | 0.8567 [0.8269, 0.8866] | PASS | FAIL | 0.9321 [0.9099, 0.9542] |

#### D1 label: share of blanket positives that are positive only through a child triple whose rule output does not depend on the target (given the other two parents' true values), and AUCs on a rule-aware label

| cell | uninformative share of positives | positives with exact H > 0.1 | non-blanket pairs with exact H < 0.1 | AUC beta, PREREG label | AUC beta, rule-aware label | AUC exact IG, PREREG label | AUC exact IG, rule-aware label |
|---|---|---|---|---|---|---|---|
| r110-T1 | 0.115 | 0.032 | 0.455 | 0.8708 | 0.8704 | 0.8703 | 0.8759 |
| r110-T2 | 0.115 | 0.032 | 0.455 | 0.8692 | 0.8663 | 0.8703 | 0.8759 |
| r90-T1 | 0.063 | 0.022 | 0.452 | 0.8476 | 0.8502 | 0.8869 | 0.8945 |
| r90-T2 | 0.063 | 0.022 | 0.452 | 0.8567 | 0.8590 | 0.8869 | 0.8945 |

#### Headline secondary numbers (P1 arm = random-order chains + plain mixture; test KL, 3-seed means)

| cell | family | P0 | rand mix M=2 (cut) | rand mix M=8 (cut) | rand mix M=16 (cut) | KL(P0) - KL(rand mix M=16) |
|---|---|---|---|---|---|---|
| r110-T1 | B1 | 0.0963 | 0.0702 (0.271 [0.231, 0.311]) | 0.0470 (0.512 [0.465, 0.560]) | 0.0481 (0.501 [0.441, 0.560]) | 0.0482 [0.0404, 0.0561] |
| r110-T1 | B2 | 0.0685 | 0.0509 (0.253 [0.108, 0.399]) | 0.0387 (0.432 [0.332, 0.532]) | 0.0399 (0.411 [0.218, 0.603]) | 0.0286 [0.0065, 0.0507] |
| r110-T1 | F | 0.0245 | 0.0169 (0.281 [-0.161, 0.724]) | 0.0326 (-0.416 [-1.728, 0.896]) | 0.0494 (-1.146 [-3.093, 0.801]) | -0.0249 [-0.0483, -0.0015] |
| r110-T2 | B1 | 0.0881 | 0.0685 (0.223 [0.200, 0.245]) | 0.0506 (0.424 [0.382, 0.465]) | 0.0491 (0.441 [0.392, 0.490]) | 0.0390 [0.0189, 0.0591] |
| r110-T2 | B2 | 0.0576 | 0.0434 (0.248 [0.169, 0.326]) | 0.0389 (0.324 [0.233, 0.416]) | 0.0414 (0.280 [0.213, 0.348]) | 0.0162 [0.0097, 0.0227] |
| r110-T2 | F | 0.0186 | 0.0145 (0.210 [0.058, 0.362]) | 0.0283 (-0.626 [-1.825, 0.572]) | 0.0437 (-1.537 [-3.677, 0.602]) | -0.0251 [-0.0425, -0.0076] |

The same readout on the replication batch (original runs, capped; my re-derivation): rule-110 B1 cut at M = 16 0.470 [0.197, 0.743] (T1) and 0.504 [0.456, 0.552] (T2); B2 0.364 [0.108, 0.620] and 0.411 [0.187, 0.634].

#### Full scan: every (selector, pool, M > 0) whose paired 95% CI of KL(P0) - KL(arm) lies above 0 (100 combinations per family and cell, uncorrected)

| cell | F | B1 | B2 | R |
|---|---|---|---|---|
| r110-T1 | none | 14: rand with mix (M 1-16), lw (8, 16), dd (16), g0 (1-8); voi with mix and g0 at M = 1 (cut 0.03-0.04) | 6: rand mix (M 2, 4, 8, 16); **stiff mix M = 4 (cut 0.110) and M = 8 (0.124)** | 5: rand mix (M 1, 2, 4); stiff mix and g0 at M = 1 (0.02-0.03) |
| r110-T2 | 1: **rand mix M = 1 (cut 0.181)** | 9: rand with mix (M 1-16), g0 (1-4), lw (16) | 13: rand mix (M 1-16), g0 (1, 2); **stiff mix M = 4 (0.086), M = 16 (0.171)**, stiff g0 M = 1, 4 (0.04, 0.07); conf mix and g0 M = 4 (0.05-0.06) | none |
| r90-T1 | none | none | 1: stiff mix M = 16 (0.0085 -> 0.0084) | none |
| r90-T2 | none | none | 3: stiff g0 M = 1, 8, voi g0 M = 2 (cuts 0.005-0.034) | none |

#### Check 0: non-finite cells per estimate file (count of "+inf" logits; no "-inf", no NaN anywhere)

| batch | runs | rows | "+inf" cells | columns affected per run | result.json non-finite numbers per run |
|---|---|---|---|---|---|
| main (original, float32, replication) | 12 | 48,000 | 694,465 | 50-61 of 122 | 496-616 |
| t8 (float32, secondary) | 6 | 24,000 | 201,621 | 55-60 of 122 | 414-446 |
| main2 (eca-b, float64, primary) | 12 | 48,000 | 0 | 0 | 0 |

#### Check 4: primary runs (result.json; replayed from train-history.csv, all "replay OK")

| run | stop_reason | best_step | steps_run | train minutes |
|---|---|---|---|---|
| r110-T1-s0 | patience | 105000 | 115000 | 30.5 |
| r110-T1-s1 | patience | 66000 | 76000 | 19.8 |
| r110-T1-s2 | patience | 76000 | 86000 | 22.7 |
| r110-T2-s0 | patience | 99000 | 109000 | 56.8 |
| r110-T2-s1 | patience | 103000 | 113000 | 51.6 |
| r110-T2-s2 | patience | 170000 | 180000 | 75.6 |
| r90-T1-s0 | patience | 62000 | 72000 | 18.5 |
| r90-T1-s1 | patience | 87000 | 97000 | 25.5 |
| r90-T1-s2 | patience | 54000 | 64000 | 16.9 |
| r90-T2-s0 | patience | 142000 | 152000 | 60.3 |
| r90-T2-s1 | patience | 104000 | 114000 | 55.0 |
| r90-T2-s2 | patience | 82000 | 92000 | 44.4 |

#### Check 5: GPU (logical-soup/GPU-LOG.md, own parser)

| item | value |
|---|---|
| DELETE / COST-AUDIT lines for vm=soup-gpu-eca* | 18 (16 DELETE, 2 COST-AUDIT for the stockout create failures at $0.00); no VM has two |
| sum of cost_usd | $17.02 (smoke 0.37; original batch 0.95 + 1.01 + 0.99 + 0.98 + 1.00 + 0.98; T = 8 1.42 + 1.34 + 1.63; eca-b 1.10 + 0.97 + 1.30 + 1.10 + 1.01 + 0.87) |
| eca/GPU-LOG.md, RESULTS.md | $17.02, $17.02 |
| CREATE lines / zones | 16; us-central1-a 12, -b 2, -c 2 |
| live eca VMs | none (gcloud, 04:12:10Z) |

## Findings, ranked by severity

No high-severity finding: every pre-registered verdict, every table cell and every number in summary.json reproduces, and no leak was found.

**Medium**

- **M1. The TL;DR's search headline rests on a post hoc readout that RESULTS does not label as post hoc.** "Random-order sampling does [help] ... the P1 arm ... cuts rule-110 bridge KL by 30-50%" (RESULTS line 9) comes from the mixture-vs-P0 readout that analyze.py gained at 03:47Z "after the replication batch and 4 primary runs had been read" (PREREG Sec. 9); /tmp/eca/msg-0327.txt (03:27Z) already reports the pattern from the replication batch. P1 itself was a pre-registered arm (PREREG Sec. 5), but no decision rule covers it, and it is one of 100 selector x pool x M readouts per family. The primary batch is not a clean held-out confirmation either: by result.json times 9 of the 12 primary runs had finished by 03:47Z (PREREG says 4 had been read). The effect itself is large and consistent: all four rule-110 bridge cells of the primary batch (cuts 0.28-0.50 at M = 16, paired KL-difference CIs above 0) and the replication (0.36-0.50). RESULTS mentions "secondary readout added 03:50Z" only for the D1 oracle AUC (line 118); lines 9, 92-95 and 147-148 present the P1 result without such a label.
- **M2. "Stiffness-first, VoI and confidence-first chains stay flat or get worse" (line 9), "do not improve bridges under either aggregator" (line 96) and "not through stiffness-guided routing" (lines 147-148) are contradicted on rule-110 B2.** Stiffness-first chains with the plain mixture beat one pass with paired CIs above 0 in both arms: r110-T1 M = 4 cut 0.110 and M = 8 cut 0.124 (KL 0.0685 -> 0.0604); r110-T2 M = 4 cut 0.086 and M = 16 cut 0.171 (0.0576 -> 0.0478). Confidence-first with the mixture at M = 4 on r110-T2 B2: cut 0.056. These gains are a third to a half of random order's (0.28-0.43), and some single hits among 100 uncorrected comparisons may be chance, but the stiff-mix B2 gain appears in both training arms. The B1 statement in line 96 (mixture KL 0.09-0.12 for T1) is accurate.
- **M3. "Where one pass is already good (rule 90, rule-110 forward queries), every search arm makes things worse" (line 9) contradicts RESULTS Sec. 4 (line 94) and the numbers.** On rule-110 F, random-order chains with the mixture lower the mean KL at M = 1-2 in both arms: T1 M = 2 0.0245 -> 0.0169 (cut 0.281, CI includes 0); T2 M = 2 0.0186 -> 0.0145 (cut 0.210 [0.058, 0.362]); T2 M = 1 paired KL-difference CI above 0 (cut 0.181). They get worse only at larger M (M = 16: 0.049 and 0.044). For rule 90 the statement holds to within 3% (largest significant gain: B2 cut 0.034).
- **M4. The one borderline D1 verdict depends on a label that does not match its description, and the "ceiling" wording is wrong.** The pre-registered label ("S contains a valid deterministic local blanket of i ... or for one of i's 3 children, the child plus its other 2 parents") counts child triples whose rule output does not depend on i given the other two parents' observed values: for rule 90 (child = left XOR right) the triple of the child directly below i never informs i. Recomputed from the regenerated (grid, target, S): such triples are the only route to a positive label for 6.3% of rule-90 and 11.5% of rule-110 positives. With a rule-aware label (a triple counts only if the child's rule output flips with i), r90-T1's 3-seed AUC is 0.8502 instead of 0.8476, i.e. D1 for r90-T1 flips from FAIL to a marginal PASS; the other three cells keep their verdicts (0.8704, 0.8663, 0.8590). The verdict as reported applies the pre-registered text correctly. Separately, "a ceiling of 0.87 and 0.89 that even the exact information gain cannot beat on this label" (line 11) and "rule 110's beta_int is at the ceiling" (line 118): the exact-IG AUC is not an upper bound for other scores on this label; beta's AUC exceeds it on the same pairs in r110-T1-s0 (0.8688 vs 0.8669), r110-T1-s1 (0.8723 vs 0.8687) and r110-T2-s0 (0.8738 vs 0.8669). The 43-49% figure (non-blanket pairs with exact H < 0.1) reproduces (my range 0.428-0.487).

**Low**

- **L1. Rule-90 T2 is unstable across seeds, beyond the "one seed on B1" in the TL;DR (line 8).** In 4 of 9 rule-90 T2 runs (primary s2; replication s1 and s2; T = 8 s1) one-pass validation KL on a bridge family is 0.069-0.143 and S1 is 0.038-0.073 (the one-step check fails); none of the 9 rule-90 T1 runs exceeds 0.015 on a bridge or 0.011 on S1. These runs also have higher best validation outcome loss than the other seeds of their cell (0.1810 vs 0.1752-0.1758; 0.1819 and 0.2140 vs 0.1793; 0.1835 vs 0.1549-0.1579). The T = 8 sentence is scoped to T1 and is correct as written; the T = 8 T2 3-seed mean on B2 is 0.033.
- **L2. Sec. 3 note (line 65): "the bridges clear the bar (B1 0.08, B2 0.05-0.06)"** — r110-T2's validation B2 KL is 0.0478, below the 0.05 bar (verdict unchanged: FAIL).
- **L3. "cuts rule-110 bridge KL by 30-50%" (line 9)** — the four cuts at M = 16 are 0.501, 0.411, 0.441, 0.280; the range is 28-50%.
- **L4. Sec. 7 (line 139): "The mixture wins where claims are soft (F, R); the learned pools win where claims are stiff (B1, B2)"** holds in r110-T1 only. On the declared P2 chains at M = 16 the mixture has the lowest KL on all four families in r110-T2 (B1 0.0947 vs lw 0.1051; B2 0.0738 vs 0.0837); lw is lowest on F and B1 in r90-T1 and on B1 only in r90-T2.
- **L5. "checked against brute force ... (max difference 3.8e-15)" (line 24, PREREG Sec. 1).** Re-running the same check gives 5.3e-14 (largest: rule 90, eps 0.05). Both are float64 round-off; the conclusion stands.
- **L6. Small support: test, validation and cal grids mostly occur verbatim in the training set.** No split leak (disjoint simulator seeds), but 81.5-82.1% of test grids, 81.1-82.5% of val grids and 80.0-82.3% of cal grids appear exactly in the 50,000 training grids (14,324 / 14,489 distinct training grids for rules 110 / 90; about 55% of grids, 0.985^40, have no noise flip). Two statements read more strongly than the data allow: the cal split giving "held-out surprise" (Sec. 10) and "already does the multi-step inference in one pass" (line 144), which this design cannot separate from fitting a small, heavily sampled support.
- **L7. D3 passes with thin margins and D3a cannot separate selective dedup from uniform crowding.** Forward/backward key cosines (3-seed means 0.83-0.97) overlap the redundant-view cosines (0.96-0.99), so Gram crowding also divides complementary claims; RESULTS says this for D3b. "Key fingerprints deduplicate redundant views well" (line 151) is true for the D3a test as defined. Two of the six passing D3b family comparisons have paired CIs that include 0: r110-T2 B1 (KL margin 0.0196 [-0.0015, 0.0407]) and r90-T1 B2 (0.0020 [-0.0024, 0.0064]); the rule uses 3-seed means, so the verdicts stand.
- **L8. Small deviations in the D1 construction and the PREREG text.** S is drawn with k ~ U{1..47} and the target is then removed, so 47-51% of pairs have k - 1 reveals and one seed-0 pair has an empty S (PREREG: "k ~ U{1..47}, i not in S"). PREREG Sec. 3 says validation every 500 steps; Sec. 10 and all runs use 1000.
- **L9. analyze.py presentation of non-finite statistics (secondary tables only).** ci() drops non-finite seeds silently: the replication's r90-T2 D3b B2 "both" KL (0.089 in Sec. 9) and log loss are means over 2 seeds with no count shown; the verdict itself is correctly NOT EVALUABLE. The D3b conjunction returns NOT EVALUABLE when one family is FAIL and the other is non-finite; no reported verdict is affected.
- **L10. The fix is not only precision.** Besides float64, it moves logit_np's clip from 1e-9 to 1e-12 (claim logits capped at +-27.6 instead of +-20.7 inside the pool features and fitted scales) and the CSV precision. All evaluation-side and declared in PREREG Sec. 12; the replication therefore differs from the primary runs in this cap as well as in float32.
- **L11. CSV precision limits re-derivation of the D1 secondary numbers.** d1-pairs.csv (beta 5 dp, exact_H 6 dp) reproduces result.json's Spearman only to 1.1e-3, and the published oracle AUC (computed from the rounded exact_H) is 0.8705 / 0.8866 against 0.8703 / 0.8869 with full-precision exact H.
- **L12. Infrastructure note (line 202): "Four original rentals lost the END line"** — 7 launch logs lack it (eca-r110-s0, eca-r90-s0, -s1, -s2, eca-t8-s0, -s1, -s2); all 7 have a "Deleted" line and a GPU-LOG DELETE line.
- **L13. eca/ecanet.py no longer matches the code that produced the results.** At 04:26:06Z, after FINAL-READY, eca/ecanet.py gained round-2 code (a keyW2 layer, key_mode, source_mask, attn_vec and a v2 key path; md5 4bef0c4e, checked 04:26:19Z). Default behaviour stays "v1", but the extra layer changes the parameter count and the global RNG draws at initialisation, so a rerun from the tree is not the as-run network. The as-run file is the frozen copy (/tmp/eca/src-eca-b-*/eca/ecanet.py, md5 db3e6121, 6 of 6 copies); RESULTS.md Sec. 12 lists `ecanet.py` without that distinction. No result file changed (raw tree and summaries rechecked below).

## Could not check

- Actual GPU billing: the ledger is ceiling rate x VM age from GPU-LOG.md (Cloud Billing API disabled per eca/GPU-LOG.md).
- Training-side reproducibility: the models are not kept and no GPU was used, so chain sampling, head fitting and pool fits could only be checked through the saved outputs, not re-executed.
- What the builder read and when (e.g. "declared at 02:35Z, after reading one finished run", "K = 16 fixed at 01:15Z", "smoke: only validation D0 read"): only file times and the message files under /tmp/eca support these. The smoke copy's --K default of 4 is consistent with K = 16 being set after the smoke code was frozen.
- The numpy version on the VMs (not logged); it does not matter for the data (pre-generated, shipped and identical everywhere) and the D1 pairs regenerate exactly.

## Commands and scripts (all under /tmp/eca/audit/; CPU only, one process at a time, PYTHONPATH=/tmp/smallnet/pylib)

- `run_all.sh final2` re-runs everything below on the final tree (outputs in out-final2/).
- `check0_inf.py`: non-finite counts per column in every test-estimates.csv. `diff /tmp/eca/ecarun-as-run-0135Z.py eca/ecarun.py`; md5sum of eca/*.py, smallnet/netdef.py and data/* in all 18 /tmp/eca/src-eca-*/ copies.
- `ECA_DATA=/tmp/eca/audit/data-regen python3 eca/ecadata.py` (brute-force check + regeneration, T = 6) and `ECA_DATA=/tmp/eca/audit/data-regen8 python3 eca/ecadata.py 8`; `check1e_regen_compare.py` (array equality with /tmp/eca/data, split overlap).
- `check1e_exact.py`: independent clamped-forward posterior vs ecadata.posterior_marginals vs the pstar column.
- `check1b_d1.py`: regenerates (grid, target, S) per seed, own blanket / parents labels, own exact H, Spearman / AUC from d1-pairs.csv, rule-aware label.
- `rederive.py` (KLMODE=exact or clip; RAW2 / RAW1 / TAG select the batch): independent D0-D3 and curves, compared with summary.json; `categorize.py` groups the exact-mode differences; `render_mine.py` + `diff_tables.sh` re-render tables.md / verdicts.md from my numbers and diff them.
- `check3_perm.py`, `check4_stop.py` (stop-rule replay and args), `check5_gpu.py` (GPU-LOG parser); `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca'` (04:12:10Z).
- `claims.py`, `claims2.py` (numbers behind RESULTS prose; full significance scan), `make_tables.py` (the tables above).
- Final-file checksums: final-checksums.txt, final-raw-checksums.txt (rechecked at the end of the pass, below).

Final recheck at 2026-10-15T04:26:47Z: RESULTS.md (md5 33adc70f), GPU-LOG.md, PREREG.md, analyze.py, ecarun.py, ecadata.py, summary.json, tables.md, verdicts.md and the replication / T = 8 summaries match final-checksums.txt; the only changed file is eca/ecanet.py (L13); the 121 raw files match final-raw-checksums.txt; gcloud lists 0 soup-gpu-eca* instances.
