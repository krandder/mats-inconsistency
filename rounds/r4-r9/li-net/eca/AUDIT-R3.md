# E-CA round 3 independent audit: pre-registration timing, data, exact inference, code, leakage, rule re-derivation, GPU spend, prose

Auditor: Opus 5.5 (1M context; model ID claude-opus-5-5[1m]), independent pass, 2026-10-15T12:18Z. Started 12:00:14Z from a snapshot of the round-3 files taken at 12:01:12Z (/tmp/eca/audit-r3/snap-1203/, sha256 list /tmp/eca/audit-r3/snap-1203.sha256, 101 files: PREREG-R3.md, RESULTS-R3.md md5 ed5c1068 (mtime 12:00:04Z), analyze_r3.py, build_results_r3.py, ecadata3.py, ecadata.py, ecarun.py, ecanet.py, eca/GPU-LOG.md, results/r3/summary.json and tables.md, and all 90 raw files under results/raw/r3/). Re-checked at 12:14:33Z and 12:18:12Z: `sha256sum -c` reports all 101 files unchanged, RESULTS-R3.md is still md5 ed5c1068, PREREG-R3.md is still da122736, and there is no PREREG-R3-AMENDMENTS.md. Every statement below refers to that snapshot.

Who did what: checks 1, 2, 3, 7 and 8 are mine. Checks 4-5 (code, leakage) and check 6 (re-derivation) were run by two forked copies of this auditor (same model, own code, scratch /tmp/eca/audit-r3/forkA/ and /tmp/eca/audit-r3/forkB/). I read their outputs and cross-checked the overlapping facts with my own scripts: the per-run training table, D1 and information-gain Spearman, branch shares, the d1-pair replay and the frozen-code md5s. No automatic text classification was used. I read and checked every prose claim in RESULTS-R3.md Secs 1-9b by hand against the numbers, so no hand-label agreement rate applies.

**Summary (5 lines)**
1. Every verdict re-derives from the raw CSV/JSON with own code. R3-D0 PASSES for rule 110 (L = 4 selected; seed-0 validation S1 0.0064, F/B1/B2 0.130/0.353/0.250). Rule 90 passes at no depth on seed 0 (L = 4 S1 0.0153, 0.0003 over the bar), so its rules are descriptive. R3-D2a, R3-D2b and R3-D3 FAIL for rule 110, and rule 90 would fail all three. Own code agrees with 2,154 summary.json numbers within 3.9e-7. analyze_r3.py re-run in scratch reproduces tables.md, summary.json and both PNGs byte for byte. RESULTS-R3 Sec. 10 is tables.md verbatim. The exact t (4.30265) changes no verdict or star.
2. Pre-registration and data hold. PREREG-R3.md (md5 da122736, equal to the frozen copy; no amendments file) was last written at 09:47:03.484Z. That is 82 ms before the smoke's frozen source copy, 18 min before the main launches and 3 h 10 min before the first round-3 result reached General (10:56:45Z); analyze_r3.py was saved at 09:48:16Z. A fresh ecadata3 regeneration equals the shipped data in all 11 frozen copies. The split is 3072/384/320/320 with 0.000 row-0 and whole-grid overlap, and the F/B1/B2 geometry matches the spec exactly.
3. Exact inference and code hold. My own dense 4096 x 4096 forward-backward (checked against brute-force enumeration, 4.9e-14) agrees with ecadata3.fb_marginals to 9.7e-15 and with the pstar and exact_p columns to their CSV rounding (5e-11, 5e-10). `ecadata3.py check` passes (worst 9.1e-14). The round-2 to round-3 ecarun.py diff (17 hunks) holds only the listed features plus a memory-only batch-size change. Training, head and pool-fitting code are byte-identical to round 2, all 11 frozen copies equal the tree, and all 18 result.json args match PREREG. Exact posteriors appear only in scoring, the gate, D1 and R3-D3, so there is no leakage.
4. GPU: round 3 cost $13.66 (10 DELETE lines: 1 smoke + 9 main VMs; an 11th create attempt failed on stockout and made no instance). The night cost $33.56 (33 lines, 31 VMs). Both equal RESULTS-R3 and eca/GPU-LOG.md. gcloud lists no soup-gpu-eca* instance at 12:06:02Z or 12:18:12Z, and no soup-gpu* instance at 12:06:02Z. RESULTS-R3 Sec. 9b says "11 on-demand L4 VMs"; the log has 10.
5. Prose misreports no verdict and no rule number. There are 3 moderate items:
   - (M1) The stiffness head ranks pairs almost exactly like plain confidence (Spearman 0.97-0.99 per seed), so the R3-D3 null is close to built in. The D1 sentence leaves out that plain confidence tracks the exact information gain better than the head (0.937 vs 0.926 at the selected depth, in every seed).
   - (M2) "Removing memorisation did what gemini predicted" credits the overlap removal for the one-pass gap. W, T, the query rows and the depth all changed together, and no arm isolates the overlap.
   - (M3) The Gram-dedup sentence blames VoI chains for an excess that is the dd pool's own M = 0 offset.
   There are also 13 low items (rounding slips, "never branched", "11 VMs", a wrong outlier seed, spec deviations missing from Sec. 9).

## Verdict table

| check | result | evidence |
|---|---|---|
| 1a. PREREG-R3 saved before the first round-3 launch | **PASS** (82 ms before the smoke copy, 18 min before the main runs) | `stat -c '%n %y %z %w'` at 12:00Z: PREREG-R3.md mtime 09:47:03.4836Z, birth 09:47:03.4826Z; /tmp/eca/PREREG-R3-prelaunch.md birth 09:47:03.4916Z. The first round-3 launch is /tmp/eca/logs/launch-eca3-smoke.log line 1, `2026-10-15T09:47:03Z LAUNCH eca3-smoke ... src=/tmp/eca/src-eca3-smoke-1792057623`; that dir was born 09:47:03.5656Z (epoch 1792057623 = 09:47:03Z). The main LAUNCH lines run 10:05:17Z-10:05:33Z (9 logs). The a-s0 relaunch src dir dates from 10:10:07Z, because the first a-s0 attempt failed on stockout in every zone (GPU-LOG CREATE-FAILED 10:08Z). The src-eca3-* dirs were born 10:05:17.835Z-10:05:33.864Z, and the relaunch dir at 10:10:07.693Z. In logical-soup/GPU-LOG.md the smoke's PENDING-CREATE/CREATE is 09:47Z and its RUN 09:48Z; the first main CREATE is 10:08Z. The first round-3 result on General is results/raw/r3/r90-L4-s0/train-history.csv at 10:56:45Z (`find -printf '%T+'`). Caveat (L10): the header's "written 09:46Z" (RESULTS: "09:46:31Z", with the header corrected in the launch command) cannot be verified from metadata. The inode was created at 09:47:03.4826Z (an in-place rewrite) and no earlier copy exists. The final text is the one that was frozen. |
| 1b. Content = frozen copy; no later change | **PASS** | `md5sum` at 12:00Z, 12:14:33Z and 12:18:12Z: PREREG-R3.md and /tmp/eca/PREREG-R3-prelaunch.md are both da122736ec758f64d595c0ee4724461c, and `cmp` reports them identical. `ls eca/PREREG-R3-AMENDMENTS.md` returns "no such file" at all three times. |
| 1c. Analysed rules = PREREG rules | **PASS** | analyze_r3.py: md5 3521d1a6, mtime 09:48:16Z, 68 min before the first result; the builder dry-ran it on fake runs in /tmp/eca/fake-r3 at 09:48:24-33Z. It implements PREREG Secs 3 and 5 as follows. **Gate (Sec. 3):** gate_ok is S1 <= 0.015 and KL >= 0.08 (r110) / 0.05 (r90) on at least 2 of F/B1/B2, read from runs[(rule, L, 0)] P0.val only; the selected L is the largest passing one. **R3-D0:** 3-seed means at the selected L. **R3-D2a:** for each of stiff, conf and hyb on F, sw, M = 16, the mean per-seed cut is >= 0.25 with the paired CI of KL(P0) - KL(arm) above 0, and the mean per-seed cut vs rand is >= 0.15 with the paired CI of KL(rand) - KL(arm) above 0. **R3-D2b:** the hyb cut is >= 0.25 with CI above 0 on F, B1 and B2; the stiff - hyb CI is above 0 on B1 and B2; the rand - hyb CI is above 0 on F. **R3-D3:** max(Spearman beta, Spearman selfcons) - Spearman confidence >= 0.05 on 3-seed means. The one-pass KL comes from the P0_logit column, the network's own one-pass belief at the query; sw at M = 0 equals it within 2.9e-6 logit (fork B: 18 runs x 3 families x 6 selectors). PREREG itself departs from the spec in six places (L9). Those departures were pre-registered, and none changes a verdict. |
| 2. Data | **PASS** | **Script:** /tmp/eca/audit-r3/me/data_check.py, 12:04:08-12:04:35Z (data_check.log, data_check.json). It imports ecadata3 from the snapshot (md5 ae0f45f3, equal to the tree) with the cache redirected to a fresh dir (ECA_DATA=/tmp/eca/audit-r3/me/data-fresh). **(a) Split:** my own default_rng(30015010 + 12).permutation(4096), cut 3072/384/320/320, equals ecadata3.row0_buckets(12) and partitions all 4,096 states. **(b) Grids:** per rule, train 50,000, cal 4,000, val 2,000 and test 2,000, shape [n, 120]. Every grid's row 0 lies in its own split's bucket (1.000). Cal/val/test row 0 found in train is 0.000, and whole grid found in train is 0.000, for both rules. All four splits are pairwise row-0 disjoint. Distinct row-0 states: train 3072, cal 384, val 320, test 318 (r110) / 320 (r90). My own CA re-simulation (Wolfram index 4l + 2c + r, periodic) gives flip rates of 0.0144-0.0152 against eps 0.015. Test rows t >= 1 that occur among the training rows: 0.99994 (r110), 0.99989 (r90); PREREG says 99.99%. **(c) Shipped data = fresh:** all 8 grid files equal the fresh regeneration array for array, both in /tmp/eca/data3 and in all 11 frozen copies (/tmp/eca/src-eca3-*/data/: smoke, 9 main and the a-s0 relaunch). remote.sh sets ECA_DATA=/root/soup/data, so the VMs read these files. The 4 inst-*.npz files are equal across all 11 copies. Their xg equals the fresh grids, their *_rev and *_q equal a fresh ecadata3.family_masks for all 5 families, and fresh fb_marginals for the first 48 instances per family and split reproduce *_post exactly (0.0). **(d) Geometry,** from the shipped val and test instances of both rules: F reveals exactly row 0 (12 cells), with query rows {6, 7, 8, 9} and all 12 columns. B1 reveals exactly rows 0 and 9 (24 cells), with query rows {4, 5}. B2 reveals exactly row 0 cols 0-5, row 9, and columns 0-1 of every row (34 cells), with query rows {4, 5} and cols {3, ..., 10}. S1 has all 3 parents revealed in 2000/2000 instances. No query cell is revealed. **(e) Entropies:** mean exact H(Y \| E) on the 2,000 test instances is r110 F 0.5600, B1 0.2029, B2 0.2208, S1 0.0320, and r90 F 0.5235, B1 0.2803, B2 0.0673, S1 0.0278, matching PREREG Sec. 1 and RESULTS Sec. 2. |
| 3. Exact inference | **PASS** | **Script:** /tmp/eca/audit-r3/me/exact_check.py, 12:05:28-12:06:07Z (exact_check.log / .json), own code. State s = sum_c x_c 2^c; dense M[s, s'] = eps^d (1 - eps)^(12 - d) with d = popcount(F(s) XOR s'); uniform row-0 prior; hard evidence; forward-backward in float64. **(i)** My dense code against my brute-force enumeration of all grids (W 3-5, rules 110/90/30, eps 0.015/0.05): max 4.9e-14. **(ii)** W = 12, 40 test grids per rule with random masks, k ~ U{0..120}: dense vs fb_marginals 2.4e-15 (r110) / 9.7e-15 (r90), and vs fb_marginals_sweep 2.1e-15 / 9.7e-15, over all 120 cells. **(iii)** Chain test instances, first 80 per family (F, B1, B2) per rule: dense vs the shipped *_post (all 120 cells) <= 6.3e-15; dense vs the pstar column of test-estimates.csv <= 5.0e-11 in all 9 runs of each rule (pstar is printed with 10 decimals). The pstar column is identical across the 9 runs of a rule, and `inst` = 0..499 (the first 500 test instances). **(iv)** R3-D3/D1 pairs: replaying ecarun's sampling (default_rng(31000 + seed) for grid and target, random_reveal(2000, 32000 + seed), target unrevealed) reproduces the grid, target_grid_cell and reveal_count columns exactly for r110 and r90 L4-s0, L4-s1 and L2-s2. Dense vs exact_p for the first 120 pairs is <= 5.0e-10 (9-decimal CSV). **(v)** `python3 ecadata3.py check`, run from the snapshot copy at 12:03:40-12:03:49Z, rc 0: scatter+channel and sweep vs the dense round-1 code (W 8, 6) <= 1.67e-15, vs brute force (W 4, 5) <= 9.13e-14, WORST 9.126e-14. PREREG quotes "<= 1.6e-15" for the dense part; the check prints 1.67e-15 for rule 90 T6 W8, a rounding-level difference. Timing claims (0.020 s per grid at W = 12) were not re-measured: the box load was 26 on 16 cores. |
| 4a. Code diff vs round 2; training unchanged | **PASS** (1 low note: L13) | **Source:** fork A, 12:01:47-12:13:41Z, /tmp/eca/audit-r3/forkA/. **Baseline:** the 5 round-2 frozen copies (/tmp/eca/src-eca-r2*/eca/) are identical to each other (ecarun 99047322, ecanet 4bef0c4e, ecadata 6ef1d729). ecanet.py and ecadata.py have no change (empty diffs). **ecarun.py,** round 2 to round 3 (f06c7dda): 17 hunks, 303 diff lines (ecarun-r2-vs-r3.diff). Line numbers are round-3 lines:<br>- data-module switch, importlib from argv (36-41);<br>- measured passes (186, 196, 752-762, 812-814);<br>- hyb selector with HYB_H = 0.20 and HYB_STATS, with voi/voiu moved into a shared helper (218-231, 249-280);<br>- --data-module, --n-eval, --n-cal and --chain-fams (381-384, 398-399, 419-423; 3000 replaced by n_eval at 633, 654, 672 and 822);<br>- the R3-D3 diagnostic and its d1-pairs columns (680-744);<br>- one hunk outside the listed features: the Infer default chunk now shrinks with sequence length (1,310 at n = 120). It affects memory only and training does not use it (L13).<br>**Training unchanged:** the training section (96 lines: AdamW, masked BCE, k ~ U{1..119}, EMA, validation outcome log loss every 1,000 steps, patience, max_minutes), the post-hoc heads section (90 lines) and the pool/metric helpers (fit_u, fit_sw, sw_pool, pool_feature, rand_reveal) are byte-identical to round 2. **VoI refactor:** behaviour-preserving. After normalising indentation, the old inline block and the new helper are the same 23 lines (only the gate naming and the return differ), and the random draws happen in the same order. |
| 4b. Frozen copies = tree | **PASS** | All 11 /tmp/eca/src-eca3-*/ copies (9 main, the a-s0 relaunch, the smoke) match the tree: eca/*.py, remote.sh (7ebdf752), smallnet/netdef.py (ce9db98f, same as round 2) and the 12 data files (fork A frozen-md5.txt). My own `md5sum` at 12:14:33Z agrees: ecarun.py is f06c7dda in all 11 copies and the tree; ecanet 4bef0c4e, ecadata3 ae0f45f3 and ecadata 6ef1d729 in all 11. Each frozen jobs.txt equals /tmp/eca/jobs3/main-*.txt, and the smoke copy equals smoke.txt. The 9 pairs cover the 18 PREREG runs (a: r110 L4 + r90 L2; b: r90 L4 + r110 L2; c: r110 L3 + r90 L3; seeds 0-2). |
| 4c. result.json args = PREREG | **PASS** | **Args** (check_args.out): across the 18 runs only rule, layers, seed and out vary, each equal to its directory name. Constant: T 10, W 12, eps 0.015, arm T1, n_train 50,000, B 256, lr 1e-3, eval_every 1,000, patience 10, max_minutes 70, ema 0.995, tau 5.0, key_mode v2, selectors stiff,conf,voi,voiu,hyb,rand, K 16, Mmax 16, n_val_inst 500, n_test_inst 500, n_pairs 2,000, data_module ecadata3, n_eval 2,000, n_cal 4,000, chain_fams F,B1,B2. max_steps is 400,000, which PREREG does not set and no run reached. **Perm:** ecadata.make_perm(10, 12) in all 18 runs. **Models:** model.pt for r110-L4-s0 and r90-L4-s0 (4 blocks, 564,706 params) and for r90-L2-s0 and r110-L2-s0 (2 blocks, 299,746) load on CPU with no missing or unexpected keys; args equal result.json and the perm is the same. |
| 5. Leakage | **PASS** | **Fitting uses outcomes only** (fork A; ecarun.py round-3 line numbers):<br>- Training loss is BCE against the true grid values of load_grids("train") (468).<br>- Early stopping is BCE against true values on 2,000 validation grids with random masks (442-453, 500-514).<br>- The stiffness head is fitted by MSE to the realised BCE on cal grids (538-546).<br>- The key head is fitted by BCE to the true cell values on cal grids (570-605).<br>- fit_u (775-776), fit_sw (783-784) and the D3b fit (909) use validation outcomes.<br>**Selectors use only network outputs** (167-247): stiff samples by exp(tau beta); conf takes the max \|q - 0.5\|; hyb takes the minimum network entropy, with the voiu helper as fallback; voi/voiu use the network's VoI; rand is uniform. Chains sample materialised cells from the network's q.<br>**Exact posteriors appear only in reporting:** P0/D0 scoring (626-642), the D1 and R3-D3 pairs (658-722), test KL (804), D3b KL (912, 915) and test-estimates.csv (920-933). Parents and blankets appear only in the D1 labels (666-667) and the D3a views (828).<br>**Depth selection** (analyze_r3.py 96-100) reads only the seed-0 validation P0 KL. That is a KL to the exact posterior on validation, which PREREG Sec. 3 allows. No test number and no seed 1-2 enters it.<br>**Hybrid:** HYB_H = 0.20 (249), and it branches only when the minimum entropy is >= 0.20 (222-227). Branch share is branches / steps, reset per split, family and selector (752, 762). A CPU replay of 100 test instances per family (check_zstar_hyb.out) found:<br>- the minimum entropy over unrevealed cells never reached 0.20 (max 0.058-0.114 nats);<br>- 20-75 cells were below 0.20 at every step;<br>- the hybrid's pick equalled conf's in 100% of steps (99.8% on rule-90 B2).<br>So a branch share of about 0 follows from the code: the candidate set is every unrevealed non-query cell in the grid.<br>**R3-D3 z\*:** it is never revealed and never the target. A CPU recomputation agrees with the logged z\* on 99.9% (r110 L4 s0) and 99.75% (r90 L4 s1) of pairs; the disagreements are near-ties (beta gap <= 0.0024 / 0.014), and self-consistency agrees within 1.8e-4 wherever z\* agrees.<br>**Reused validation grids:** the grids behind early stopping also feed the gate KL and the pool fit (with different masks), as PREREG states. This is not leakage. |
| 6. Re-derived verdicts and numbers vs tables.md, summary.json and RESULTS-R3 | **PASS** (0 verdict mismatches; 8 prose-number slips, all under check 8) | **Method** (fork B, 12:03-12:10Z: /tmp/eca/audit-r3/forkB/rederive.py, compare.py, REPORT-forkB.md): exact log-sigmoid KL in float64 from the logit columns vs pstar; own paired t-intervals with t = 4.303 (as the builder) and with the exact 4.302653; own average-tie Spearman and AUC; nothing imported from analyze_r3.py. The gate, R3-D0, R3-D2a, R3-D2b and R3-D3 tables are below.<br>**summary.json:** 2,154 numbers (P0 test KL, all curve means and CIs, D2a, D2b, D0) agree within 3.9e-7.<br>**Re-run:** analyze_r3.py run with ECA_OUT_R3=forkB/rerun gives tables.md, summary.json and both PNGs byte-identical to the shipped ones.<br>**RESULTS-R3:** Sec. 10 equals tables.md byte for byte (1,068 decimal tokens), and about 210 numbers in Secs 1-9 were checked.<br>**R3-D3:** recomputed from d1-pairs.csv, it equals result.json R3D3 within 1.6e-6 overall and 1.3e-5 entropy-matched. Each of the 10 bins holds 200 pairs, so quantile edges and rank deciles agree.<br>**kl_epi:** equals KL(exact_p \|\| q) recomputed from the rounded columns within 2.4e-7 on every row of all 18 runs, and mean kl_epi matches result.json within 1.2e-11. Confidence computed from the q column reproduces result.json.<br>**My cross-check:** my own Spearman(beta, ln 2 - H*) from d1-pairs is 0.9256 for r110 L4 (result.json mean 0.9254; the difference comes from CSV ties). |
| 7. GPU | **PASS** | **Script:** my own parser /tmp/eca/audit-r3/me/gpu_sum.py over logical-soup/GPU-LOG.md (mtime 11:58:09Z), run at 12:05:57Z. **Round 3:** vm=soup-gpu-eca3* has 10 CREATE and 10 DELETE lines, none left open, plus 1 CREATE-FAILED (soup-gpu-eca3-a-s0-1015100517-10835, no instance). The 10 DELETE cost_usd values sum to 0.63 + 1.24 + 1.40 + 1.35 + 1.45 + 1.51 + 1.52 + 1.53 + 1.50 + 1.53 = $13.66 (rate ceiling x age; ages 0.72-1.76 h). **Night:** all vm=soup-gpu-eca* DELETE/COST-AUDIT lines come to 33 lines, 31 VMs created and deleted, $33.56. Both totals equal RESULTS-R3 ($13.66, $33.56) and eca/GPU-LOG.md ("Total (ceiling): $33.56 over 31 VMs"). **Live VMs:** `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca' --format=json` at 12:06:02Z returned rc 0, `[]`; with `--filter='name~soup-gpu'` it returned rc 0, empty. The first query, repeated at 12:18:12Z, again returned rc 0, `[]`. RESULTS-R3 Sec. 9b's "11 on-demand L4 VMs (1 smoke + 9 main)" is a count slip (L7). |
| 8. Overstatements in RESULTS-R3 prose | 0 high, 3 moderate, 13 low | Findings M1-M3 and L1-L13 below. The secondary readouts are labelled: Sec. 1 says "Secondary (not a rule)", and Sec. 5 says "secondary readout, computed 11:58Z after the results were read; not a decision rule". |

## Re-derived numbers (own code from the raw files; paired 95% t-intervals, 2 df)

#### Depth gate (seed-0 validation one-pass KL, result.json P0.val, 2,000 instances per family)

| rule | L | F | B1 | B2 | S1 | S1 minus 0.015 | families >= bar | gate | S1 seeds 0 / 1 / 2 | stop (s0 / s1 / s2) |
|---|---|---|---|---|---|---|---|---|---|---|
| 110 | 2 | 0.1312 | 0.3649 | 0.3718 | 0.0174 | +0.0025 | 3 | fail | 0.0174 / 0.0165 / 0.0226 | patience / cap / cap |
| 110 | 3 | 0.1309 | 0.3767 | 0.3148 | 0.0072 | -0.0078 | 3 | pass | 0.0072 / 0.0069 / 0.0079 | cap / cap / patience |
| 110 | 4 | 0.1302 | 0.3526 | 0.2501 | 0.0064 | -0.0086 | 3 | **pass, selected** | 0.0064 / 0.0051 / 0.0058 | cap / cap / cap |
| 90 | 2 | 0.1509 | 0.2231 | 0.1596 | 0.0512 | +0.0362 | 3 | fail | 0.0512 / 0.0246 / 0.0625 | patience / cap / patience |
| 90 | 3 | 0.0644 | 0.0619 | 0.0520 | 0.0248 | +0.0098 | 3 | fail | 0.0248 / 0.0118 / 0.0110 | patience x 3 |
| 90 | 4 | 0.0594 | 0.0572 | 0.0446 | 0.0153 | +0.0003 | 2 | fail (none selected) | 0.0153 / 0.0070 / 0.0100 | patience (43.2 min) / cap / cap |

Runs trained for 43.2-70.7 min, and 10 of 18 stopped on the 70-minute cap (rule 110: 7 of 9). Best validation outcome loss at L = 4 is 0.1644-0.1674 for rule 110 and 0.1521-0.1567 for rule 90. The result.json field D0_gate_val.pass uses the round-1 rule and differs from this gate for r110-L2-s0 (L12). Source: /tmp/eca/audit-r3/me/runinfo.py, 12:07Z.

#### R3-D0 (3-seed validation means at L = 4)

- Rule 110: F 0.1285 [0.1219, 0.1351], B1 0.2946 [0.1686, 0.4206], B2 0.2440 [0.2307, 0.2574], S1 0.0058 [0.0042, 0.0073]: **PASS**.
- Rule 90 (gate failed, descriptive): F 0.0693, B1 0.0754, B2 0.0370, S1 0.0108 [0.0004, 0.0212]. The 3-seed mean would pass, as RESULTS says.

#### R3-D2a (F, M = 16, sw; cut = mean of per-seed 1 - KL(arm)/KL(ref))

| rule | arm | KL | cut vs P0 | KL(P0) - KL(arm) | cut vs rand | KL(rand) - KL(arm) | (i) | (ii) |
|---|---|---|---|---|---|---|---|---|
| 110 | stiff | 0.1115 | 0.130 [0.052, 0.209] | 0.0168 [0.0060, 0.0275] | 0.153 | 0.0202 [0.0096, 0.0308] | no | yes |
| 110 | conf | 0.1094 | 0.146 [0.096, 0.197] | 0.0188 [0.0117, 0.0259] | 0.168 | 0.0222 [0.0100, 0.0344] | no | yes |
| 110 | hyb | 0.1108 | 0.136 [0.057, 0.214] | 0.0174 [0.0067, 0.0281] | 0.158 | 0.0208 [0.0090, 0.0327] | no | yes |
| 90 | stiff | 0.0317 | 0.522 | 0.0353 [-0.0002, 0.0708] | 0.135 | 0.0098 [-0.0353, 0.0549] | no | no |
| 90 | conf | 0.0341 | 0.497 | 0.0330 [0.0083, 0.0577] | 0.105 | 0.0075 [-0.0268, 0.0418] | yes | no |
| 90 | hyb | 0.0340 | 0.499 | 0.0331 [0.0086, 0.0575] | 0.109 | 0.0075 [-0.0265, 0.0415] | yes | no |

Rule 110 **FAIL**: all three arms miss the 25% bar, although all three beat random by >= 15% with CIs above 0. Rule 90 is descriptive and would fail.

#### R3-D2b (hyb, sw, M = 16)

| rule | family | KL hyb | cut vs P0 | comparison |
|---|---|---|---|---|
| 110 | F | 0.1108 | 0.136 [0.057, 0.214] | rand - hyb 0.0208 [0.0090, 0.0327] |
| 110 | B1 | 0.2626 | 0.119 [-0.090, 0.329] | stiff - hyb -0.0099 [-0.0580, 0.0383] |
| 110 | B2 | 0.2206 | 0.137 [0.028, 0.246] | stiff - hyb -0.0014 [-0.0093, 0.0065] |
| 90 | F / B1 / B2 | 0.0340 / 0.0554 / 0.0379 | 0.499 / 0.183 / 0.027 | rand - hyb F 0.0075 [-0.0265, 0.0415]; stiff - hyb B1 -0.0003, B2 0.0003 |

Rule 110 **FAIL**: no family reaches 25%, neither stiff comparison has its CI above 0, and only rand - hyb on F holds.

#### R3-D3 (Spearman with -KL(P* \|\| q), 2,000 test pairs per seed; 3-seed means)

| rule | beta | confidence ln 2 - H(q) | self-consistency | -s_hat | best - confidence | entropy-matched beta / conf / self-cons | verdict |
|---|---|---|---|---|---|---|---|
| 110 | 0.6998 [0.6477, 0.7519] | 0.6973 [0.6489, 0.7457] | 0.6965 [0.6502, 0.7427] | 0.6998 | +0.0025 | 0.1715 / 0.2010 / 0.1322 | **FAIL** |
| 90 | 0.3851 [0.2784, 0.4918] | 0.3789 [0.2747, 0.4830] | 0.3420 [0.2787, 0.4053] | 0.3874 | +0.0063 | 0.1748 / 0.1659 / -0.0083 | FAIL (descriptive) |

#### Not in RESULTS-R3 (own code /tmp/eca/audit-r3/me/infogain.py and beta_vs_conf.py, 12:09-12:11Z; fork A check_td_kl.out)

| quantity (L = 4; per seed s0 / s1 / s2, or 3-seed mean [CI]) | rule 110 | rule 90 |
|---|---|---|
| Spearman(beta, ln 2 - H(q)) on the R3-D3 pairs | 0.993 / 0.993 / 0.973 | 0.993 / 0.990 / 0.986 |
| Spearman(self-consistency, ln 2 - H(q)) | 0.894 / 0.898 / 0.894 | 0.657 / 0.686 / 0.720 |
| Spearman with the exact information gain ln 2 - H*: beta | 0.926 [0.889, 0.962] | 0.890 [0.875, 0.905] |
| same: plain confidence | 0.937 [0.924, 0.949] | 0.893 [0.874, 0.912] |
| same: self-consistency | 0.839 [0.820, 0.859] | 0.708 [0.648, 0.768] |
| beta minus confidence vs ln 2 - H*, per seed | -0.005 / -0.006 / -0.022 | -0.004 / -0.006 / +0.002 |
| self-consistency in the spec's KL form (fork A CPU replay of model.pt), Spearman with -KL, overall / entropy-matched | 0.642 / 0.109 (squared form 0.696 / 0.131) | 0.300 / -0.014 |
| R3-D3 gap under the spec's pass rule (within matched-entropy strata, either self-consistency form) | -0.030 | +0.010 |

#### Secondary readouts (rule 110, L = 4, M = 16; cut = mean of per-seed cuts; * = paired CI of KL(P0) - KL(arm) above 0)

| family | P0 KL | stiff sw | conf sw | hyb sw | voi sw | voi mix | voiu sw | rand sw | rand mix |
|---|---|---|---|---|---|---|---|---|---|
| F | 0.1282 | +0.130* | +0.146* | +0.136* | +0.310* | +0.258* | +0.000 | -0.027 | +0.091 |
| B1 | 0.3013 | +0.155 | +0.145 | +0.119 | +0.229 | +0.053 | -0.000 | +0.243* | +0.156 |
| B2 | 0.2559 | +0.143* | +0.149* | +0.137* | +0.197 | +0.159 | -0.000 | +0.226* | +0.225* |

- **Match:** all 27 cells and stars match RESULTS-R3 Sec. 5 at its rounding.
- **voiu:** voiu\|sw\|16 is fitted at lam = 1e5 (the grid maximum), eta = 0, in all three rule-110 seeds, which puts weight 0.99984 on P0. That is why it is exactly neutral.
- **hyb vs conf:** the 3-seed-mean KL difference at M = 16 is sw F 0.0014, B1 0.0070, B2 0.0029, and dd B1 0.0087. Single seeds differ by up to 0.021 (sw) and 0.025 (dd).
- **Hybrid branch share** (result.json):
  - Rule 110 L4: 0 everywhere except seed 1 test B2, 7.8e-6 (1 of 128,000 chain steps).
  - Rule 90 L4: seed 0 F 3.0e-4 (38 steps), seed 2 F 1.1e-4 (14 steps).
  - Rule 90 L2 seed 2 (secondary depth): F 0.393.
- **Measured passes per query at M = 16:** 272 for stiff, conf, rand and hyb (hyb up to 272.31), and 3,344 for voi and voiu.

## Findings ranked by severity

No high-severity finding. No verdict, rule number or pre-registered bar is misreported, nothing leaks, and the data and exact posteriors are correct.

**M1 (moderate). The R3-D3 comparison of the stiffness head with plain confidence is close to degenerate. The prose explains the null with an untested mechanism and reports only the favourable half of the information-gain comparison.**
- **Near-identical rankings:** on the R3-D3 pairs, beta_int ranks the pairs almost exactly like ln 2 - H(q). Spearman is 0.993 / 0.993 / 0.973 for the rule-110 L4 seeds and 0.986-0.993 for rule 90. So part (a) of R3-D3 is nearly the same ranking as part (b), and the +0.002 gap is close to built in. Only self-consistency is a genuinely different score (0.89 with confidence), and it does worse.
- **Untested explanation:** RESULTS Sec. 6 (lines 131-133) says "Where the network is confident it is mostly right, so all confidence-like scores rank its errors about equally". That is a mechanism stated as fact, and the audit found no test of it. The section never says that the head is a near-monotone transform of the network's confidence.
- **Half of the comparison:** Sec. 6 ends (line 134) "The stiffness head (Mechanism A) still tracks the exact information gain (D1 Spearman 0.925, AUC 0.846 for rule 110), as in rounds 1-2". The spec's R3-D3 asks for all three scores against ln 2 - H*. At the selected depth, plain confidence tracks it better than the head: 0.937 [0.924, 0.949] vs 0.926 [0.889, 0.962], higher in all three seeds; self-consistency is 0.839. result.json already logs this comparison (D1.spearman_selfentropy_vs_info, 3-seed mean 0.9365).

**M2 (moderate). Causal attribution without a control.**
- **The claims:** Sec. 8 (line 144) says "Removing memorisation did what gemini predicted for rule 110: a 4-layer network now learns the local rule (one-step KL 0.006) but cannot compose ten noisy steps in one pass". The TL;DR (lines 12-13) says "The overlap problem is gone ... the one-pass network now has a real gap".
- **Why they don't follow:** round 3 changed the row-0 overlap, W (8 to 12), T, the query rows (F is now rows 6-9), the training cap and the depth search at the same time. No arm keeps W = 12, T = 10 with overlapping splits. So this design cannot attribute the measured gap (rule 110 F 0.13, B1 0.29, B2 0.24 nats) to the removed overlap.
- **Wording:** "cannot compose" interprets a KL gap. "Ten noisy steps" is wrong for F, whose queries are 6-9 steps from row 0 (T = 10 rows allow at most 9 steps).

**M3 (moderate; secondary readout). The Gram-dedup sentence blames the VoI chains for the dd pool's own offset.**
- **The claim:** Sec. 5 (line 118) says "The Gram-dedup pool (dd) makes VoI chains worse than one pass on bridges (KL 32% higher on B1, 30% on B2)". The numbers reproduce, as the mean of per-seed ratios against the test one pass.
- **The offset:** the dd pool is already worse than one pass at M = 0, before any chain step: B1 0.3595 vs 0.3013 (+19%) and B2 0.4060 vs 0.2559 (+59%). It is anchored on the network's no-evidence prior with a fitted scale: ecarun.py 288-347 gives logit mu = logit p0 + u x feature, with p0 the network's prior at the query (line 749; the p0_logit column).
- **The chains themselves:** relative to the pool's own M = 0, VoI chains lower the dd KL on B2 (-0.074 [-0.146, -0.003]), and the B1 excess over one pass is not significant (+0.086 [-0.084, +0.257]). The excess comes from the pool, not from the chains.

**L1 (low). "The hybrid never branched".**
- **Where:** Sec. 1 (lines 20-21) and Sec. 5 (lines 115-117): "never branched (branch share 0.000; a stiff frontier cell with H < 0.20 nats always existed)", and "on every test query some unrevealed cell had predictive entropy below 0.20 nats at every step".
- **Off by one step for rule 110:** seed 1 branched once on test B2 (share 7.8e-6). The rounded 0.000 is right.
- **Rule 90:** at L4 it branched 38 (s0) and 14 (s2) times on F. At L2, seed 2 branched on 39% of F steps, so Sec. 8's "collapses to confidence-first here" holds for the L4 runs only.
- **"Frontier":** the candidate set is every unrevealed cell in the grid. In fork A's replay, 3-57% of the picks were in row 1, on average 1.5-5.8 rows from the query, so these are the network's most confident cells anywhere, not a frontier near the query.

**L2 (low).** Sec. 5 (line 116): "their KLs differ only through sampling (by at most 0.007 at M = 16)". This holds for 3-seed means under sw (0.0070). Under dd the gap is 0.0087, and single seeds differ by up to 0.021 (sw) and 0.025 (dd). The "only through sampling" part is supported: the replayed picks are identical in 100% of steps (99.8% on rule-90 B2).

**L3 (low). Rounding slips.**
- Sec. 5 (line 113): "B1 0.30 -> 0.42" should be 0.30 -> 0.43 (0.4260).
- Sec. 6 (line 134): "AUC 0.846" should be 0.845 (0.84546; tables.md shows 0.8455, so this is a double rounding).

**L4 (low).** Sec. 5 (line 121), rule 90: "whose CIs include 0 because seed 0 differs". The outlier is seed 2: its one-pass F KL is 0.092 against 0.056 and 0.053, and its one-pass-minus-VoI gain is 0.079 against 0.036 and 0.036. The "because" is also an untested explanation.

**L5 (low).** Sec. 1 (line 28): "Search with the anchored pool improves on one pass in this zero-overlap regime, but by less than the pre-registered bars." VoI under sw cuts 31% on F, with its CI above 0, which is above the 25% bar. The sentence holds only for the arms that the rules name (stiff, conf, hyb).

**L6 (low).** Sec. 6 (lines 132-133): "within matched entropy bins the correlations drop to 0.13-0.20". That is rule 110's range; rule 90's self-consistency is -0.008.

**L7 (low).** Sec. 9b (line 165): "11 on-demand L4 VMs (1 smoke + 9 main)". The log shows 10 VMs. The 11th GPU-LOG entry is a failed create (stockout) with no instance and $0.

**L8 (low).** Sec. 2 (line 35): "a 1-layer network (round 2) could not learn the rule at all". The spec and RESULTS-R2 say one layer broke one-step learning: S1 KL 0.037 / 0.077 against a 0.025 bar, vs 0.008 at 4 layers. That is degraded, not absent.

**L9 (low). Sec. 9 "Deviations" lists pre-registered choices but omits PREREG's departures from the spec.** It lists choices that were pre-registered (all 18 runs launched together; chains on F/B1/B2 with 500/500 instances). It does not list where PREREG departs from the spec (Rank 2):
- (a) 50,000 training grids, where the spec asks for 100k-150k "so the 1-step rule ... converges cleanly"; 10 of 18 runs then stopped on the time cap.
- (b) W = 12, where the spec's primary is W = 14 (W in {12, 14, 16} is allowed).
- (c) R3-D3 is decided on the overall Spearman, where the spec decides "within matched-entropy strata". The verdict is unchanged: -0.030 (r110) and +0.010 (r90) within strata.
- (d) Self-consistency is a squared probability difference, where the spec uses a KL drift. The verdict is unchanged; the KL form scores lower (0.642 vs 0.696).
- (e) Depth selection uses seed 0, where the spec's R3-D0 is a 3-seed mean. Under the 3-seed form, rule 90 would pass the gate at L = 4. Its R3-D2a would still fail on part (ii) for all three arms, and its R3-D3 on the 0.05 bar; RESULTS notes the near miss.
- (f) The hybrid samples the chosen cell from q, where the spec says "clamp". With the minimum entropy at most 0.114 nats, the two rarely differ.

**L10 (low).** PREREG's header "written 09:46Z, before any round-3 launch command" and RESULTS' "written 09:46:31Z; its header line was corrected at 09:47:03Z" cannot be verified: the inode was born at 09:47:03.4826Z and no earlier copy exists. This is not a timing failure, since the final version is 82 ms older than the smoke source copy and 18 min older than the main launches.

**L11 (low). Stale documentation.**
- ecadata3.py's docstring says "Round 3 defaults: T = 10, W in {14, 16}" and "train 80% / cal 5% / val 7.5% / test 7.5%".
- ecarun.py's comment says "--data-module ecadata3 switches to the W in {14, 16}".
- The code (W_DEF = 12; BUCKETS 0.75 / 0.09375 / 0.078125 / 0.078125) matches PREREG.

**L12 (low).** result.json D0_gate_val.pass still applies the round-1 rule: all three families >= 0.05 and S1 <= 0.02. It is true for r110-L2-s0 (S1 0.0174), which the PREREG gate fails. analyze_r3.py does not read this field, so no verdict is affected, but a reader of result.json could be misled.

**L13 (low). Code and wording notes outside the listed round-3 features** (fork A).
- **Batch size:** the Infer default batch size now shrinks with sequence length (1,310 at n = 120). This affects memory only, and training and early stopping do not use it.
- **Reveal count:** PREREG's "k ~ U{1..119}" for the R3-D3 pairs describes the draw. The target is removed afterwards, so the realised reveal count runs from 0 to 118, and one rule-90 pair has an empty S.

## What I could not check

- The PREREG header time "09:46Z" (RESULTS: "09:46:31Z") from before the 09:47:03Z header correction: there is no earlier copy and no git, and the inode birth time is 09:47:03.4826Z (L10).
- Training throughput and the exact-inference timing claims (0.020 s per grid at W = 12, 0.065 s at W = 14, the sweep at 1.07 s at W = 16): not re-measured, because the box load was about 26 on 16 cores.
- True GPU spend: the dollar figures are the ledger's rate ceiling x age (the Cloud Billing API is disabled, per GPU-LOG).
- VM-side execution: I checked that the frozen source and data equal the tree and a fresh regeneration, and that the raw outputs are internally consistent and reproduce. I did not check the execution on the VMs themselves; their logs went with the deleted VMs.
- The round-1/2 comparisons quoted in RESULTS-R3 (random order best on bridges "as in rounds 1-2"; round-2 key cosines -0.10 / 0.07 in RESULTS-R2.md lines 21 and 75-76) were checked against RESULTS.md and RESULTS-R2.md only, not re-derived.
- Partial replays: fork A's CPU replays of the hybrid and z\* choices covered two runs (r110-L4-s0, r90-L4-s1) with 100 test instances per family and K = 1, not the full 500 x 16 chains of every run. The exact-posterior comparison against pstar / exact_p covered the first 80 chain instances per family and the first 120 pairs in 3 runs per rule. The full columns were compared only with the shipped instance posteriors and across runs.
- Whether the 50,000-grid choice (L9a) contributed to rule 90's seed-0 gate miss cannot be tested with these runs.

## Commands run (all read-only outside /tmp/eca/audit-r3/; times UTC)

- 12:00:14 `date`; `cat /tmp/eca/audit-r3-prompt.md`; `ls -la --time-style=full-iso` of eca/, results/r3 and results/raw/r3. `cat` of PREREG-R3.md, RESULTS-R3.md, analyze_r3.py, AUDIT-R2.md (for the format), ecadata3.py and the spec's Rank 2 section (overnight-followups-round-3.md lines 88-150).
- 12:00 `md5sum` and `cmp` of PREREG-R3.md vs /tmp/eca/PREREG-R3-prelaunch.md; `stat -c '%n %y %z %w'` on both files and on every /tmp/eca/src-eca3-* dir; `date -u -d @<epoch>` for the src dir suffixes; `grep LAUNCH` over /tmp/eca/logs/launch-eca3-*.log.
- 12:01:12 Snapshot: `cp -p` of the round-3 files to /tmp/eca/audit-r3/snap-1203/ and `sha256sum` of 101 files to snap-1203.sha256.
- 12:03:40 `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/smallnet/pylib python3 ecadata3.py check` on the snapshot copy (log /tmp/eca/audit-r3/me/ecadata3-check.log).
- 12:04:08 `python3 /tmp/eca/audit-r3/me/data_check.py` with a fresh ECA_DATA (data_check.log, data_check.json).
- 12:05:28 `python3 /tmp/eca/audit-r3/me/exact_check.py` (exact_check.log, exact_check.json).
- 12:05:57 `python3 /tmp/eca/audit-r3/me/gpu_sum.py`; `cat eca/GPU-LOG.md`.
- 12:06:02 `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca' --format=json`, then the same with `--filter='name~soup-gpu' --format='value(name,status,zone)'`.
- 12:07-12:15 Scripts: `python3 /tmp/eca/audit-r3/me/runinfo.py`, `infogain.py` and `beta_vs_conf.py`. Lookups: `find -newermt` around the PREREG time; `find results/raw/r3 -printf '%T+'` for the first result time; `grep` of RESULTS-R2.md, RESULTS.md and the spec for the quoted round-1/2 numbers; a per-run read of result.json D0_gate_val.
- 12:14:33 `md5sum` of /tmp/eca/src-eca3-*/eca/*.py and the tree; `sha256sum -c --quiet snap-1203.sha256` (101 OK); `md5sum PREREG-R3.md RESULTS-R3.md`; `ls PREREG-R3-AMENDMENTS.md` (absent).
- 12:18:12 The same re-check before finishing: `sha256sum -c --quiet snap-1203.sha256` (101 OK), `md5sum PREREG-R3.md RESULTS-R3.md` (da122736, ed5c1068), `ls PREREG-R3-AMENDMENTS.md` (absent), and `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca' --format=json` (rc 0, `[]`).
- **Fork A** (checks 4-5, /tmp/eca/audit-r3/forkA/):
  - diffs and section cuts: ecarun-r2-vs-r3.diff, ecanet-r2-vs-r3.diff and ecadata-r2-vs-r3.diff (both empty), train-section-r2/r3.txt, voi-old/new.norm;
  - md5s: frozen-md5.txt;
  - check_args.py / .out (args, stop reasons, model.pt loads);
  - check_zstar_hyb.py / .out / .json (CPU replay of the hybrid and z\*);
  - check_td_kl.py / .out / .json (spec KL-form self-consistency).
- **Fork B** (check 6, /tmp/eca/audit-r3/forkB/):
  - rederive.py and rederive-out.txt;
  - compare.py and compare.log;
  - rerun/ (the analyze_r3.py re-run);
  - results-sec10.md, results-sec1-9.md and sec10-vs-tables.diff (empty);
  - REPORT-forkB.md.
