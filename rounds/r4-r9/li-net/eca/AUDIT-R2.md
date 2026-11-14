# E-CA round 2 independent audit: pre-registration timing, code, leakage, rule re-derivation, sw_mix, GPU spend, prose

Auditor: Opus 5.5 (1M context; model ID claude-opus-5-5[1m]), independent pass, 2026-10-15T05:56Z. Started 05:26Z from a snapshot of the round-2 files (/tmp/eca/audit-r2/snap-0527/, sha256 list snap-0527.sha256). At the last check (05:55:09Z), every input was unchanged except RESULTS-R2.md and the header line of PREREG-R2.md (pre-run version md5 8e33ae51): analyze_r2.py a121bb56, results/r2/summary.json 023aa756, results/r2/tables.md 7b4662b5, ecarun.py 99047322, ecanet.py 4bef0c4e, eca/GPU-LOG.md 27a2c4ab, 45 raw files (/tmp/eca/audit-r2/raw-checksums.txt). The builder edited RESULTS-R2.md during the audit (05:37:12Z, then from 05:49Z onward, applying this audit's draft findings) and edited the header line of PREREG-R2.md at 05:49:11Z. Checks 1-7 and the findings below refer to PREREG-R2.md md5 8e33ae51 (saved 04:30:36Z) and RESULTS-R2.md md5 d59e2588 (05:37:12Z); copies are in /tmp/eca/audit-r2/ (snap-0527/PREREG-R2.md, RESULTS-R2-d59e2588.md). The section "Edits made during the audit" checks the final versions (RESULTS-R2.md md5 06060788, mtime 05:49:22Z).

**Summary (5 lines)**
1. All four round-2 verdicts re-derive from the raw CSV/JSON with my own code (exact log-sigmoid KL, own t-intervals, nothing imported from analyze_r2.py): R2-D2a FAIL for rules 110 and 90, R2-D2b FAIL for both (F, B1 and B2 qualify in both), R2-D3a PASS, R2-D3b FAIL. My rendition of results/r2/tables.md matches it line for line with t = 4.303 (the value analyze_r2.py uses). 3,492 summary.json numbers agree to within 4.4e-7. analyze_r2.py re-run into scratch reproduces tables.md and summary.json byte for byte, and the RESULTS-R2 full-tables section (Sec. 9; Sec. 10 in the final version) is tables.md verbatim.
2. The code is as described. The round-1 to round-2 diff contains only the flagged features, and the training loop is unchanged: keyW2 is excluded from the optimiser, and every shared parameter initialises identically. The 5 frozen copies equal the tree, and all 9 result.json and model.pt args match PREREG-R2. sw_mix is fitted on validation outcomes only, the v2 key on cal-bag outcomes, and voiu ranks cells by the network's own q. At M = 0, sw_mix equals P0 to within 2.8e-6 in logit, and all 198 (lam, eta) fits are on the grid.
3. Check 1 FAILS on the file times. The pre-run version of PREREG-R2.md was saved at 04:30:36Z, 23 s after the first LAUNCH (04:30:13Z) and 15 s after the last (04:30:21Z), and there is no earlier copy. It still predates every VM RUN line (04:31Z onward) and the first round-2 result (04:47Z), and its rules are the ones applied, so no verdict is at risk. The statement "written before the launches" is not supported. The builder edited the header line again at 05:49:11Z; the body is unchanged.
4. GPU spend is $2.88 for round 2 (5 DELETE lines) and $19.90 for the night (23 DELETE / COST-AUDIT lines), equal to RESULTS-R2.md and eca/GPU-LOG.md. gcloud at 05:36:30Z and again at 05:55:10Z listed no soup-gpu-eca* instance.
5. Prose overstatements: (a) R2-D3a's PASS comes from restricting the key fingerprint to the new cells, which are disjoint for forward and backward chains by construction, not from dropping h_i: the same v2 key head computed over all of S gives 0.73 (B1) and 0.81 (B2), above the 0.35 bar, and disjoint same-direction cells also score about 0.08; (b) on the 4-layer R2b runs every "sw" number comes from an anchored uniform mixture (eta = 0 in 89 of 90 fits), which RESULTS-R2 does not say; (c) five smaller wording items (L1-L5). The builder revised RESULTS-R2.md from this audit's draft during the audit; the final text reflects H1, M2, L1-L3 and L5, and the remaining slips are listed in the last section.

## Verdict table

| check | result | evidence |
|---|---|---|
| 1a. PREREG-R2 written before the round-2 launches | **FAIL** (late by 15-23 s; it still preceded every VM job and every result) | `stat` at 05:26Z: PREREG-R2.md mtime 04:30:36.4379Z (the pre-run version; the file was saved again at 05:49:11Z, see M1). LAUNCH lines (/tmp/eca/logs/launch-eca-r2*.log): r2a-r110 04:30:13Z, r2a-r90 04:30:15Z, r2b-s0/s1/s2 04:30:17Z / 04:30:19Z / 04:30:21Z. The frozen source dirs /tmp/eca/src-eca-r2*-* were made between 04:30:13.7Z and 04:30:21.6Z. logical-soup/GPU-LOG.md (minute resolution): CREATE lines at 04:30Z, 04:32Z, 04:33Z, 04:36Z and 04:36Z; RUN lines at 04:31Z, 04:33Z, 04:34Z, 04:37Z and 04:37Z. The first round-2 result file is r2a/r90-T1L1-s0/train-history.csv at 04:47:01Z. /tmp/eca/msg-r2.txt was saved 2 ms after PREREG-R2.md (04:30:36.4398Z) and says "Round 2 launched 04:30Z per PREREG-R2.md". PREREG-R2's pre-run header cites "the round-2 launches at 04:30:13Z", which could only be known after that launch. There is no earlier copy under /tmp/eca or eca/ and no git; a session-history search (`cass search`, 45 s) and a transcript grep (60 s) both timed out. |
| 1b. Rules applied = PREREG-R2 rules | PASS | analyze_r2.py implements R2-D2a/b and R2-D3a/b as written (bars 0.025 / 0.05 and 2 of 3; qualifying families at test P0 KL >= 0.05; cut >= 25% with the KL-difference CI > 0, and KL(rand) - KL(sel) CI > 0 at M = 8 or 16, under mix AND sw, Stiff or Conf; key cos <= 0.35 on B1 and B2, redundant >= 0.85, ratio m=5 < 0.20; voiu cuts >= 30% vs voi and >= 35% vs P0 with CIs > 0, on B1 and B2, mix and sw, M = 8). RESULTS-R2 quotes the same bars. The 5 frozen jobs.txt files equal the PREREG-R2 run list. |
| 2. Code | PASS | The ecarun.py diff against round 1 (/tmp/eca/ecarun-r1-float64.py, 39 hunks, /tmp/eca/audit-r2/ecarun-r1-vs-r2.diff) touches only the flagged features: header note; pools + "sw", SW_ETAS (0, 0.5, 1, 2), SW_LAMS {0} U logspace(-3, 5, 33); Infer rev_base / qidx and attention vectors; run_chains rev_base, want_av, CFG tau and the voiu branch; sw_pool, fit_sw, pooled("sw"); --key-mode / --tau / --selectors / --save-model; net.key_mode; key-head fit on keyW2 for v2 (same cal-bag loop); the model.pt save; SEL loops; sw fit on validation and sw test pooling; the voiu pass count; D3a / D3b attention cosines. --layers already existed in round 1 (line 281). The training loop (round-2 lines 371-478) has no hunk. Trainable parameters exclude stiff* and keyW* (line 398), which covers keyW2. With torch.manual_seed(0), the round-1 and round-2 Net give identical values for all 26 (1 layer) and 62 (4 layers) shared tensors; keyW2 is the only extra one. The ecanet.py diff (2 hunks) adds keyW2 / key_mode, source_mask, attn_vec and the v2 branch of key(); the v1 branch is the round-1 code. md5: all 5 frozen ecarun.py equal the tree (99047322), as do ecanet.py (4bef0c4e), ecadata.py (6ef1d729, also equal to the round-1 copy) and netdef.py (ce9db98f, also equal to the round-1 copy). In result.json (9 runs), r2a has layers 1, tau 5.0, key v1 and selectors stiff,conf,rand; r2b has layers 4, tau 1.0, key v2 and selectors stiff,voi,voiu,conf,rand; all have K 16, Mmax 16, T1 and seeds 0/1/2. Compared with the round-1 primary run, the only other arg changes are max_minutes (140 to 60 / 50; never binding, since every run stopped on patience after 11-29 min) and out. All 9 model.pt files load on CPU with the stated layer count, args equal to result.json, and the same perm. |
| 3. No exact posterior or rule knowledge in training or selection | PASS | fit_sw(qv, sv, P0v, yv): yv = chain["val"][fam]["y"], the grid value at the query (fam_arrays returns xg[i, q]). q, s and P0v (chain q at M = 0) all come from the network, and the grid search runs on validation outcome log loss (ecarun.py 284-300, 684-691). Test pooling uses the validation fit with P0 = the test chain's q at M = 0 (706-707). keyW2 and log_u are trained by the BCE of the Gram-crowded pool against xb[tgt] (cal grids) on shared-core and disjoint bags of random reveals (522-571). voiu takes the network's q(1 - q) + 1e-4 chain-generator noise, then the same network VoI as voi (209-235). The diff never references pstar or the posterior. |
| 4. Re-derived verdicts and numbers vs tables.md, summary.json and RESULTS-R2 | PASS (0 mismatches) | rederive.py + render.py give tables.md identically with t = 4.303. With the exact t (4.30265), 21 lines differ in a 4th-decimal CI bound and all 40 verdict tokens are unchanged. summary.json: 3,492 numbers compared, 10 differ by > 1e-6 relative (max 4.4e-7 absolute, all lw-pool curve CI bounds: clipped vs exact KL). The round-1 voi column (main2 r110-T1, own code) gives B1 0.1058 [0.0925, 0.1192] and B2 0.0745 [0.0447, 0.1043], the same as the table. The D2b qualifying set is unchanged if the 3,000-instance test P0 in result.json is used instead (r90 B2 0.0532). D1 recomputed from d1-pairs.csv with own rank code: AUC equals result.json to within 1.5e-5, Spearman to within 1.8e-3 (ties from CSV rounding). Validation P0 KL replayed on CPU from model.pt (r2a r110-s0 and r90-s2, F/B1/B2/R/S1) equals result.json to within 6.1e-5. Prose numbers: items L4 below. |
| 5. sw_mix sanity | PASS | max abs(sw\|0 - P0_logit) is 2.0e-6 (r2a) and 2.8e-6 (r2b) in logit, at most 1.3e-7 in probability. mix\|0 differs from P0 by the same amounts, which is the spread of float32 q across the 16 identical chain copies, so sw at M = 0 equals P0 up to that noise. All 198 fitted (lam, eta) (54 + 54 + 90) are on the grid; lam = 1e5 is never chosen, and lam = 0 is chosen 16 / 11 / 20 times (r2a-r110 / r2a-r90 / r2b). |
| 6. GPU | PASS | Own parser over logical-soup/GPU-LOG.md. The 5 DELETE lines for vm=soup-gpu-eca-r2* sum to 0.37 + 0.51 + 0.60 + 0.67 + 0.73 = $2.88; there are 5 CREATE and 5 DELETE, none left open. All 23 eca DELETE / COST-AUDIT lines sum to $19.90, equal to the eca/GPU-LOG.md total and to RESULTS-R2. `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca' --format=json` at 05:36:30Z: rc 0, `[]`; with the broader filter name~soup-gpu the list is also empty. Repeated at 05:55:10Z: rc 0, `[]`. |
| 7. Overstatements in RESULTS-R2 prose | 1 high, 2 moderate, 5 low | Findings H1 (high), M1 (= check 1a) and M2 (moderate), L1-L5 (low) |

## Re-derived numbers (own code, from the raw files; 95% t-intervals with 2 df, t = 4.30265)

#### R2-D2a: validation P0 KL (result.json P0.val, 3,000 instances per family; s0 / s1 / s2 and 3-seed mean, 95% t-interval)

| rule | F | B1 | B2 | R | S1 | families >= 0.05 | S1 <= 0.025 | R2-D2a |
|---|---|---|---|---|---|---|---|---|
| r110 | 0.3466 / 0.3454 / 0.3251; mean 0.3390 | 0.4309 / 0.5382 / 0.3929; mean 0.4540 | 0.1788 / 0.3459 / 0.2534; mean 0.2594 | 0.0856 / 0.1295 / 0.0849; mean 0.1000 | 0.0497 / 0.0318 / 0.0296; mean 0.0370 | 3 of 3 | False | FAIL |
| r90 | 0.1471 / 0.1483 / 0.0184; mean 0.1046 | 0.5142 / 0.5144 / 0.0495; mean 0.3594 | 0.0523 / 0.0429 / 0.0527; mean 0.0493 | 0.0449 / 0.0583 / 0.0162; mean 0.0398 | 0.1006 / 0.1078 / 0.0211; mean 0.0765 | 2 of 3 | False | FAIL |

#### R2-D2b: per arm at M = 16 (test-estimates.csv, 1,000 instances x 3 seeds; cut = mean of per-seed 1 - KL/KL(P0); CIs are paired 95% t-intervals of KL differences)

| rule | family (test P0 KL) | arm | cut | KL(P0) - KL(arm) | part 1 (cut >= 0.25 and CI > 0) | KL(rand) - KL(arm) M=8 | M=16 | part 2 |
|---|---|---|---|---|---|---|---|---|
| r110 | F (0.3346) | stiff\|mix | 0.034 | 0.0108 [-0.0702, 0.0919] | fail | 0.0226 [-0.0144, 0.0597] | -0.0061 [-0.0941, 0.0819] | fail |
| r110 | F (0.3346) | stiff\|sw | 0.237 | 0.0789 [0.0349, 0.1230] | fail | 0.0378 [0.0226, 0.0531] | 0.0658 [0.0494, 0.0822] | pass |
| r110 | F (0.3346) | conf\|mix | -0.088 | -0.0302 [-0.2137, 0.1534] | fail | 0.0118 [-0.0079, 0.0314] | -0.0471 [-0.2365, 0.1422] | fail |
| r110 | F (0.3346) | conf\|sw | 0.198 | 0.0658 [0.0112, 0.1203] | fail | 0.0324 [0.0155, 0.0492] | 0.0526 [0.0140, 0.0913] | pass |
| r110 | B1 (0.4464) | stiff\|mix | 0.066 | 0.0384 [-0.2067, 0.2836] | fail | -0.0390 [-0.1497, 0.0717] | -0.0631 [-0.3212, 0.1950] | fail |
| r110 | B1 (0.4464) | stiff\|sw | 0.288 | 0.1324 [-0.0332, 0.2979] | fail | -0.0105 [-0.0862, 0.0653] | 0.0080 [-0.1711, 0.1871] | fail |
| r110 | B1 (0.4464) | conf\|mix | 0.007 | 0.0146 [-0.3052, 0.3343] | fail | -0.0490 [-0.2036, 0.1056] | -0.0870 [-0.4215, 0.2475] | fail |
| r110 | B1 (0.4464) | conf\|sw | 0.268 | 0.1236 [-0.0361, 0.2833] | fail | -0.0126 [-0.1018, 0.0767] | -0.0008 [-0.1736, 0.1720] | fail |
| r110 | B2 (0.2637) | stiff\|mix | -0.222 | -0.0313 [-0.3481, 0.2856] | fail | -0.0458 [-0.2299, 0.1382] | -0.0795 [-0.3939, 0.2350] | fail |
| r110 | B2 (0.2637) | stiff\|sw | 0.208 | 0.0642 [-0.0978, 0.2262] | fail | -0.0079 [-0.1032, 0.0874] | 0.0008 [-0.1358, 0.1374] | fail |
| r110 | B2 (0.2637) | conf\|mix | -0.318 | -0.0560 [-0.3582, 0.2462] | fail | -0.0633 [-0.2521, 0.1256] | -0.1042 [-0.4024, 0.1940] | fail |
| r110 | B2 (0.2637) | conf\|sw | 0.194 | 0.0586 [-0.0778, 0.1949] | fail | -0.0145 [-0.0970, 0.0680] | -0.0049 [-0.1179, 0.1082] | fail |
| r90 | F (0.1023) | stiff\|mix | -0.907 | -0.0326 [-0.0491, -0.0162] | fail | -0.0006 [-0.0141, 0.0130] | -0.0099 [-0.0319, 0.0120] | fail |
| r90 | F (0.1023) | stiff\|sw | -0.048 | -0.0022 [-0.0063, 0.0019] | fail | -0.0032 [-0.0076, 0.0013] | -0.0018 [-0.0050, 0.0014] | fail |
| r90 | F (0.1023) | conf\|mix | -1.221 | -0.0544 [-0.0642, -0.0445] | fail | -0.0050 [-0.0184, 0.0084] | -0.0317 [-0.0365, -0.0269] | fail |
| r90 | F (0.1023) | conf\|sw | -0.149 | -0.0048 [-0.0123, 0.0028] | fail | -0.0046 [-0.0054, -0.0039] | -0.0044 [-0.0109, 0.0022] | fail |
| r90 | B1 (0.3610) | stiff\|mix | -0.142 | -0.0171 [-0.0270, -0.0071] | fail | 0.0047 [-0.0385, 0.0479] | 0.0045 [-0.0182, 0.0271] | fail |
| r90 | B1 (0.3610) | stiff\|sw | 0.099 | 0.0040 [-0.0223, 0.0304] | fail | -0.0016 [-0.0084, 0.0052] | -0.0029 [-0.0077, 0.0020] | fail |
| r90 | B1 (0.3610) | conf\|mix | -0.173 | -0.0201 [-0.0347, -0.0056] | fail | 0.0036 [-0.0436, 0.0507] | 0.0014 [-0.0238, 0.0266] | fail |
| r90 | B1 (0.3610) | conf\|sw | 0.087 | 0.0033 [-0.0207, 0.0272] | fail | -0.0017 [-0.0098, 0.0063] | -0.0036 [-0.0114, 0.0042] | fail |
| r90 | B2 (0.0582) | stiff\|mix | -0.042 | -0.0030 [-0.0172, 0.0111] | fail | -0.0102 [-0.0255, 0.0052] | -0.0169 [-0.0394, 0.0056] | fail |
| r90 | B2 (0.0582) | stiff\|sw | 0.078 | 0.0042 [-0.0032, 0.0116] | fail | -0.0022 [-0.0086, 0.0042] | -0.0013 [-0.0117, 0.0092] | fail |
| r90 | B2 (0.0582) | conf\|mix | -0.060 | -0.0044 [-0.0290, 0.0202] | fail | -0.0105 [-0.0282, 0.0073] | -0.0183 [-0.0444, 0.0079] | fail |
| r90 | B2 (0.0582) | conf\|sw | 0.072 | 0.0037 [-0.0047, 0.0122] | fail | -0.0022 [-0.0102, 0.0059] | -0.0017 [-0.0143, 0.0110] | fail |

#### R2-D3a (result.json; s0 / s1 / s2, 3-seed mean)

| quantity | per seed | mean [95% CI] | bar | result |
|---|---|---|---|---|
| cos(k_fwd, k_bwd) B1 | 0.0072 / -0.3761 / 0.0818 | -0.0957 [-0.7060, 0.5145] | <= 0.35 | pass |
| cos(k_fwd, k_bwd) B2 | -0.2119 / 0.1186 / 0.2908 | 0.0658 [-0.5688, 0.7005] | <= 0.35 | pass |
| redundant-view key cos | 0.8611 / 0.9177 / 0.9444 | 0.9077 [0.8021, 1.0134] | >= 0.85 | pass |
| drift ratio m=5 (dd/lw drift, recomputed) | 0.0608 / 0.0515 / 0.0428 | 0.0517 [0.0293, 0.0741] | < 0.20 | pass |
| drift ratio m=2 | 0.1636 / 0.1558 / 0.1286 | 0.1494 [0.1036, 0.1951] | (reported) | |
| attention-vector cos fwd/bwd B1, B2 | 0.0000 / 0.0000 / 0.0000; 0.0000 / 0.0000 / 0.0000 | 0 | (by construction) | |
| attention-vector cos, redundant views | 0.6893 / 0.7409 / 0.8574 | 0.7625 [0.5485, 0.9765] | (reported) | |

#### R2-D3b at M = 8 (test-estimates.csv; voi and voiu on the same R2b models)

| family, pool | KL P0 | KL voi | KL voiu | cut vs voi | KL(voi) - KL(voiu) | cut vs P0 | KL(P0) - KL(voiu) | rule |
|---|---|---|---|---|---|---|---|---|
| B1\|mix | 0.0902 | 0.0985 | 0.0676 | 0.310 [0.146, 0.474] | 0.0309 [0.0050, 0.0568] | 0.250 [0.116, 0.384] | 0.0226 [0.0090, 0.0362] | FAIL |
| B1\|sw | 0.0902 | 0.0763 | 0.0621 | 0.187 [-0.062, 0.435] | 0.0142 [-0.0051, 0.0335] | 0.313 [0.224, 0.402] | 0.0280 [0.0230, 0.0331] | FAIL |
| B2\|mix | 0.0632 | 0.0757 | 0.0620 | 0.164 [-0.159, 0.488] | 0.0137 [-0.0140, 0.0414] | 0.006 [-0.232, 0.244] | 0.0012 [-0.0151, 0.0175] | FAIL |
| B2\|sw | 0.0632 | 0.0488 | 0.0482 | 0.004 [-0.129, 0.137] | 0.0006 [-0.0062, 0.0074] | 0.233 [0.160, 0.307] | 0.0150 [0.0025, 0.0275] | FAIL |

#### Secondary readout, R2b (4 layers) under sw_mix at M = 16: cut vs P0 and paired CI of KL(P0) - KL(arm)

| family | stiff | voi | voiu | conf | rand |
|---|---|---|---|---|---|
| F | +0.432 [-0.0004, +0.0227] | +0.433 [-0.0016, +0.0242] | -0.023 [-0.0054, +0.0040] | +0.399 [+0.0007, +0.0199] | +0.332 [-0.0019, +0.0193] |
| B1 | +0.240 [+0.0088, +0.0349] | +0.269 [+0.0041, +0.0452] | +0.319 [+0.0237, +0.0336] | +0.197 [+0.0056, +0.0302] | +0.455 [+0.0251, +0.0572] |
| B2 | +0.325 [+0.0064, +0.0350] | +0.275 [+0.0050, +0.0301] | +0.235 [+0.0030, +0.0271] | +0.305 [-0.0003, +0.0396] | +0.396 [+0.0016, +0.0496] |
| R | +0.226 [+0.0025, +0.0053] | +0.263 [+0.0019, +0.0074] | +0.130 [-0.0015, +0.0060] | +0.192 [+0.0025, +0.0041] | +0.236 [+0.0030, +0.0051] |

#### Fitted sw_mix (lam, eta) at M = 16 (result.json pool_sw; s0 / s1 / s2) and eta counts over all M

| run set | selector | (lam, eta) at M = 16 |
|---|---|---|
| r2a-r110 | stiff | (56.2, 0.5); (10, 0.5); (31.6, 0.5) |
| r2a-r110 | conf | (56.2, 0.5); (17.8, 0.5); (31.6, 0.5) |
| r2a-r110 | rand | (10, 0.5); (10, 0.5); (5.62, 0.5) |
| r2a-r90 | stiff | (56.2, 0); (17.8, 0); (17.8, 0) |
| r2a-r90 | conf | (56.2, 0); (17.8, 0); (56.2, 0.5) |
| r2a-r90 | rand | (562, 0.5); (56.2, 0); (562, 1) |
| r2b-r110 | stiff | (17.8, 0); (31.6, 0); (17.8, 0) |
| r2b-r110 | voi | (31.6, 0); (31.6, 0); (17.8, 0) |
| r2b-r110 | voiu | (17.8, 0); (31.6, 0); (31.6, 0) |
| r2b-r110 | conf | (31.6, 0); (31.6, 0); (17.8, 0) |
| r2b-r110 | rand | (10, 0); (17.8, 0); (10, 0) |

- r2a-r110: eta counts by M: M=0: eta 0 x9; M=1: eta 0 x5, eta 0.5 x3, eta 2 x1; M=2: eta 0 x5, eta 0.5 x3, eta 1 x1; M=4: eta 0 x7, eta 0.5 x2; M=8: eta 0.5 x9; M=16: eta 0.5 x9; grid fits 54, off-grid 0, lam = 1e5 chosen 0, lam = 0 chosen 16
- r2a-r90: eta counts by M: M=0: eta 0 x9; M=1: eta 0 x3, eta 0.5 x3, eta 1 x1, eta 2 x2; M=2: eta 0 x3, eta 0.5 x3, eta 1 x1, eta 2 x2; M=4: eta 0 x3, eta 0.5 x4, eta 2 x2; M=8: eta 0 x5, eta 0.5 x2, eta 1 x2; M=16: eta 0 x6, eta 0.5 x2, eta 1 x1; grid fits 54, off-grid 0, lam = 1e5 chosen 0, lam = 0 chosen 11
- r2b-r110: eta counts by M: M=0: eta 0 x15; M=1: eta 0 x14, eta 0.5 x1; M=2: eta 0 x15; M=4: eta 0 x15; M=8: eta 0 x15; M=16: eta 0 x15; grid fits 90, off-grid 0, lam = 1e5 chosen 0, lam = 0 chosen 20
- r2a-r110: max abs(sw|0 logit - P0_logit) = 2.0e-06 (prob 1.3e-07); same for mix|0: 2.0e-06
- r2a-r90: max abs(sw|0 logit - P0_logit) = 2.0e-06 (prob 6.4e-08); same for mix|0: 2.0e-06
- r2b-r110: max abs(sw|0 logit - P0_logit) = 2.8e-06 (prob 7.9e-08); same for mix|0: 2.8e-06

#### R2-D3a ablation on the R2b checkpoints (CPU replay of the D3b chains, 1,000 test instances, M = 16; scripts in /tmp/eca/audit-r2/ablation/)

| family | quantity | s0 / s1 / s2 | mean [95% CI] |
|---|---|---|---|
| B1 | result.json cos(k_fwd, k_bwd) (as run, GPU) | 0.0072 / -0.3761 / 0.0818 | -0.0957 [-0.7060, 0.5145] |
| B1 | v2 key, restricted to S \ E_0 (as-run definition, CPU replay) | 0.0018 / -0.3767 / 0.0771 | -0.0992 [-0.7034, 0.5049] |
| B1 | **same v2 key head, fingerprint over all of S** | 0.7284 / 0.6757 / 0.7969 | **0.7337 [0.5826, 0.8847]** |
| B1 | attention-vector cosine over all of S | 0.4568 / 0.4458 / 0.4273 | 0.4433 [0.4062, 0.4804] |
| B2 | result.json cos(k_fwd, k_bwd) (as run, GPU) | -0.2119 / 0.1186 / 0.2908 | 0.0658 [-0.5688, 0.7005] |
| B2 | v2 key, restricted (CPU replay) | -0.2116 / 0.1148 / 0.2917 | 0.0650 [-0.5693, 0.6992] |
| B2 | **same v2 key head, fingerprint over all of S** | 0.7803 / 0.8280 / 0.8269 | **0.8117 [0.7441, 0.8794]** |
| B2 | attention-vector cosine over all of S | 0.6046 / 0.6503 / 0.5892 | 0.6147 [0.5358, 0.6936] |

The replayed keys equal run_chains' own key output on the captured chain states exactly. The replay differs from result.json by at most 0.0054 per seed, because tie-breaks and sampled values use fresh RNGs. In 23.8% of B2 rows (every seed) the backward chain has no new cell and falls back to all of S; B1 has none. A support control on B1 (all 1,000 test instances, query in row 2 or 3; three forced chains of 8 new cells each: the adjacent hidden row on the side with two hidden rows, the next row on that side, and the single hidden row on the other side; key read at M = 8) gives these restricted-key cosines:

- same cells, re-sampled: 0.978 / 0.993 / 0.975.
- same direction, disjoint rows: 0.034 / 0.232 / -0.039 (mean 0.076).
- opposite directions, adjacent rows: -0.043 / -0.251 / 0.293 (mean 0.000).
- opposite directions, far row vs opposite row: 0.107 / -0.340 / -0.399 (mean -0.210).

Over all of S the three disjoint pairs give 0.73-0.84 and the same-cells pair 0.98-0.99. The restricted key separates disjoint supports whatever their direction.

## Findings, ranked by severity

**H1 (high for interpretation; the R2-D3a verdict arithmetic is unaffected): the R2-D3a PASS measures support disjointness, not direction, and it does not show that dropping h_i fixed the key.**
- The v2 key changed three things at once: it dropped h_i, it restricted the attention fingerprint to S \ E_0 (renormalised), and it added a LayerNorm.
- Forward and backward chains materialise disjoint cells, so their restricted fingerprints have disjoint supports by construction (PREREG-R2 says so for the attention vectors).
- With the same trained v2 head and no restriction (fingerprint over all of S, the definition used in D3a and in key training), the cosine is 0.73 (B1) and 0.81 (B2). The h_i-free, LayerNormed key without the S \ E_0 restriction would therefore fail the 0.35 bar.
- In the B1 support control (8 new cells per chain), disjoint cells in the same direction also get a restricted-key cosine of about 0.08. The key separates supports, whatever their direction.
- The D3a redundant-view cosine (0.91) is computed over all of S (no E_0 in D3a), so the two halves of R2-D3a use different fingerprint supports.
- Not supported in RESULTS-R2:
  - TL;DR: "keys separate directions"; "Without the target's own hidden state the key cosine ... drops from 0.95 ... to -0.10 / 0.07".
  - Sec. 4: "Gemini's diagnosis (the target's own hidden state swamping the attention fingerprint) was right".
  - "the informative separation is in the learned key". It is mostly in the support restriction. Gemini's review (overnight-followups-round-1.md line 22) also named the E_0 restriction, and that part is what the data support.
- The rule was applied as pre-registered. The limitation is in the test design, which cannot tell direction from support, and the prose reads more into the PASS than the test can show.

**M1 (moderate, process): pre-registration timing (check 1a).**
- PREREG-R2.md's pre-run save (04:30:36Z) came 15-23 s after the five launch commands, in the same second as the launch message.
- PREREG-R2's header ("written 04:29-04:30Z, before the round-2 launches at 04:30:13Z"), the RESULTS-R2 header line ("written 04:29-04:30Z, before the launches") and /tmp/eca/msg-r2-results.txt ("written before launch") are therefore not supported. Nothing shows whether the text changed between drafting and that save.
- Mitigating facts:
  - No VM had started a job (first RUN line 04:31Z), and the first round-2 result file reached home at 04:47Z.
  - The frozen job lists match the PREREG run list.
  - The applied rules equal the PREREG rules.
- Accurate wording: "written 04:29-04:30:36Z; last saved 15-23 s after the launch commands, before any job ran or any result existed."
- Edit at 05:49:11Z, after the results and after this audit's draft: the builder changed PREREG-R2.md's header line to "rules drafted 04:29-04:30Z immediately before the launch command at 04:30:13Z; this header line was edited at 04:30:36Z, after the launches, see AUDIT-R2 M1". The body is byte-identical to the 04:30:36Z save (diff against the 05:26Z snapshot: line 1 only). The claim that the 04:30:36Z save changed only the header cannot be checked, because there is no earlier copy. The file now carries two post-launch saves. A pre-registration erratum is usually kept outside the pre-registered file, so the file stays as it was when the runs started.

**M2 (moderate): on the 4-layer R2b models, sw_mix is an anchored uniform mixture, not a stiffness-weighted pool.**
- The fitted eta is 0 in 89 of 90 R2b fits, including all 15 at M = 8 (the R2-D3b sw rows) and all 15 at M = 16 (the Sec. 5 secondary readout). lam is 10-31.6 at M = 16, i.e. 38-66% of the weight sits on P0 with 16 chains.
- RESULTS-R2 does not say this. Sec. 5 is titled "The anchored stiffness pool". It ends: "it does show that the anchor, not the stiffness weights, is what the round-1 pool was missing (the fitted eta is 0.5-1 where stiffness helps)".
- Across the 33 fits at M = 16, eta is 0.5 in 9 of 9 rule-110 1-layer fits, 0 in 6 of 9 rule-90 fits and 0 in 15 of 15 R2b fits; eta = 1 occurs once (rule 90, rand, seed 2).
- The round-1 lw pool differs from sw_mix in more than the anchor: it is log-linear rather than linear, it is anchored at the network prior q(x_i | nothing) rather than P0, and it uses one fitted u rather than (lam, eta) per selector and M. No readout isolates "the anchor".
- The secondary numbers themselves re-derive exactly, and they are labelled post hoc.

**L1 (low).** Sec. 4 says "With the new keys the forward + backward pool beats both single chains on B1 (0.089 vs 0.118 / 0.127) and B2 (0.090 vs 0.120 / 0.137)". This holds: single minus both is B1 fwd 0.0286 [0.0091, 0.0481], bwd 0.0375 [0.0146, 0.0604]; B2 fwd 0.0306 [0.0164, 0.0448], bwd 0.0470 [0.0352, 0.0588]. But round 1, with the old keys (cos 0.95), showed the same ordering (main2 r110-T1 dd: B1 0.0933 vs 0.1250 / 0.1345; B2 0.1028 vs 0.1341 / 0.1605). It is not an effect of the new keys.

**L2 (low).** The TL;DR lists "forward queries -33% to -43%" among the examples after "14 of 20 ... with paired CIs above 0; e.g.". On F only conf's CI is above 0 ([0.0007, 0.0199]); the CIs for stiff [-0.0004, 0.0227], voi [-0.0016, 0.0242] and rand [-0.0019, 0.0193] include 0. Sec. 5 flags stiff's CI but not random's.

**L3 (low).** Sec. 3 says "'routing through stiff arcs' has no stiff arcs to route through". This is a mechanism stated as fact. The next sentence reports that stiffness still ranks pairs by information gain (AUC 0.88 / 0.86). On rule-110 F under sw_mix, stiffness-first also beats random order: KL(rand) - KL(stiff) is 0.0378 [0.0226, 0.0531] at M = 8 and 0.0658 [0.0494, 0.0822] at M = 16.

**L4 (low): number slips.**
- TL;DR and Sec. 3 give rule-90 B2 validation KL as "0.05". It is 0.0493, under the 0.05 bar; the "2 of 3" count is right.
- "4-layer round 1: 0.006-0.10": round-1 3-seed validation P0 KL on F/B1/B2 spans 0.0060-0.0804 (0.0020-0.1347 per seed).
- "S1 ... 0.005-0.008 at 4 layers": the 3-seed means are 0.0077 (r110-T1), 0.0084 (r90-T1) and 0.0077 (R2b); per seed 0.0048-0.0106.
- "not at all for rule 90 (eta = 0 in 6 of 9)": the other 3 of 9 fits use eta 0.5-1.

**L5 (low).** Sec. 7 says keyW2 "changes initialisation draws relative to round 1". The initial values of all shared parameters are identical (checked for 1 and 4 layers). What changes is the global RNG stream after init, which drives the dropout masks. The conclusion (new trainings, not continuations) stands.

**Notes (no change needed).**
- R2-D3a passes on every seed individually: key cos max 0.291, redundant min 0.861, ratio max 0.061. The 3-seed CIs are wide ([-0.71, 0.51] and [-0.57, 0.70]).
- R2-D3b B1|mix meets the voi part (cut 0.310, KL difference [0.0050, 0.0568]) and fails the P0 part (0.250 < 0.35). RESULTS states this correctly.
- The closest R2-D2b miss is rule-110 F stiff|sw: 0.237 against the 0.25 bar, with the KL-difference CI and the random-order comparison above 0. RESULTS states this correctly.
- The builder's 05:37Z edit fixed "bridges -24% to -46%" to "-20% to -46%" (conf on B1 is +0.197).

## Edits made during the audit (checked 05:54:30Z on RESULTS-R2.md md5 06060788, PREREG-R2.md md5 5728dc5a)

- At 05:49:05-05:49:22Z the builder ran /tmp/eca/patch_r2_audit.py, which applied this audit's 05:48Z draft. It edited PREREG-R2.md line 1 and the RESULTS-R2 sources, rebuilt RESULTS-R2.md, and copied the draft to eca/AUDIT-R2.md (05:49:11Z). This message is the final audit text.
- PREREG-R2.md: the body (line 2 onward) is byte-identical to the 04:30:36Z save; only the header changed (see M1).
- RESULTS-R2.md (final) states H1, M2, L1, L2, L3 and L5 accurately. Its new numbers match the re-derivation: B2 0.049; 3-seed means 0.006-0.080; about 0.008; 0.066 [0.049, 0.082]; 89 of 90 with lam 10-32 and 38-66%; 0.73 / 0.81 and about 0.08; every seed passes. Sec. 10 is tables.md verbatim.
- Remaining slips in the final RESULTS-R2.md:
  - The TL;DR still says "rule 90: ... B2 0.05 nats" and "vs 0.005-0.008 at 4 layers" (L4 was applied in Sec. 3 only).
  - The header says PREREG-R2's "last save, a header-timestamp edit, came at 04:30:36Z", and Sec. 9 says "PREREG-R2's last save came 15-23 s after the launch commands". The last save is now 05:49:11Z, and "header-timestamp edit" cannot be checked.
  - Sec. 9 dates this pass "05:26-05:47Z" and says L1-L5 were all applied. The pass ran until the time on the Auditor line, and L4 is only partly applied.
  - Sec. 5's "(Sec. 9 table)" now points at the audit section; the tables are Sec. 10.

## Could not check

- The content of PREREG-R2.md before its 04:30:36Z save: there is no earlier copy and no git, and the history search timed out.
- True GPU spend: the ledger is a ceiling at $0.87/h, and the Billing API is disabled.
- RESULTS-R2's "MATS cluster: not used".
- Gemini's "20,000:1 variance" ratio: quoted from overnight-followups-round-1.md line 74, not recomputed.
- An exact replay of the as-run D3b chains: the GPU RNG state is not available, so the CPU replay matches to within 0.0054 per seed. The B2 support control was not run.

## Side effect of this audit

My init-equality check at 05:39:33Z imported /tmp/eca/ecanet-r1.py and the frozen smallnet/netdef.py without disabling bytecode, which wrote /tmp/eca/__pycache__/ecanet-r1.cpython-312.pyc and /tmp/eca/src-eca-r2b-s0-1792038617/smallnet/__pycache__/netdef.cpython-312.pyc. Both were outside my scratch. I removed them with trash-put at 05:45:38Z (restorable with trash-restore); that changed the directory mtimes of /tmp/eca and of that smallnet/ dir. The source files are unchanged (md5 db3e6121, ce9db98f). Nothing under eca/ was written: eca/__pycache__ keeps its 01:04-04:27Z files.

## Commands and scripts

All run under /tmp/eca/audit-r2/, CPU only, at most 2 processes, PYTHONPATH=/tmp/smallnet/pylib.
- Timing: `stat` / `ls --time-style=full-iso` on PREREG-R2.md, the src-eca-r2* dirs, the raw files and msg-r2.txt; `grep LAUNCH|END /tmp/eca/logs/launch-eca-r2*.log`; grep of the CREATE / RUN / DELETE lines in logical-soup/GPU-LOG.md; `grep -r R2-D2a|PREREG-R2 /tmp/eca`.
- Code: `md5sum` over the tree, the frozen copies and the round-1 references; `diff /tmp/eca/ecarun-r1-float64.py eca/ecarun.py > ecarun-r1-vs-r2.diff`; `diff /tmp/eca/ecanet-r1.py eca/ecanet.py`; inline torch init-equality check; inline model.pt loader (args, layers, perm vs result.json); args diff against main2.
- Re-derivation: `rederive.py [t]` writes rederived.json (exact t) and rederived-t4303.json; `render.py` writes mine-texact.md and mine-t4303.md, compared with `diff` against results/r2/tables.md. An inline script compares summary.json (3,492 numbers). `ECA_OUT_R2=/tmp/eca/audit-r2/rerun-analyze python3 eca/analyze_r2.py`, then `cmp` / `md5sum`. `d1check.py` (D1 from d1-pairs.csv). Round-1 comparison numbers come from main2 result.json and CSVs via rederive.load.
- GPU: inline regex parser over logical-soup/GPU-LOG.md; `gcloud compute instances list --project GCP_PROJECT --filter='name~soup-gpu-eca' --format=json` (05:36:30Z, saved as gcloud-eca-0536.json; repeated 05:55:10Z).
- Ablation (helper process, same rules): ablation/d3b_keycos.py (05:38-05:41Z), ablation/d3b_support_control.py (05:42-05:43Z), ablation/p0_val_replay.py (05:41Z). Each imports the frozen ecarun/ecanet from ablation/src (md5-identical copies), loads model.pt on CPU and writes .json / .log beside the script.
- Final state: `md5sum -c raw-checksums.txt` and md5 of the inputs at 05:55:09Z; final_check.py (phrase and section checks on the final RESULTS-R2.md and PREREG-R2.md, 05:54:30Z); the 04:30:36Z and final versions are kept as snap-0527/PREREG-R2.md, PREREG-R2-5728dc5a.md, RESULTS-R2-d59e2588.md and RESULTS-R2-06060788.md.
- Tables in this file: generated from rederived.json by an inline script into tables-for-audit.md, then build_audit.py.
