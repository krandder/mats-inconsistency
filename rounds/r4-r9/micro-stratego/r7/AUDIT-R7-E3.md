# AUDIT-R7-E3: independent audit of R7-E3 (E-MS-TRUST-SEARCH, 6x6 Stratego)

Auditor: a read-only board investigator subagent running on Opus 5.5 (claude-opus-5-5[1m]), separate from the executor.
Audit run 2026-10-17 10:03Z-10:29Z UTC. Written here because this auditor may only write in its own scratch folder:
`/home/agent/workspaces/fable-strategy/scratch/audit-r7e3/AUDIT-R7-E3.md` (the brief's target was
`/tmp/r7e3/audit/AUDIT-R7-E3.md`; RESULTS points to `r7/AUDIT-R7-E3.md`, so the coordinator should copy it there).
Paths below are relative to `/home/agent/workspaces/mats/work/micro-stratego/` unless absolute.

Files audited (sha256 at audit time): `r7/PREREG-R7-E3.md` f3868522c283b7fd...; `r7/RESULTS-R7-E3.md` 5124d331af664689...
(mtime 10:04:33Z, re-read at 10:22Z after it changed during the audit); spec
`/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md` cbece88d738601336e5a... (equals the hash the prereg cites).
Nothing in r7/, r6/, r5/, /tmp/r5e2 or /tmp/r7e3 was edited; no git, no GPU, no gcloud create/delete, no messages.

## Verdict

**Supported with corrections.** With my own code I re-derived every pre-registered statistic from the per-item CSVs and
got the same verdicts as `r7/report/verdicts.csv` and RESULTS-R7-E3.md: the E3-D0 gate passes, E3-D1 passes through K3c
(K3 fails its own bar), and E3-D2 fails: part (1) passes, part (2) fails because no trust rule beats unif_pool, and
trust_resample is significantly worse. For a rule statistic, the largest difference from the JSON summaries is 9.2e-5
(rounding in the CSVs). No verdict is within rounding distance of its bar. An independent CPU recomputation from the seed-101 checkpoint, with
my own forward pass, tapes and rollout code, matches the D0 anchor and route-B KLs (means within 4e-6 nats). It also
matches both rollout tiers within sampling error. Nothing oracle-derived enters any fit or selection. The
pre-registration hash, the timeline and the verbatim rules check out, and the superseded first hash rebuilds exactly
from the disclosed one-sentence change. The corrections are disclosure and wording fixes. None changes a verdict.

## Checks A-H

| check | result | evidence (read-only checks, UTC) |
|---|---|---|
| A. Pre-registration | PASS (one disclosure correction, C1) | `sha256sum r7/PREREG-R7-E3.md` = f3868522... = `r7/PREREG-R7-E3.sha256` (10:24Z). Hash 09:40:03Z < `r7/gpu/launches.log` first line 09:40:26Z < PENDING-CREATE (GPU-LOG.md line 2096, minute stamp 09:40Z; the VM name stamp 1017094026 = 09:40:26Z). Section 1 body = spec lines 139-147 byte for byte (1,892 bytes). The superseded hash 9824d81b... rebuilds exactly when the final file's header sentence is swapped for the draft's (`/tmp/r7e3/prereg_head.md`), so that sentence was the only change. The amendments file (09:24:14Z) has no R7-E3 content. Every number disclosed in item 11 is in the smoke outputs. But the smoke also computed the full test D0 battle and NLL numbers for seed 101 (see C1). No design change after any test number: details below. |
| B. No oracle in fitting or selection | PASS | grep over r7lib.py, run_r7e3.py, r6/e3lib.py, r6/run_e3.py, r5/model6.py, r5/train6.py: oracle files are loaded only at run_r7e3.py:223 (part_d1), :309 (part_d2) and r6/run_e3.py:161,170 (part_d0). fit_aggregators and fit_trust_select take only the calibration split Rc. Their targets are model-only references (coverage_reference, mean leaf value) and `Rc["result"]`, the realised result from the root player's view (generate6.py:68 `result_for(final, P)`). tape_from reads slot_true only where publicly revealed (model6.py:304-309). Trust-training roots perm[:4000] and selection roots perm[4000:5000] are disjoint slices of one permutation. |
| C. Headline numbers from per-item CSVs | PASS | Own script audit_cd (10:09Z): 231 JSON quantities recomputed (77 per seed). Largest difference 3.2e-4 (descriptive pruned-exploratory share, from beta rounding). For a rule statistic the largest is 9.2e-5 (strata Spearman of -H). MSE differences are < 5e-8. Pools rebuilt from d2_rollouts.csv against d2_actions.csv: unif, trust_pool and trust_resample within 7.5e-7, stiff2 within 4.4e-6, trust_gate within 3.6e-5. The 3.6e-5 case is seed 101, pair 1648: 16 betas between 19.903 and 20.000 (std 0.031) near the beta cap of 20, where 6-digit rounding is coarse; its MSE is unchanged to 8 digits. Rounding moves D2 (1d) from 0.2497 [0.2107, 0.2887] to 0.2497 [0.2105, 0.2889]. The nearest margin to a bar is 0.05 (D2 (1d) against 0.20). |
| D. Pre-registered verdicts, own seed-t CIs (t = 4.303) | PASS | All 21 sub-rule rows and all part verdicts equal verdicts.csv and the RESULTS verdict table (numbers below). "That key head" = K3c: K3c passes (1) and K3 fails. K3c also has the larger calibration gap (0.674 vs 0.249), so the tie-break would pick it too. attn_L3 with K3 gives the same D1 (2) outcome (JSD 0.00023, KL 0.1016). |
| E. Independent CPU recomputation, seed 101 | PASS | (i) audit_e1 (10:13Z), 200 calibration D0 positions. My forward pass matches r5 model6.RealNet to 9.3e-6. Anchor KL mean 0.184697 vs CSV 0.184700 (max per-position difference 1.6e-4). Route-B KL 0.088428 vs 0.088430 (max 4.4e-4). n_hidden identical. The run's clamp of messages to [-8, 8] never binds here. (ii) audit_e2 (10:15Z) and audit_e2b (10:20Z): my own clean and exploratory tiers on 506 (s, a) pairs from 48 random test positions, compared pair by pair with the executor's rollouts on the same pairs (table below). In the first replicate, 16 of 18 differences have \|z\| <= 1.50. Two clean-tier values sat at z = -2.00 (cone_lp) and +2.25 (H), which 18 tests produce by chance (Bonferroni bar \|z\| = 2.99). Both vanished in a 16-per-pair replicate with a new seed (z = -0.25 and +0.01). |
| F. Code vs prereg spot checks | PASS | Each item checked in the code (list below): K3/K3c, fitting order and data, the "that key head" rule, tier temperatures, swap, no tier label, untempered cone_lp/ply_lp, trust data, disjoint selection data, the gate formula, the resample construction, stiffness weights, the strata and the within-pair Spearman. The staged code copy (`/tmp/r7e3/src`, 09:40:09Z) is byte-identical to the current r5/, r6/ and r7/ files used. |
| G. Compute | PASS | Ledger $0.27 = 0.31 h x $0.87/h for the one VM, soup-gpu-r7e3-main-1017094026-8653 (GPU-LOG.md DELETE line, age_h 0.31, cost_usd 0.27). The VM name stamp (09:40:26Z) and DELETE minute (09:59Z) bound the age at 0.31-0.33 h, so $0.27-$0.28. That is under the $1.50 ceiling. `gcloud compute instances list` and `disks list` (project GCP_PROJECT, filter name~soup-gpu-r7e3-) were empty at 10:07:04Z/10:07:07Z and again at 10:27:18Z. Positive control: the same queries with name~soup-gpu- list the R7-E2 and R7-E1 VMs and disks, so empty means none. |
| H. RESULTS-R7-E3.md | PASS with corrections C2-C6 | Every rule has a verdict with per-seed values, mean and seed-t interval. Failures are stated as failures (K3, all three unif comparisons, E3-D2). Post hoc items are labelled. 43 groups of numbers checked against the files, all match (list below). Overreach and wording: C2-C6. |

## Details

### A. Timeline (file mtimes and log stamps; these are executor-controlled)

- 09:13Z spec; 09:22:19Z `/tmp/r7e3/diag_temp.py` (legal policy mass, 256 CALIBRATION roots); 09:24:14Z amendments file.
- 09:29:27Z `r7/run_r7e3.py` last change; 09:29:37Z `/tmp/r7e3/test_tier.py` (tier check, 48 CALIBRATION roots).
- 09:30:11Z first smoke starts. 09:32:49Z `r7/r7lib.py` last change (the disclosed argsort fix). The fixed torch argsorts
  are first called in fit_trust_select, and main() runs fit, then trust, then d0, d1, d2. So the crashed first attempt
  can have printed only calibration fit numbers. Its log was overwritten, so this is inferred.
- 09:34:14Z `r7/report_r7e3.py` last change. 09:34:27Z and 09:35:19Z prereg drafts (`/tmp/r7e3/prereg_head.md`,
  `prereg_tail.md`). The hashed sections 2-6 (items 1-10) equal these drafts. The only additions are the verbatim rule
  block, item 11 and section 7.
- Smoke outputs (`/tmp/r7e3/smoke/run.log`): fit 09:34:40 and trust 09:34:56 (calibration); d0 09:36:49; d1 09:37:59;
  d2 09:38:52 (test). So the code and the operational definitions were last written before the smoke printed its first
  test number.
- 09:39:45Z superseded hash; 09:40:03Z hash; 09:40:09Z staging; 09:40:26Z launch; 09:58:22Z all seeds exit 0; ~09:59Z
  VM deleted. Post-run files: `r7/expl_r7d2.py` 10:01Z (post hoc, labelled), `r7/gpu/ledger.py` 10:01Z (ledger tool
  only).
- Sensitivity (audit_smoke_excl): without the smoke-seen test positions (D1 first 48, D2 first 12), every D1 and D2
  verdict is unchanged. K3 gap 0.2317 [0.2141, 0.2494] fails. K3c 0.6586 [0.6423, 0.6749] passes. JSD ratios 12.7 and
  11.8 with difference CIs > 0. KL differences 0.0071, 0.0206, 0.0182, all CIs > 0. D2 (1): 0.350, 0.370, 0.745,
  0.250, all pass. unif minus gate/pool/resample: -0.00023 [-0.00069, 0.00022], -0.00043 [-0.00103, 0.00018],
  -0.00064 [-0.00117, -0.00011], all fail.

### D. Own verdict recomputation (from CSVs; 3-seed mean [95% seed-t CI])

| rule | per seed | mean [CI] | verdict |
|---|---|---|---|
| D0 (0) anchor KL >= 0.12 | 0.1859, 0.1856, 0.1841 | 0.1852 [0.1828, 0.1876] | pass |
| D0 (1) battle >= 0.99 / 7-known NLL <= 0.01 | 0.9999, 1.0000, 0.9999 / 0.0001, 0.0002, 0.0001 | 0.9999 / 0.0001 | pass |
| D0 (2) cut >= 0.40 | 0.4805, 0.4779, 0.4880 | 0.4821 [0.4690, 0.4952] | pass; gate PASS |
| D1 (1) K3 gap | 0.2258, 0.2409, 0.2362 | 0.2343 [0.2151, 0.2535] | fail |
| D1 (1) K3c gap | 0.6521, 0.6659, 0.6637 | 0.6606 [0.6421, 0.6790] | pass |
| D1 (2) JSD ratio scaled_mix / attn_L3 (K3c) | 11.26, 8.32, 12.92 | 10.83; diff 0.002109 [0.001856, 0.002361] | pass |
| D1 (2) JSD ratio ataraxos_mmd / attn_L3 (K3c) | 9.97, 7.20, 13.31 | 10.16; diff 0.001941 [0.001220, 0.002662] | pass |
| D1 (2) KL scaled_mix / ataraxos_mmd / kc_dedup(K3c) minus attn_L3 | | 0.0071 [0.0064, 0.0078] / 0.0206 [0.0138, 0.0273] / 0.0181 [0.0099, 0.0264] | pass; D1 PASS |
| D2 (1a) / (1b) | 0.3276, 0.3807, 0.3337 / 0.3617, 0.3960, 0.3498 | 0.3473 [0.2751, 0.4195] / 0.3691 [0.3096, 0.4287] | pass |
| D2 (1c) / (1d) | 0.7380, 0.7801, 0.7093 / 0.2322, 0.2628, 0.2540 | 0.7425 [0.6541, 0.8309] / 0.2497 [0.2105, 0.2889] | pass |
| D2 (2) trust_gate | cut 0.6235, 0.5976, 0.6968 | cut 0.6393 (stiff - gate 0.02481 [0.01800, 0.03161]); unif - gate -0.00028 [-0.00076, 0.00021] | fail |
| D2 (2) trust_pool | cut 0.6233, 0.5837, 0.6961 | cut 0.6344 (0.02462 [0.01722, 0.03202]); unif - pool -0.00046 [-0.00105, 0.00013] | fail |
| D2 (2) trust_resample | cut 0.6156, 0.5831, 0.6902 | cut 0.6296 (0.02443 [0.01730, 0.03157]); unif - resample -0.00065 [-0.00112, -0.00017] | fail; D2 FAIL |

Note: every seed's JSD ratio is at least 7.2, but the seed-t interval of the ratio itself is wide ([2.56, 17.77] for
ataraxos_mmd). The prereg tests ratios through the 3-seed mean and the interval of the difference, so the pass stands
as pre-registered.

Robustness: with the spec's literal stiffness base (1 - H/log 2)^2, stiff_pool's MSE is 0.0267 [0.0227, 0.0307]
unclamped and 0.0895 clamped at 0. The trust rules still beat the unclamped version by 0.47-0.48 (bar 0.20). D2 (2)
fails on the unif comparison, which does not involve the stiffness base (audit_d2b).

### E (ii). Rollout tiers: mine vs the executor's, same 506 pairs (seed 101); pair-clustered 95% CI of the difference

| tier, statistic | mine | executor | difference [CI] | z | d2.json, all 2,361 pairs |
|---|---|---|---|---|---|
| clean, share H < 0.15 | 0.1277 | 0.1282 | -0.0005 [-0.0109, 0.0099] | -0.09 | 0.1421 |
| clean, KL(Q* \|\| q) | 0.3524 | 0.3624 | -0.0099 [-0.0275, 0.0076] | -1.11 | 0.3643 |
| clean, squared value error | 0.0509 | 0.0528 | -0.0020 [-0.0046, 0.0006] | -1.50 | 0.0463 |
| clean, flag-capture endings | 0.1198 | 0.1201 | -0.0002 [-0.0104, 0.0099] | -0.05 | 0.1164 |
| clean, piece-count violations | 0.0005 | 0.0000 | +0.0005 [-0.0002, 0.0012] | +1.42 | 0.0001 |
| clean, cone_lp | -1.0682 | -1.0592 | -0.0091 [-0.0179, -0.0002] | -2.00 | -1.0146 |
| clean, ply_lp | -2.2085 | -2.2013 | -0.0073 [-0.0280, 0.0134] | -0.69 | -2.1954 |
| clean, H | 0.6793 | 0.6691 | +0.0102 [0.0013, 0.0192] | +2.25 | 0.6475 |
| clean replicate (16 per pair, seed 777): H < 0.15 / KL / sq err / cone_lp / ply_lp / H | 0.1357 / 0.3741 / 0.0547 / -1.0600 / -2.2012 / 0.6691 | 0.1282 / 0.3624 / 0.0528 / -1.0592 / -2.2013 / 0.6691 | | +1.64 / +1.55 / +1.52 / -0.25 / +0.01 / +0.01 | |
| exploratory, share H < 0.15 | 0.0842 | 0.0889 | -0.0047 [-0.0160, 0.0066] | -0.81 | 0.0931 |
| exploratory, KL | 0.3429 | 0.3397 | +0.0033 [-0.0187, 0.0252] | +0.29 | 0.3540 |
| exploratory, squared value error | 0.0549 | 0.0524 | +0.0025 [-0.0014, 0.0065] | +1.26 | 0.0471 |
| exploratory, flag-capture endings | 0.0768 | 0.0805 | -0.0037 [-0.0144, 0.0069] | -0.68 | 0.0748 |
| exploratory, piece-count violations | 0.0351 | 0.0311 | +0.0040 [-0.0040, 0.0119] | +0.97 | 0.0337 |
| exploratory, cone_lp | -3.3520 | -3.4152 | +0.0632 [-0.0366, 0.1631] | +1.24 | -3.5430 |
| exploratory, ply_lp | -3.6457 | -3.6314 | -0.0143 [-0.1015, 0.0729] | -0.32 | -3.6642 |
| exploratory, moves with untempered p < 1e-3 per move | 0.1248 | | | | 0.1278 |

The same-pair comparison is the valid test. My 48 positions are a subset, so the d2.json column (all pairs) differs by
position mix (for example, clean squared error 0.0528 on these pairs vs 0.0463 overall in the executor's own data).

### F. Code spot checks (file:line)

- K3: `r7lib.py:46-67`, s / (||s|| + 1e-6) over the 8 slot norms of dl (`e3lib.py:157`); W_K Linear(8, 16, bias=False);
  target s. K3c: `r7lib.py:51-80`, bag mean over valid rollouts of the same bag; Linear(16, 16, bias=False); target st.
  L_key: `e3lib.py:207-211` (off-diagonal Gram difference).
- Fitting order and data: `run_r7e3.py:43-145`. Calibration roots only. scaled_mix (unskewed, lr 1e-2), ataraxos_mmd
  (Beta(0.5, 0.5) bags, lr 1e-2), then per head kc_dedup + key + L_key (lr 3e-3), key frozen, then attn_L3 (lr 1e-3).
  1,500 roots, 1,500 steps, pools 32 + 32, references 64 + 64 (union call). All as prereg.
- "That key head": `report_r7e3.py:102-106` (the passing head; else the larger 3-seed calibration gap).
- Tiers: `r7lib.py:85-114` (cone types from softmax(logits / 2.5), stiff_voi order from the untempered belief + slot
  trust), `:118-134` (swap of 2 materialised cone slots), `:162` (plies at softmax(logits / 2.0) over all 1,296 moves),
  `:163` and `:258` (ply_lp untempered), `:247-248` (cone_lp = untempered teacher-forced log-prob of the final, swapped
  types in sampling order). The clean tier is r5 `sample_cone` + `value_rollout` at temperature 1.
- No tier label: feats_x = r5 trust features (h_root, n_ply, transition entropies, path disagreement, |l - l0|, H, e,
  terminal) + cone_lp + ply_lp (`r7lib.py:257-260`). The tier vector Ts is used only for reported means
  (`run_r7e3.py:176, 182-183`).
- Trust data: `run_r7e3.py:163-179`. n_ply >= 1, t in [4, 36], the played move (`r6/run_e3.py:124-127`), 8 + 8
  rollouts, excess surprise against the realised result, Adam 1e-3, weight decay 1e-4, 3,000 steps of 512
  (`r7lib.py:283-300`). Selection on the disjoint perm[4000:5000] (12 clean + 8 exploratory), log loss against the
  realised result, ties to the first grid value (Python min).
- trust_gate: `r7lib.py:318-322`, torch.quantile (linear interpolation), std with ddof 1, + 1e-6. trust_resample:
  `run_r7e3.py:367-369`, c1..c6 + x1..x6, stable argsort drops the 4 lowest (earlier index on ties), adds c9..c12, mean
  of 12. stiff_pool: `r7lib.py:313-315`, (1 - H/log 3)^2. Strata: `run_r7e3.py:353-358`, stable rank, (i * 10) // n.
  Within-pair Spearman: `run_r7e3.py:344-352`, nan when constant, nanmean; no pair was skipped in any seed.

### H. Numbers in RESULTS-R7-E3.md checked against the files (line numbers of the 10:04:33Z version)

All match (3-seed means from my scripts or from the JSONs):

- L17-18: 48.2% [46.9, 49.5]; 0.185 -> 0.096; 0.047.
- L20-22: 0.234 [0.215, 0.254]; K4 0.228 (r6 d1.json 0.2276); 0.661 [0.642, 0.679]; -0.133.
- L23-24: 10.8x and 10.2x.
- L27-29: 0.347 [0.275, 0.420]; 0.369 [0.310, 0.429]; -0.395; -0.551; 0.743; 0.250.
- L31-33: 0.0388; 0.01366; 0.01394 and -0.0003 [-0.0008, 0.0002]; 0.01412 and -0.0005 [-0.0011, 0.0001]; 0.01431 and
  -0.0006 [-0.0011, -0.0002].
- L35-39: 0.0476 vs 0.0453; KL 0.363 vs 0.361; 9.4% vs 14.2%; beta 3.88 vs 3.95; 50.4% (JSON 0.5036); 0.0155 vs 0.0137
  (difference 0.0018 [0.0013, 0.0023]).
- L40-43: 2.4x and 3.3x; 0.0115 of 0.0137; 0.0004 and 0.0005; $0.27 and 0.31 h.
- L49-73 verdict table: every cell.
- L77-80: d0.json identical to r6 d0.json in every field except run time, all seeds; 0.078 (route-B 0.0776).
- L82, L89-90, L92-94: cosines 0.869 and 0.635; kc_dedup 0.1188 and 0.1196; attn_L3 0.1016 and 0.1014; JSD 0.00022,
  0.00233, 0.00216; KL 0.1014, 0.1086, 0.1220, 0.1196, 0.1004.
- L98-101: 0.213 and 0.225 (R5-E2 head); 13.9; 0.0003; 0.0006.
- L105-107: 7.4% vs 11.7%; 3.7% vs 0.02%; 13% vs 0.4% (0.1296 and 0.0042 per move); 1.057 vs 1.077 (trust.json).
- L111-112: 8.0; 0.0388 vs 0.0137; gamma 0 in every seed, also 0 in all R6-E3 seeds (r6 d2.json).
- L116 and L123-130: 1.0M parameters (1,001,644); alpha 0.5; (0.35, 0.5), (0.25, 0.5), (0.35, 0.5); 64,000 rollouts;
  2,361 pairs; 56,664 rollouts; 796 s per seed (done.json); parameter hashes equal train_meta.json param_hash_final
  for all seeds.
- L136-161 descriptive table: every row. That includes the calibration log losses 0.6267, 0.6253, 0.6261, 0.6280,
  0.6996, 0.6344 and the delusional counts (stiff2 only, 1 position in seeds 102 and 103).
- L165-173 post hoc: 0.090 vs 0.038; 0.128 vs 0.039; 0.251 [0.189, 0.313]; 0.272 [0.254, 0.290]; -0.181; -0.368;
  0.0022; 0.0115; -0.021 (my own computation gives -0.02053 [-0.02776, -0.01330]); 0.00042 [0.00027, 0.00056];
  0.00046 [-0.00013, 0.00105].
- L186-190: VM, times, $0.27; no VM or disk left (my gcloud checks above).
- L208 self-check: 117 rows, largest difference 1.9e-5 (`r7/report/selfcheck.csv`, summary.json).

## Corrections (exact suggested wording)

C1 (disclosure; the prereg is hashed, so add to `r7/PREREG-R7-E3-AMENDMENTS.md` and to the RESULTS deviations log,
after "The pre-hash CPU smoke used 12 D2 and 48 D1 test positions"):
"Correction to prereg section 6 item 11(c): besides 48 calibration and 48 test D0 positions for the gate KL, the
smoke's D0 part also computed, for the seed-101 checkpoint, battle accuracy on all 11,034 test battles (0.99991), the
7-known-slot NLL (9.4e-05) and the one-pass belief KL (0.0484) on all test D0 positions. These are the inherited R6-E3
gate numbers. They are deterministic and identical to the published R6-E3 values, so the smoke showed no new test
information." (Evidence: `/tmp/r7e3/smoke/eval/d0.json`; `r6/run_e3.py:160-165` ignore the smoke limit.)

C2 (TL;DR, line 34, scope): replace "The spec's premise did not hold in this testbed: the exploratory rollouts were not
hallucination traps." with "With the exploratory tier built here (cone types drawn at temperature 2.5 plus one type
swap, plies at temperature 2.0; prereg section 6 items 1-3), the exploratory rollouts were not hallucination traps. The
spec's other exploratory option (a cone drawn uniformly over hidden slots) and a clean tier at temperature 0.3 were not
run."

C3 (TL;DR, line 40: the evidence in brackets is about H, not about trust): replace "trust does flag the worse rollouts
(terminal-like claims have 2.4-3.3x the squared error of the others)" with "terminal-like claims (H < 0.15) have
2.4-3.3x the squared error of the others, and trust ranks them low. A rollout with H >= 0.15 has the higher beta with
probability 0.67 [0.59, 0.74], and terminal-like rollouts are 21% of trust_resample's pruned rollouts against 12% of
its initial batch (auditor's numbers from d2_rollouts.csv)."

C4 (TL;DR line 42 and post hoc lines 171-173, "matches"): TL;DR: "Unequal weights add about 0.0004 of noise by a
fixed-weight estimate, about the size of trust_pool's extra MSE (0.0005 [-0.0001, 0.0011])." Post hoc: replace "which
matches its extra MSE (0.00046 [-0.00013, 0.00105])" with "which is about the size of its extra MSE (0.00046
[-0.00013, 0.00105]). The noise estimate treats the weights as fixed, although beta depends on each rollout's own e and
H".

C5 (post hoc line 169-171, optional precision that supports the claim): after "plus Q*'s own noise from 64 playouts",
add: "Q*'s playout noise is about 0.0022 of it (plug-in variance / 64, mean over the 2,361 pairs), so the rollouts'
shared bias is about 0.0093 of the uniform pool's 0.0137."

C6 (line 207-208): copy this file to `r7/AUDIT-R7-E3.md` and replace "(pending at the time of writing; its verdict is
added below when it is in)" with ": supported with corrections. Every verdict was reproduced from the per-item CSVs,
and the D0 KLs and both rollout tiers were recomputed independently from the seed-101 checkpoint on CPU."

Optional (style): TL;DR line 26 "ranks rollout errors well" -> "ranks rollout errors above the pre-registered bars".

## What I could not check

- GPU draws cannot be reproduced on CPU. E (ii) is a statistical check: one seed, 506 pairs, 8 + 8 rollouts per pair
  plus a 16-per-pair clean replicate. E (i) is deterministic up to TF32 on the GPU vs fp32 here (largest
  per-position difference 4.4e-4).
- Seeds 102 and 103 were checked from their CSVs, JSONs and parameter hashes only, not recomputed from checkpoints.
- The calibration fits were not re-run: key heads, kc_dedup, attn_L3, scaled_mix, ataraxos_mmd, theta_trust and the
  selection grids. I checked their code, data sources and outputs (fit.json, trust.json). The D1 key cosines were
  re-aggregated from per-position values, not recomputed from keys, because per-rollout keys and saliences are not
  saved. So the RESULTS explanation of how centring creates the K3c gap cannot be tested from the files. RESULTS marks
  part of it as not measured.
- The timeline rests on executor-controlled file mtimes and log stamps. The first smoke attempt's log was overwritten.
- True spend: $0.87/h is an unverified list-price ceiling (Billing API disabled).
- The declared alternatives were not run: a clean tier at temperature 0.3 and a uniform-random exploratory cone. Only
  the log 2 stiffness base was tested (from the CSVs).
- I did not run `report_r7e3.py`, `expl_r7d2.py` or `gpu/ledger.py`, because they write into r7/. The page
  `/home/agent/memo-serve/lir-agent/r7-e3.html` was not audited.
- No text was classified, so the 30-item hand-label rule did not apply. Every number check is a direct comparison.

## Commands I ran (all read-only; UTC)

- 10:03Z `cat /tmp/r7e3/audit_brief.md`; listings of r7/, runs/, report/, gpu/; `sed -n` of spec sections 0 and 2.
- 10:04-10:06Z `cat` of the prereg, amendments, sha256 file, launches.log, GPU-LOG-R7-E3.md, RESULTS, r7lib.py,
  run_r7e3.py, report_r7e3.py, r6/e3lib.py, r6/run_e3.py, r5/model6.py, parts of r5/train6.py and r5/eval6.py;
  verdicts.csv, summary.json, selfcheck.csv; the per-seed JSONs.
- 10:05Z `sha256sum` of `/tmp/r7e3/src/*` against current r5/r6/r7 files (all SAME); `diff` of the prereg drafts
  against the hashed prereg; `cat` of diag_temp.py, test_tier.py, the smoke log and launch-r7e3-main.log;
  `grep r7e3` in `/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md`.
- 10:07Z and 10:27Z `gcloud compute instances list|disks list --project GCP_PROJECT --filter="name~soup-gpu-r7e3-"`,
  plus the name~soup-gpu- control.
- `grep -n "oracle|pstar|qstar|slot_true|np.load|.npz|result"` over r7lib.py, run_r7e3.py, r6/e3lib.py, r6/run_e3.py,
  r5/model6.py, r5/train6.py.
- The python scripts below, run as
  `OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r5e2/pylib R5E2_DATA=/tmp/r5e2/data python3 - <<'EOF' ... EOF`
  in one process, torch at 4 threads. Times: audit_cd 10:10Z, audit_e1 10:13Z, audit_e2 10:15Z (211 s), audit_e2b
  10:20Z (123 s), and audit_h, audit_h2, audit_d2b, audit_smoke_excl, prereg_hash and gate_diff 10:22-10:26Z.

## Appendix: scripts (verbatim; paste each into the command above to re-run)

### audit_cd (checks C and D)

```python
# audit_cd: re-derive R7-E3 headline numbers and verdicts from the per-item CSVs (own code; not report_r7e3.py)
import csv, json, math, numpy as np
from scipy.stats import spearmanr, rankdata
RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs"
SEEDS = (101, 102, 103); T = 4.302652729749464; LOG3 = math.log(3)
def rd(s, f):
    with open(f"{RUN}/s{s}/eval/{f}") as fh:
        r = list(csv.DictReader(fh))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}
def js(s, f): return json.load(open(f"{RUN}/s{s}/eval/{f}.json"))
def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = T * v.std(ddof=1) / math.sqrt(len(v)); return m, m - h, m + h
def sp(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(spearmanr(x, y).correlation)
out = {s: {} for s in SEEDS}; jsn = {s: {} for s in SEEDS}; maxdiff = {}
def put(s, k, mine, theirs):
    out[s][k] = mine; jsn[s][k] = theirs
for s in SEEDS:
    d0, d1, d2, fit, tr = js(s, "d0"), js(s, "d1"), js(s, "d2"), js(s, "fit"), js(s, "trust")
    b = rd(s, "d0_battles.csv"); n7 = rd(s, "d0_nll7.csv"); g = rd(s, "d0_gate_cal.csv"); gt = rd(s, "d0_gate_test.csv")
    put(s, "battle", b["correct"].mean(), d0["battle_acc_ply0"]); put(s, "nll7", n7["nll7"].mean(), d0["nll_7known"])
    put(s, "anchor_cal", g["kl_anchor"].mean(), d0["cal"]["kl_anchor"]); put(s, "pool16_cal", g["kl_pool16"].mean(), d0["cal"]["kl_pool16"])
    put(s, "red_cal", 1 - g["kl_pool16"].mean() / g["kl_anchor"].mean(), d0["cal"]["reduction_pool16_vs_anchor"])
    put(s, "routeb_cal", g["kl_routeb_all"].mean(), d0["cal"]["kl_routeb_all"]); put(s, "p0_cal", g["kl_p0"].mean(), d0["cal"]["kl_p0"])
    put(s, "red_test", 1 - gt["kl_pool16"].mean() / gt["kl_anchor"].mean(), d0["test"]["reduction_pool16_vs_anchor"])
    c = rd(s, "d1_positions.csv")
    for h in ("K3", "K3c"):
        gL = c[f"key_within_{h}_L"] - c[f"key_across_{h}_L"]; gR = c[f"key_within_{h}_R"] - c[f"key_across_{h}_R"]
        with np.errstate(invalid="ignore"):
            gp = np.nanmean(np.stack([gL, gR]), 0)
        put(s, f"gap_{h}", np.nanmean(gp), d1[f"key_gap_{h}"])
        put(s, f"within_{h}", np.nanmean(np.r_[c[f"key_within_{h}_L"], c[f"key_within_{h}_R"]]), d1[f"key_cos_within_{h}"])
        put(s, f"across_{h}", np.nanmean(np.r_[c[f"key_across_{h}_L"], c[f"key_across_{h}_R"]]), d1[f"key_cos_across_{h}"])
        out[s][f"calgap_{h}"] = fit[f"cal_key_gap_{h}"]; out[s][f"npos_nan_gap_{h}"] = int(np.isnan(gp).sum())
    for a in ("union", "mix", "scaled_mix", "ataraxos_mmd", "kc_dedup_K3", "kc_dedup_K3c", "attn_L3_K3", "attn_L3_K3c"):
        put(s, f"jsd_{a}", c[f"jsd_{a}"].mean(), d1[a]["jsd"]); put(s, f"kl_{a}", c[f"kl_{a}"].mean(), d1[a]["kl"])
    r = rd(s, "d2_rollouts.csv"); A = rd(s, "d2_actions.csv"); P = len(A["pos"])
    assert len(r["pair"]) == 24 * P and (r["slot"].reshape(P, 24) == np.arange(24)).all() and (r["pair"].reshape(P, 24) == np.arange(P)[:, None]).all()
    assert (r["in_mixed"].reshape(P, 24) == ((np.arange(24) < 8) | (np.arange(24) >= 16))).all()
    mx = r["in_mixed"] == 1
    E, H, B, B5, KL = (r[k][mx].reshape(P, 16) for k in ("e", "H", "beta", "beta_r5", "kl"))
    y = -KL.ravel()
    put(s, "sp_trust", sp(B.ravel(), y), d2["spearman_trust"]); put(s, "sp_stiff", sp(-H.ravel(), y), d2["spearman_stiff"])
    put(s, "sp_r5", sp(B5.ravel(), y), d2["spearman_trust_r5head"])
    wt = [sp(B[p], -KL[p]) for p in range(P)]; ws = [sp(-H[p], -KL[p]) for p in range(P)]; w5 = [sp(B5[p], -KL[p]) for p in range(P)]
    put(s, "within_trust", np.nanmean(wt), d2["within_trust"]); put(s, "within_stiff", np.nanmean(ws), d2["within_stiff"])
    put(s, "within_r5", np.nanmean(w5), d2["within_trust_r5head"]); out[s]["within_n_skipped"] = int(np.isnan(wt).sum())
    Hr = H.ravel(); n = Hr.size; order = np.argsort(Hr, kind="stable"); st = np.empty(n, int); st[order] = (np.arange(n) * 10) // n
    stt = [sp(B.ravel()[st == k], y[st == k]) for k in range(10)]; sts = [sp(-Hr[st == k], y[st == k]) for k in range(10)]
    put(s, "strata_trust", np.nanmean(stt), d2["strata_trust"]); put(s, "strata_stiff", np.nanmean(sts), d2["strata_stiff"])
    out[s]["strata_sizes_match"] = np.bincount(st, minlength=10).tolist() == d2["strata_sizes"]
    out[s]["H_unique_csv"] = int(np.unique(Hr).size)
    sel = tr; al, q, tau, gam = sel["alpha"], sel["gate_q"], sel["gate_tau"], sel["gamma"]
    assert d2["selected"] == {"gamma": gam, "alpha": al, "gate_q": q, "gate_tau": tau}
    Ra = {k: r[k].reshape(P, 24) for k in ("e", "H", "beta", "beta_r5", "kl", "terminal", "flag", "uniq", "offsup", "cone_lp", "ply_lp")}
    wm = lambda e, w: (w * e).sum(-1) / np.maximum(w.sum(-1), 1e-12)
    stiffw = lambda Hh, gg: np.clip(1 - Hh / LOG3, 0, None) ** gg
    Qq = np.quantile(B, q, axis=1, keepdims=True); sd = B.std(1, ddof=1, keepdims=True)
    gw = 1 / (1 + np.exp(-(B - Qq) / (tau * sd + 1e-6)))
    def prune(e, bb, k, add=None):
        idx = np.argsort(bb, 1, kind="stable")[:, k:]; keep = np.take_along_axis(e, idx, 1)
        return np.concatenate([keep, add], 1).mean(1) if add is not None else keep.mean(1)
    C16e, C16b, C16b5 = Ra["e"][:, :16], Ra["beta"][:, :16], Ra["beta_r5"][:, :16]
    X8e, X8b = Ra["e"][:, 16:], Ra["beta"][:, 16:]
    pools = {"unif": E.mean(1), "stiff2": wm(E, stiffw(H, 2.0)), "stiff_fit": wm(E, stiffw(H, gam)), "trust_pool": wm(E, B ** al),
             "trust_gate": wm(E, gw),
             "trust_resample": prune(np.c_[C16e[:, :6], X8e[:, :6]], np.c_[C16b[:, :6], X8b[:, :6]], 4, C16e[:, 8:12]),
             "trust_pool_r5head": wm(E, B5), "clean8": C16e[:, :8].mean(1), "explore8": X8e.mean(1), "clean16": C16e.mean(1),
             "trust_trim12_clean16": prune(C16e, C16b, 4), "trust_trim12_clean16_r5head": prune(C16e, C16b5, 4),
             "trust_pool_clean16": wm(C16e, C16b ** al)}
    qe = A["qstar_eff"]
    for k, v in pools.items():
        dd = np.abs(v - A[f"q_{k}"]).max(); maxdiff[(s, k)] = dd
        put(s, f"mse_{k}", ((v - qe) ** 2).mean(), d2[f"mse_{k}"])
        out[s][f"mse_{k}_fromactions"] = ((A[f"q_{k}"] - qe) ** 2).mean()
    out[s]["neff_trust_pool"] = ((B ** al).sum(1) ** 2 / ((B ** al) ** 2).sum(1)).mean(); jsn[s]["neff_trust_pool"] = d2["neff_trust_pool"]
    out[s]["neff_trust_gate"] = (gw.sum(1) ** 2 / (gw ** 2).sum(1)).mean(); jsn[s]["neff_trust_gate"] = d2["neff_trust_gate"]
    w2 = stiffw(H, 2.0); out[s]["neff_stiff2"] = (w2.sum(1) ** 2 / (w2 ** 2).sum(1)).mean(); jsn[s]["neff_stiff2"] = d2["neff_stiff2"]
    pr = np.argsort(np.c_[C16b[:, :6], X8b[:, :6]], 1, kind="stable")[:, :4]
    put(s, "pruned_explore_share", (pr >= 6).mean(), d2["resample_pruned_explore_share"])
    for nm, sl in (("clean", slice(0, 16)), ("explore", slice(16, 24))):
        Z = {k: v[:, sl] for k, v in Ra.items()}; tj = d2[f"tier_{nm}"]
        for k, v in (("kl", Z["kl"].mean()), ("sq_err", ((Z["e"] - qe[:, None]) ** 2).mean()), ("abs_err", np.abs(Z["e"] - qe[:, None]).mean()),
                     ("terminal_like_H_lt_0.15", (Z["H"] < 0.15).mean()), ("flag_end", Z["flag"].mean()), ("count_violation", Z["uniq"].mean()),
                     ("cone_lp", Z["cone_lp"].mean()), ("ply_lp", Z["ply_lp"].mean()), ("beta", Z["beta"].mean()), ("beta_r5", Z["beta_r5"].mean()), ("H", Z["H"].mean())):
            put(s, f"tier_{nm}.{k}", v, tj[k])
    out[s]["mean_signed_err_unif"] = (pools["unif"] - qe).mean()
    out[s]["n_pairs"] = P
V = []
def per(k, src=out): return np.array([src[s][k] for s in SEEDS], float)
def row(name, vals, ok, extra=""):
    m, lo, hi = ci(vals); V.append((name, ", ".join(f"{x:.4f}" for x in vals), f"{m:.4f} [{lo:.4f}, {hi:.4f}]", ok, extra))
anc, bat, nl7, red = per("anchor_cal"), per("battle"), per("nll7"), per("red_cal")
g0, g1a, g1b, g2 = anc.mean() >= 0.12, bat.mean() >= 0.99, nl7.mean() <= 0.01, red.mean() >= 0.40
row("D0(0) anchor KL >= 0.12", anc, g0); row("D0(1) battle >= 0.99", bat, g1a); row("D0(1) nll7 <= 0.01", nl7, g1b); row("D0(2) cut >= 0.40", red, g2)
gate = g0 and g1a and g1b and g2
ph = {}
for h in ("K3", "K3c"):
    v = per(f"gap_{h}"); m, lo, hi = ci(v); ph[h] = bool(m >= 0.28 and lo > 0.25); row(f"D1(1) {h} gap >= 0.28, lo > 0.25", v, ph[h])
d1a = ph["K3"] or ph["K3c"]
head = ("K3" if ph["K3"] else "K3c") if ph["K3"] != ph["K3c"] else max(("K3", "K3c"), key=lambda h: per(f"calgap_{h}").mean())
oks = [d1a]
ja = per(f"jsd_attn_L3_{head}")
for X in ("scaled_mix", "ataraxos_mmd"):
    jx = per(f"jsd_{X}"); m, lo, hi = ci(jx - ja); ok = bool((jx / ja).mean() >= 5 and lo > 0); oks.append(ok)
    row(f"D1(2) JSD ratio {X}/attn_L3_{head} >= 5", jx / ja, ok, f"diff {m:.6f} [{lo:.6f}, {hi:.6f}]")
ka = per(f"kl_attn_L3_{head}")
for X in ("scaled_mix", "ataraxos_mmd", f"kc_dedup_{head}"):
    v = per(f"kl_{X}") - ka; m, lo, hi = ci(v); ok = bool(lo > 0); oks.append(ok); row(f"D1(2) KL {X} - attn_L3_{head} > 0", v, ok)
d1 = all(oks)
st, ss, wt, tt, ts = per("sp_trust"), per("sp_stiff"), per("within_trust"), per("strata_trust"), per("strata_stiff")
a1 = bool(st.mean() >= 0.30 and ci(st)[1] > 0); a2 = bool(wt.mean() >= 0.25 and ci(wt)[1] > 0)
a3 = bool((st - ss).mean() >= 0.55 and ci(st - ss)[1] > 0); a4 = bool((tt - ts).mean() >= 0.20 and ci(tt - ts)[1] > 0)
row("D2(1a) sp(beta,-KL) >= 0.30", st, a1); row("D2(1b) within >= 0.25", wt, a2); row("D2(1c) sp diff >= 0.55", st - ss, a3); row("D2(1d) strata diff >= 0.20", tt - ts, a4)
crit = {}
for X in ("trust_gate", "trust_pool", "trust_resample"):
    mX, mS, mU = per(f"mse_{X}"), per("mse_stiff2"), per("mse_unif")
    c1 = bool((1 - mX / mS).mean() >= 0.20 and ci(mS - mX)[1] > 0); c2 = bool(ci(mU - mX)[1] > 0); crit[X] = c1 and c2
    m, lo, hi = ci(mS - mX); row(f"D2(2) {X} cut vs stiff2 >= 0.20", 1 - mX / mS, c1, f"diff {m:.5f} [{lo:.5f}, {hi:.5f}]")
    m, lo, hi = ci(mU - mX); row(f"D2(2) unif - {X} > 0", mU - mX, c2, f"{m:.5f} [{lo:.5f}, {hi:.5f}]")
d2b = (crit["trust_gate"] or crit["trust_pool"]) and crit["trust_resample"]
print("== VERDICTS (own, from CSVs)  gate", gate, "D1(1)", d1a, "head", head, "D1", d1, "D2(1)", a1 and a2 and a3 and a4, "D2(2)", d2b, "D2", a1 and a2 and a3 and a4 and d2b)
for v in V: print(" | ".join(str(x) for x in v))
print("== CSV-derived minus JSON (max abs over seeds), by quantity")
diffs = sorted(((max(abs(out[s][k] - jsn[s][k]) for s in SEEDS), k) for k in jsn[101]), reverse=True)
for dd, k in diffs[:25]: print(f"{k}: {dd:.3g}")
print("n quantities compared:", len(diffs) * 3, " max:", diffs[0])
print("== pools rebuilt from d2_rollouts.csv vs d2_actions.csv q_* columns: max abs diff", max(maxdiff.values()), max(maxdiff, key=maxdiff.get))
for k in ("unif", "stiff2", "trust_pool", "trust_gate", "trust_resample"):
    print(k, [f"{maxdiff[(s, k)]:.2e}" for s in SEEDS])
print("== misc per seed")
for s in SEEDS:
    print(s, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[s].items() if k in ("within_n_skipped", "strata_sizes_match", "H_unique_csv", "n_pairs", "npos_nan_gap_K3", "npos_nan_gap_K3c", "calgap_K3", "calgap_K3c", "mean_signed_err_unif")})
print("== 3-seed means (mine) of descriptive quantities")
for k in ("mse_unif", "mse_stiff2", "mse_trust_gate", "mse_trust_pool", "mse_trust_resample", "mse_trust_pool_r5head", "mse_clean8", "mse_explore8", "mse_clean16",
          "mse_trust_trim12_clean16", "mse_trust_trim12_clean16_r5head", "mse_trust_pool_clean16", "neff_trust_pool", "neff_trust_gate", "neff_stiff2",
          "pruned_explore_share", "sp_stiff", "within_stiff", "sp_r5", "within_r5", "within_K3", "across_K3", "within_K3c", "across_K3c", "calgap_K3", "calgap_K3c",
          "routeb_cal", "p0_cal", "pool16_cal", "red_test", "mean_signed_err_unif",
          "tier_clean.kl", "tier_explore.kl", "tier_clean.sq_err", "tier_explore.sq_err", "tier_clean.terminal_like_H_lt_0.15", "tier_explore.terminal_like_H_lt_0.15",
          "tier_clean.flag_end", "tier_explore.flag_end", "tier_clean.count_violation", "tier_explore.count_violation", "tier_clean.cone_lp", "tier_explore.cone_lp",
          "tier_clean.ply_lp", "tier_explore.ply_lp", "tier_clean.beta", "tier_explore.beta",
          "jsd_attn_L3_K3c", "jsd_attn_L3_K3", "jsd_scaled_mix", "jsd_ataraxos_mmd", "kl_union", "kl_attn_L3_K3c", "kl_attn_L3_K3", "kl_scaled_mix", "kl_kc_dedup_K3c", "kl_kc_dedup_K3", "kl_mix", "kl_ataraxos_mmd"):
    v = per(k); m, lo, hi = ci(v); print(f"{k}: {', '.join(f'{x:.5f}' for x in v)} | {m:.5f} [{lo:.5f}, {hi:.5f}]")
```

### audit_e1 (check E (i))

```python
# audit_e1: independent D0 anchor KL and all-slots marginal window (route-B) posterior on 200 calibration D0 positions, seed 101.
# Own forward pass from the state dict (checked once against r5 model6.RealNet), own tapes, own window likelihoods.
import csv, math, time, sys, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0)
D = "/tmp/r5e2/data"; CK = "/tmp/r5e2/ckpt/s101/primary"; RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs/s101/eval"
sd = {k: v.float() for k, v in torch.load(f"{CK}/real_100.pt", map_location="cpu").items()}
NROOT, NTOK, d, NH = 45, 57, 128, 4
MASK = torch.zeros(NTOK, NTOK, dtype=torch.bool); MASK[:NROOT, :NROOT] = True
for i in range(NROOT, NTOK): MASK[i, :NROOT] = True; MASK[i, NROOT:i + 1] = True
E_ = lambda n, ix: sd[n + ".weight"][ix]
lin = lambda x, n: x @ sd[n + ".weight"].T + sd[n + ".bias"]
lnm = lambda x, n: F.layer_norm(x, (x.shape[-1],), sd[n + ".weight"], sd[n + ".bias"])
@torch.no_grad()
def fwd(tp, root_only=False):
    B = tp["turn"].shape[0]
    sq = E_("e_sqc", tp["sq_content"]) + E_("e_own", tp["sq_ownflag"]) + E_("e_sslot", tp["sq_slot"])
    sl = E_("e_loc", tp["slot_loc"]) + E_("e_sfl", tp["slot_flags"]) + E_("e_mc", tp["slot_mcount"]) + E_("e_rev", tp["slot_pubrev"]) + E_("e_val", tp["slot_val"])
    gl = E_("e_turn", tp["turn"].clamp(0, 60))[:, None]
    A = E_("e_actor", tp["ply_actor"]) + E_("e_from", tp["ply_from"]) + E_("e_to", tp["ply_to"])
    O = E_("e_out", tp["ply_out"]) + E_("e_att", tp["ply_att"]) + E_("e_def", tp["ply_def"])
    x = torch.cat([sq, sl, gl, torch.stack([A, O], 2).reshape(B, 12, d)], 1) + sd["pos"]
    n = NROOT if root_only else NTOK
    x = x[:, :n]; m = MASK[:n, :n]
    for l in range(4):
        p = f"blocks.{l}."
        q, k, v = lin(lnm(x, p + "ln1"), p + "qkv").view(B, n, 3, NH, d // NH).permute(2, 0, 3, 1, 4)
        a = torch.softmax((q @ k.transpose(-1, -2) / math.sqrt(d // NH)).masked_fill(~m, float("-inf")), -1)
        x = x + lin((a @ v).transpose(1, 2).reshape(B, n, d), p + "proj")
        x = x + lin(F.gelu(lin(lnm(x, p + "ln2"), p + "mlp.0")), p + "mlp.2")
    return lnm(x, "lnf")
def load(split):
    z = np.load(f"{D}/{split}.npz"); return {k: z[k] for k in z.files}
PK = ("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")
def tape(R, ix):
    t = {k: torch.from_numpy(R[k][ix].astype(np.int64)) for k in ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev")}
    t["slot_val"] = torch.from_numpy(np.where(R["slot_pubrev"][ix] > 0, R["slot_true"][ix], 0).astype(np.int64))
    t["turn"] = torch.from_numpy(R["turn"][ix].astype(np.int64))
    for k in PK: t[k] = torch.zeros(len(ix), 6, dtype=torch.long)
    return t
Rc = load("cal")
idx = {(int(g), int(t)): i for i, (g, t) in enumerate(zip(Rc["game"], Rc["t"]))}
o = np.load(f"{D}/oracle_d0_cal.npz"); N = 200
rows = np.array([idx[(int(g), int(t))] for g, t in zip(o["game"][:N], o["t"][:N])])
tt = Rc["t"][rows].astype(int); t0 = np.maximum(tt - 4, tt % 2)
rows0 = np.array([idx[(int(g), int(a))] for g, a in zip(Rc["game"][rows], t0)]); nw = tt - t0
sys.path.insert(0, "/home/agent/workspaces/mats/work/micro-stratego/r5"); import model6
net = model6.make_net("primary"); net.load_state_dict(torch.load(f"{CK}/real_100.pt", map_location="cpu")); net.eval()
tw = tape(Rc, rows0[:64]); pl = Rc["ply"][rows0[:64]].astype(np.int64)
for c, k in enumerate(PK): tw[k][:, :4] = torch.from_numpy(np.where(np.arange(4)[None] < nw[:64, None], pl[:, :4, c], 0))
with torch.no_grad(): h_ref, _ = net(tw)
print("forward self-check max|mine - model6| over all token states:", float((fwd(tw) - h_ref).abs().max()))
t1 = time.time(); _ = fwd(tape(Rc, np.resize(rows, 2048))); print("full fwd 2048 rows sec %.2f" % (time.time() - t1))
T_t = tape(Rc, rows); hid = ((T_t["slot_val"] == 0) & (T_t["slot_loc"] < 36)).numpy()
Tw = tape(Rc, rows0); pl = Rc["ply"][rows0].astype(np.int64)
live = np.arange(4)[None] < nw[:, None]
for c, k in enumerate(PK): Tw[k][:, :4] = torch.from_numpy(np.where(live, pl[:, :4, c], 0))
anchor = torch.log_softmax(lin(fwd(Tw, root_only=True)[:, 36:44], "h_belief"), -1).numpy().astype(np.float64)
opp = live & (pl[:, :4, 0] == 1); mv = pl[:, :4, 1] * 36 + pl[:, :4, 2]
LL = np.zeros((N, 8, 6))
jobs = [(b, s, k) for b in range(N) for s in range(8) if hid[b, s] and opp[b].any() for k in range(1, 7)]
J = np.array(jobs); print("window tapes:", len(J))
t1 = time.time()
for c0 in range(0, len(J), 2048):
    jb = J[c0:c0 + 2048]; b = jb[:, 0]
    tp = {k: v[b].clone() for k, v in Tw.items()}
    tp["slot_val"][np.arange(len(jb)), jb[:, 1]] = torch.from_numpy(jb[:, 2])
    h = fwd(tp)
    g = np.zeros(len(jb))
    for j in range(4):
        pos = 44 if j == 0 else NROOT + 2 * (j - 1) + 1
        lp = torch.log_softmax(lin(h[:, pos], "h_policy"), -1).numpy().astype(np.float64)
        g += lp[np.arange(len(jb)), mv[b, j]] * opp[b, j]
    LL[jb[:, 0], jb[:, 1], jb[:, 2] - 1] = g
print("window pass sec %.1f" % (time.time() - t1))
def lsm(x): return x - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)) - x.max(-1, keepdims=True)
post = lsm(anchor + LL)
msg = np.clip(np.where(hid[..., None], post - anchor, 0), -8, 8); post_c = lsm(anchor + msg)
P = o["pstar_full"][:N].astype(np.float64)
def kl(p, lq): return np.where(p > 0, p * (np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(np.exp(lq), 1e-9, 1))), 0).sum(-1)
f = lambda lq: np.where(hid, kl(P, lq), 0).sum(1) / np.maximum(hid.sum(1), 1)
ka, kb, kbc = f(anchor), f(post), f(post_c)
with open(f"{RUN}/d0_gate_cal.csv") as fh: rr = list(csv.DictReader(fh))[:N]
assert all(int(r["game"]) == int(g) and int(r["t"]) == int(t) for r, g, t in zip(rr, o["game"][:N], o["t"][:N]))
ca = np.array([float(r["kl_anchor"]) for r in rr]); cb = np.array([float(r["kl_routeb_all"]) for r in rr])
cn = np.array([float(r["n_hidden"]) for r in rr])
print("n_hidden match:", bool((cn == hid.sum(1)).all()))
print("kl_anchor   mine mean %.6f  csv mean %.6f  max|diff| %.2e  mean diff %.2e" % (ka.mean(), ca.mean(), np.abs(ka - ca).max(), (ka - ca).mean()))
print("kl_routeb   mine mean %.6f  csv mean %.6f  max|diff| %.2e  mean diff %.2e  (no clamp)" % (kb.mean(), cb.mean(), np.abs(kb - cb).max(), (kb - cb).mean()))
print("kl_routeb_c mine mean %.6f  csv mean %.6f  max|diff| %.2e  (msg clamped to [-8,8] as the run does)" % (kbc.mean(), cb.mean(), np.abs(kbc - cb).max()))
print("reduction route-B vs anchor on these 200: mine %.4f csv %.4f" % (1 - kb.mean() / ka.mean(), 1 - cb.mean() / ca.mean()))
```

### audit_e2 (check E (ii); audit_e2b is the same code with the exploratory branch removed, K = 16 and `manual_seed(777)`)

```python
# audit_e2: independent clean / exploratory rollout tiers (seed-101 checkpoint) on the (s, a) pairs of 48 random D2 test
# positions, 8 + 8 rollouts per pair; compared pair by pair with the executor's rollouts in d2_rollouts.csv and with d2.json.
import csv, json, math, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
D = "/tmp/r5e2/data"; CK = "/tmp/r5e2/ckpt/s101/primary"; RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs/s101/eval"
sd = {k: v.float() for k, v in torch.load(f"{CK}/real_100.pt", map_location="cpu").items()}
st_ = {k: v.float() for k, v in torch.load(f"{CK}/slot_trust.pt", map_location="cpu").items()}
NROOT, NTOK, d, NH, NS = 45, 57, 128, 4, 8
MASK = torch.zeros(NTOK, NTOK, dtype=torch.bool); MASK[:NROOT, :NROOT] = True
for i in range(NROOT, NTOK): MASK[i, :NROOT] = True; MASK[i, NROOT:i + 1] = True
E_ = lambda n, ix: sd[n + ".weight"][ix]
lin = lambda x, n, S=sd: x @ S[n + ".weight"].T + S[n + ".bias"]
lnm = lambda x, n: F.layer_norm(x, (x.shape[-1],), sd[n + ".weight"], sd[n + ".bias"])
@torch.no_grad()
def fwd(tp, root_only=False):
    B = tp["turn"].shape[0]
    sq = E_("e_sqc", tp["sq_content"]) + E_("e_own", tp["sq_ownflag"]) + E_("e_sslot", tp["sq_slot"])
    sl = E_("e_loc", tp["slot_loc"]) + E_("e_sfl", tp["slot_flags"]) + E_("e_mc", tp["slot_mcount"]) + E_("e_rev", tp["slot_pubrev"]) + E_("e_val", tp["slot_val"])
    gl = E_("e_turn", tp["turn"].clamp(0, 60))[:, None]
    A = E_("e_actor", tp["ply_actor"]) + E_("e_from", tp["ply_from"]) + E_("e_to", tp["ply_to"])
    O = E_("e_out", tp["ply_out"]) + E_("e_att", tp["ply_att"]) + E_("e_def", tp["ply_def"])
    x = torch.cat([sq, sl, gl, torch.stack([A, O], 2).reshape(B, 12, d)], 1) + sd["pos"]
    n = NROOT if root_only else NTOK
    x = x[:, :n]; m = MASK[:n, :n]
    for l in range(4):
        p = f"blocks.{l}."
        q, k, v = lin(lnm(x, p + "ln1"), p + "qkv").view(B, n, 3, NH, d // NH).permute(2, 0, 3, 1, 4)
        a = torch.softmax((q @ k.transpose(-1, -2) / math.sqrt(d // NH)).masked_fill(~m, float("-inf")), -1)
        x = x + lin((a @ v).transpose(1, 2).reshape(B, n, d), p + "proj")
        x = x + lin(F.gelu(lin(lnm(x, p + "ln2"), p + "mlp.0")), p + "mlp.2")
    return lnm(x, "lnf")
def strust(f):  # slot-trust MLP from slot_trust.pt: Linear-GELU-Linear-GELU-Linear, softplus
    return F.softplus(lin(F.gelu(lin(F.gelu(lin(f, "f.0", st_)), "f.2", st_)), "f.4", st_)).squeeze(-1)
PK = ("ply_actor", "ply_from", "ply_to", "ply_out", "ply_att", "ply_def")
def tape(R, ix):
    t = {k: torch.from_numpy(R[k][ix].astype(np.int64)) for k in ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev")}
    t["slot_val"] = torch.from_numpy(np.where(R["slot_pubrev"][ix] > 0, R["slot_true"][ix], 0).astype(np.int64))
    t["turn"] = torch.from_numpy(R["turn"][ix].astype(np.int64))
    for k in PK: t[k] = torch.zeros(len(ix), 6, dtype=torch.long)
    return t
def ent(p): p = p.clamp_min(1e-9); return -(p * p.log()).sum(-1)
COUNTS = torch.tensor([1, 2, 2, 1, 1, 1])
@torch.no_grad()
def rollouts(tp, fr, to, explore, gen):
    B = fr.shape[0]; ar = torch.arange(B)
    loc = tp["slot_loc"]; hidden = (tp["slot_val"] == 0) & (loc < 36)
    dto = torch.maximum((loc // 6 - to[:, None] // 6).abs(), (loc % 6 - to[:, None] % 6).abs())
    dfr = torch.maximum((loc // 6 - fr[:, None] // 6).abs(), (loc % 6 - fr[:, None] % 6).abs())
    key = torch.where(hidden, (dto * 64 + dfr * 8 + torch.arange(8)[None]).float(), torch.full((B, 8), 1e9))
    ks, ki = key.sort(1); cone = torch.where(ks[:, :3] < 1e9, ki[:, :3], torch.full_like(ki[:, :3], -1))
    incone = torch.zeros(B, 8, dtype=torch.bool)
    for c in range(3): incone[ar[cone[:, c] >= 0], cone[cone[:, c] >= 0, c]] = True
    ncone = incone.sum(1).clamp_min(1).float()
    tp = {k: v.clone() for k, v in tp.items()}
    step_of = torch.full((B, 8), -1, dtype=torch.long); lp_samp = torch.zeros(B)
    for stp in range(3):
        todo = incone & (tp["slot_val"] == 0)
        if not todo.any(): break
        h = fwd(tp, root_only=True); lg = lin(h[:, 36:44], "h_belief"); q = torch.softmax(lg, -1); H = ent(q)
        nrev = (tp["slot_val"] > 0).float().sum(1, keepdim=True).expand(-1, 8) / 8
        e = strust(torch.cat([h[:, 36:44], H[..., None], nrev[..., None]], -1))
        s = (H + e).masked_fill(~todo, float("inf")).argmin(1); act = todo.any(1)
        probs = torch.softmax(lg[ar, s] / 2.5, -1) if explore else q[ar, s]
        v = torch.multinomial(probs.clamp_min(1e-9), 1, generator=gen).squeeze(1)
        lp_samp[act] += torch.log(q[ar, s].gather(1, v[:, None]).squeeze(1).clamp_min(1e-9))[act]
        step_of[ar[act], s[act]] = stp
        sv = tp["slot_val"].clone(); sv[ar[act], s[act]] = v[act] + 1; tp["slot_val"] = sv
    if explore:
        samp = step_of >= 0
        r = torch.rand(B, 8, generator=gen).masked_fill(~samp, 9.0); o2 = r.argsort(1)[:, :2]
        ok = samp.sum(1) >= 2; sv = tp["slot_val"].clone()
        a_, b_ = o2[ok, 0], o2[ok, 1]; va, vb = sv[ar[ok], a_].clone(), sv[ar[ok], b_].clone()
        sv[ar[ok], a_] = vb; sv[ar[ok], b_] = va; tp["slot_val"] = sv
        cone_lp = torch.zeros(B); final = tp["slot_val"].clone()
        for stp in range(3):
            if not (step_of == stp).any(): break
            t2 = dict(tp); t2["slot_val"] = torch.where(samp & (step_of >= stp), torch.zeros_like(final), final)
            lq = torch.log_softmax(lin(fwd(t2, root_only=True)[:, 36:44], "h_belief"), -1)
            msk = step_of == stp; has = msk.any(1); sidx = msk.float().argmax(1)
            val = lq[ar, sidx].gather(1, (final[ar, sidx] - 1).clamp_min(0)[:, None]).squeeze(1)
            cone_lp[has] += val[has]
        cone_lp = cone_lp / ncone
    else:
        cone_lp = lp_samp / ncone
    uniq = (F.one_hot(tp["slot_val"].clamp(0, 6), 7)[..., 1:].sum(1) > COUNTS[None]).any(1)
    T = 2.0 if explore else 1.0
    done = torch.zeros(B, dtype=torch.bool); term = torch.zeros(B, 3); nply = torch.zeros(B, dtype=torch.long)
    lps, lpn, offs = torch.zeros(B), torch.zeros(B), torch.zeros(B); flag_end = torch.zeros(B, dtype=torch.bool)
    for j in range(4):
        if j == 0: f_, t_ = fr, to
        else:
            lgp = lin(fwd(tp)[:, NROOT + 2 * (j - 1) + 1], "h_policy")
            mov = torch.multinomial(torch.softmax(lgp / T, -1), 1, generator=gen).squeeze(1)
            lp = torch.log_softmax(lgp, -1).gather(1, mov[:, None]).squeeze(1); lv = (~done).float()
            lps += lp * lv; lpn += lv; offs += (lp.exp() < 1e-3).float() * lv
            f_, t_ = mov // 36, mov % 36
        live = ~done
        for k, vv in (("ply_actor", torch.full_like(f_, j % 2)), ("ply_from", f_), ("ply_to", t_)):
            x = tp[k].clone(); x[live, j] = vv[live]; tp[k] = x
        z = lin(fwd(tp)[:, NROOT + 2 * j], "h_trans")
        oc, at, de = (torch.multinomial(torch.softmax(z[:, a:b], -1), 1, generator=gen).squeeze(1) for a, b in ((0, 5), (5, 12), (12, 19)))
        for k, vv in (("ply_out", oc), ("ply_att", at), ("ply_def", de)):
            x = tp[k].clone(); x[live, j] = vv[live]; tp[k] = x
        nply = torch.where(live, torch.full_like(nply, j + 1), nply)
        flag = live & (oc == 4); tv = torch.zeros(B, 3); tv[:, 0 if j % 2 == 0 else 2] = 1
        term = torch.where(flag[:, None], tv, term); flag_end |= flag
        clock = live & ~flag & (tp["turn"] + j + 1 >= 60); tv = torch.zeros(B, 3); tv[:, 1] = 1
        term = torch.where(clock[:, None], tv, term); done = done | flag | clock
    hv = fwd(tp); q = torch.softmax(lin(hv[torch.arange(B), NROOT + 1 + 2 * (nply - 1)], "h_value"), -1)
    q = torch.where(done[:, None], term * 0.98 + 0.02 / 3, q)
    e = (q[:, 0] + 0.5 * q[:, 1]).clamp(1e-5, 1 - 1e-5)
    ply_lp = torch.where(lpn > 0, lps / lpn.clamp_min(1), torch.zeros(B))
    return dict(q=q, e=e, H=ent(q), cone_lp=cone_lp, ply_lp=ply_lp, flag=flag_end.float(), uniq=uniq.float(), offs=offs, moves=(nply - 1).clamp_min(0).float(), term=done.float())
z = np.load(f"{D}/test.npz"); R = {k: z[k] for k in z.files}
ix = {(int(g), int(t)): i for i, (g, t) in enumerate(zip(R["game"], R["t"]))}
o = np.load(f"{D}/oracle_q_test.npz"); nl = o["n_legal"].astype(int); off = np.r_[0, np.cumsum(nl)]
rows = np.array([ix[(int(g), int(t))] for g, t in zip(o["game"], o["t"])])
assert all((R["legal"][rows[i], :nl[i]] == o["canon"][i, :nl[i]]).all() for i in range(240))
pos = np.sort(np.random.default_rng(20261017).choice(240, 48, replace=False))
pairs = [(i, j) for i in pos for j in range(nl[i])]; P = len(pairs); pid = np.array([off[i] + j for i, j in pairs])
K = 8; prow = np.repeat(rows[[i for i, _ in pairs]], K)
act = np.repeat([j for _, j in pairs], K)
fr = torch.from_numpy(R["legal"][prow, act, 0].astype(np.int64)); to = torch.from_numpy(R["legal"][prow, act, 1].astype(np.int64))
Qs = np.stack([o["qstar"][i, j] for i, j in pairs]).astype(np.float64); Qe = Qs[:, 0] + 0.5 * Qs[:, 1]
gen = torch.Generator().manual_seed(4242)
res = {}
t1 = time.time()
for tier, ex in (("clean", False), ("explore", True)):
    outs = []
    for c0 in range(0, len(prow), 2048):
        sl = slice(c0, c0 + 2048)
        outs.append(rollouts(tape(R, prow[sl]), fr[sl], to[sl], ex, gen))
    res[tier] = {k: torch.cat([u[k] for u in outs]).numpy().astype(np.float64) for k in outs[0]}
    print(tier, "done sec %.0f" % (time.time() - t1), flush=True)
def kl(p, q): return np.where(p > 0, p * (np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(q, 1e-9, 1))), 0).sum(-1)
mine = {}
for tier in ("clean", "explore"):
    r = res[tier]; Qr = np.repeat(Qs, K, 0); Qer = np.repeat(Qe, K)
    mine[tier] = {"H<0.15": (r["H"] < 0.15), "kl": kl(Qr, r["q"]), "sq_err": (r["e"] - Qer) ** 2, "flag": r["flag"], "uniq": r["uniq"],
                  "cone_lp": r["cone_lp"], "ply_lp": r["ply_lp"], "offsup_per_rollout": r["offs"], "H": r["H"]}
    mine[tier]["_offsup_per_move"] = r["offs"].sum() / max(1, r["moves"].sum())
with open(f"{RUN}/d2_rollouts.csv") as fh: rr = list(csv.DictReader(fh))
cols = {k: np.array([float(x[k]) for x in rr]).reshape(-1, 24) for k in ("e", "H", "kl", "flag", "uniq", "offsup", "cone_lp", "ply_lp")}
with open(f"{RUN}/d2_actions.csv") as fh: aa = list(csv.DictReader(fh))
qe_run = np.array([float(x["qstar_eff"]) for x in aa])
assert np.abs(qe_run[pid] - Qe).max() < 1e-5
dj = json.load(open(f"{RUN}/d2.json"))
print(f"pairs {P} from {len(pos)} positions; rollouts per tier {P * K}")
for tier, sl, nm in (("clean", slice(0, 16), "tier_clean"), ("explore", slice(16, 24), "tier_explore")):
    th = {"H<0.15": cols["H"][pid, sl] < 0.15, "kl": cols["kl"][pid, sl], "sq_err": (cols["e"][pid, sl] - Qe[:, None]) ** 2, "flag": cols["flag"][pid, sl],
          "uniq": cols["uniq"][pid, sl], "cone_lp": cols["cone_lp"][pid, sl], "ply_lp": cols["ply_lp"][pid, sl], "offsup_per_rollout": cols["offsup"][pid, sl], "H": cols["H"][pid, sl]}
    jmap = {"H<0.15": "terminal_like_H_lt_0.15", "kl": "kl", "sq_err": "sq_err", "flag": "flag_end", "uniq": "count_violation", "cone_lp": "cone_lp", "ply_lp": "ply_lp", "H": "H"}
    print(f"-- {tier}: stat | mine (same pairs) | executor (same pairs) | diff [pair-clustered 95% CI] z | d2.json all pairs")
    for k in ("H<0.15", "kl", "sq_err", "flag", "uniq", "cone_lp", "ply_lp", "offsup_per_rollout", "H"):
        m_p = mine[tier][k].reshape(P, K).mean(1); t_p = th[k].mean(1); dd = m_p - t_p
        se = dd.std(ddof=1) / math.sqrt(P); zz = dd.mean() / se if se > 0 else float("nan")
        print(f"{k}: {m_p.mean():.4f} | {t_p.mean():.4f} | {dd.mean():+.4f} [{dd.mean() - 1.96 * se:+.4f}, {dd.mean() + 1.96 * se:+.4f}] z={zz:+.2f} | {dj[nm].get(jmap.get(k, ''), float('nan')):.4f}")
    print(f"offsup per move (mine, ratio of sums): {mine[tier]['_offsup_per_move']:.4f} | d2.json {dj[nm]['offsup_per_move']:.4f}")
print("total sec %.0f" % (time.time() - t1))
```

### audit_h (quoted descriptive numbers, hashes, R6 gate equality, Q* noise)

```python
import json, math, numpy as np, torch
M = "/home/agent/workspaces/mats/work/micro-stratego"; T = 4.302652729749464
def ci(v): v = np.asarray(v, float); m = v.mean(); h = T * v.std(ddof=1) / math.sqrt(3); return f"{m:.5f} [{m - h:.5f}, {m + h:.5f}]"
J = lambda s, p, r="r7": json.load(open(f"{M}/{r}/runs/s{s}/eval/{p}.json"))
S = (101, 102, 103)
tr = {s: J(s, "trust") for s in S}; d2 = {s: J(s, "d2") for s in S}
for k in ("target_mean_clean", "target_mean_explore", "train_beta_mean_clean", "train_beta_mean_explore", "train_spearman_beta_neg_target", "sel_beta_mean_clean", "sel_beta_mean_explore"):
    print(k, [round(tr[s][k], 4) for s in S], ci([tr[s][k] for s in S]))
for k in ("unif", "trust_pool", "trust_gate", "trust_resample", "stiff2", "clean8", "trust_pool_r5head"):
    print("logloss", k, ci([tr[s]["logloss"][k] for s in S]))
print("selected", {s: (tr[s]["alpha"], tr[s]["gate_q"], tr[s]["gate_tau"], tr[s]["gamma"]) for s in S})
print("pruned explore share (JSON)", [round(d2[s]["resample_pruned_explore_share"], 5) for s in S], ci([d2[s]["resample_pruned_explore_share"] for s in S]))
print("offsup per move clean/explore", ci([d2[s]["tier_clean"]["offsup_per_move"] for s in S]), ci([d2[s]["tier_explore"]["offsup_per_move"] for s in S]))
print("delusional counts:", {s: {k[len("delusional_count_"):]: v for k, v in d2[s].items() if k.startswith("delusional_count_") and v} for s in S})
print("sec per seed (done.json):", [round(J(s, "done")["sec"], 1) for s in S])
for s in S:
    meta = json.load(open(f"/tmp/r5e2/ckpt/s{s}/primary/train_meta.json"))
    print("param hash", s, J(s, "done")["param_hash"] == meta["param_hash_final"], J(s, "done")["unchanged"])
for s in S:
    a, b = J(s, "d0"), J(s, "d0", "r6")
    diffs = []
    def walk(x, y, p=""):
        if isinstance(x, dict):
            for k in x:
                if k == "sec": continue
                if k not in y: diffs.append((p + k, "missing in r6")); continue
                walk(x[k], y[k], p + k + ".")
        elif x != y: diffs.append((p, x, y))
    walk(a, b)
    print("d0 r7 vs r6 seed", s, "n_diff", len(diffs))
sd = torch.load("/tmp/r5e2/ckpt/s101/primary/real_100.pt", map_location="cpu"); print("theta_real params:", sum(v.numel() for v in sd.values()))
o = np.load("/tmp/r5e2/data/oracle_q_test.npz"); qs = [o["qstar"][i, j] for i in range(240) for j in range(int(o["n_legal"][i]))]
qs = np.array(qs, float); qe = qs[:, 0] + 0.5 * qs[:, 1]; var = qs[:, 0] + 0.25 * qs[:, 1] - qe ** 2
print("pairs", len(qs), "plug-in Q* noise var/64 mean %.5f" % (var / 64).mean())
r6 = [json.load(open(f"{M}/r6/runs/s{s}/eval/d1.json")) for s in S]; print("R6 K4 gap", ci([r["key_gap"] for r in r6]))
```

Output (10:23Z): target means clean 1.07673, exploratory 1.05693; log losses 0.62671 / 0.62528 / 0.62612 / 0.62798 /
0.69964 / 0.63441; pruned share 0.50356 [0.48144, 0.52569]; off-support per move 0.00424 / 0.12960; delusional only
stiff2 (seeds 102, 103: 1 each); 796.0 s per seed; hashes equal; r7 vs r6 d0.json 0 differences per seed; 1,001,644
parameters; Q* noise 0.00215; R6 K4 gap 0.22758 [0.21086, 0.24430].

### audit_h2 (trust vs terminal-like rollouts; C3)

```python
import csv, math, numpy as np
from scipy.stats import mannwhitneyu
RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs"; T = 4.302652729749464
def ci(v): v = np.asarray(v, float); m = v.mean(); h = T * v.std(ddof=1) / math.sqrt(3); return f"{m:.4f} [{m - h:.4f}, {m + h:.4f}]"
res = {k: [] for k in ("beta_term", "beta_other", "auc_other_gt_term", "within_pair_rank_term", "pruned_term_share", "base_term_share_in_batch", "clean8_minus_unif")}
for s in (101, 102, 103):
    with open(f"{RUN}/s{s}/eval/d2_rollouts.csv") as fh: r = list(csv.DictReader(fh))
    g = {k: np.array([float(x[k]) for x in r]).reshape(-1, 24) for k in ("H", "beta")}
    mix = np.r_[0:8, 16:24]; H, B = g["H"][:, mix], g["beta"][:, mix]; t = H < 0.15
    res["beta_term"].append(B[t].mean()); res["beta_other"].append(B[~t].mean())
    u = mannwhitneyu(B[~t], B[t]).statistic / (t.sum() * (~t).sum()); res["auc_other_gt_term"].append(u)
    rk = np.argsort(np.argsort(B, 1, kind="stable"), 1, kind="stable") / 15.0
    res["within_pair_rank_term"].append(rk[t].mean())
    b12 = np.c_[g["beta"][:, :6], g["beta"][:, 16:22]]; h12 = np.c_[g["H"][:, :6], g["H"][:, 16:22]]
    pr = np.argsort(b12, 1, kind="stable")[:, :4]; res["pruned_term_share"].append((np.take_along_axis(h12, pr, 1) < 0.15).mean())
    res["base_term_share_in_batch"].append((h12 < 0.15).mean())
    with open(f"{RUN}/s{s}/eval/d2_actions.csv") as fh: a = list(csv.DictReader(fh))
    q = np.array([float(x["qstar_eff"]) for x in a]); c8 = np.array([float(x["q_clean8"]) for x in a]); un = np.array([float(x["q_unif"]) for x in a])
    res["clean8_minus_unif"].append(((c8 - q) ** 2).mean() - ((un - q) ** 2).mean())
for k, v in res.items(): print(k, [round(x, 4) for x in v], ci(v))
```

Output: AUC 0.6676 [0.5913, 0.7438]; mean within-pair rank of terminal-like 0.3227 (0.5 = no signal); pruned
terminal-like share 0.2116 [0.1591, 0.2641] vs initial-batch share 0.1176 [0.1009, 0.1343]; clean8 - unif MSE 0.0018
[0.0013, 0.0023].

### audit_d2b (stiffness base log 2 vs log 3)

```python
import csv, math, numpy as np
RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs"; T = 4.302652729749464
def ci(v): v = np.asarray(v, float); m = v.mean(); h = T * v.std(ddof=1) / math.sqrt(3); return f"{m:.5f} [{m - h:.5f}, {m + h:.5f}]"
out = {k: [] for k in ("mse_stiff_log3", "mse_stiff_log2_noclamp", "mse_stiff_log2_clamp", "red_gate_vs_log2", "red_resample_vs_log2", "red_pool_vs_log2")}
for s in (101, 102, 103):
    with open(f"{RUN}/s{s}/eval/d2_rollouts.csv") as fh: r = list(csv.DictReader(fh))
    g = {k: np.array([float(x[k]) for x in r]).reshape(-1, 24)[:, np.r_[0:8, 16:24]] for k in ("e", "H")}
    with open(f"{RUN}/s{s}/eval/d2_actions.csv") as fh: a = list(csv.DictReader(fh))
    A = {k: np.array([float(x[k]) for x in a]) for k in ("qstar_eff", "q_trust_gate", "q_trust_pool", "q_trust_resample")}
    wm = lambda w: (w * g["e"]).sum(1) / np.maximum(w.sum(1), 1e-12)
    mse = lambda q: ((q - A["qstar_eff"]) ** 2).mean()
    m3 = mse(wm((1 - g["H"] / math.log(3)) ** 2)); m2 = mse(wm((1 - g["H"] / math.log(2)) ** 2)); m2c = mse(wm(np.clip(1 - g["H"] / math.log(2), 0, None) ** 2))
    out["mse_stiff_log3"].append(m3); out["mse_stiff_log2_noclamp"].append(m2); out["mse_stiff_log2_clamp"].append(m2c)
    for k, c in (("red_gate_vs_log2", "q_trust_gate"), ("red_resample_vs_log2", "q_trust_resample"), ("red_pool_vs_log2", "q_trust_pool")):
        out[k].append(1 - mse(A[c]) / min(m2, m2c))
for k, v in out.items(): print(k, [round(x, 5) for x in v], ci(v))
```

### audit_smoke_excl (D1 and D2 without the smoke-seen test positions)

```python
import csv, math, numpy as np
from scipy.stats import spearmanr
RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs"; T = 4.302652729749464
def ci(v): v = np.asarray(v, float); m = v.mean(); h = T * v.std(ddof=1) / math.sqrt(3); return m, m - h, m + h
def rd(s, f):
    with open(f"{RUN}/s{s}/eval/{f}") as fh: r = list(csv.DictReader(fh))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}
def sp(x, y): return float("nan") if np.std(x) == 0 or np.std(y) == 0 else float(spearmanr(x, y).correlation)
o = {k: [] for k in ("gap_K3", "gap_K3c", "jr_sm", "jd_sm", "jr_mmd", "jd_mmd", "kd_sm", "kd_mmd", "kd_kc", "st", "wt", "d_ov", "d_strata", "u_gate", "u_pool", "u_res", "r_gate", "r_res")}
for s in (101, 102, 103):
    c = rd(s, "d1_positions.csv"); k = slice(48, None)
    for h in ("K3", "K3c"): o[f"gap_{h}"].append(np.nanmean(c[f"key_gap_{h}"][k]))
    ja = c["jsd_attn_L3_K3c"][k].mean(); ka = c["kl_attn_L3_K3c"][k].mean()
    for X, t in (("scaled_mix", "sm"), ("ataraxos_mmd", "mmd")):
        jx = c[f"jsd_{X}"][k].mean(); o[f"jr_{t}"].append(jx / ja); o[f"jd_{t}"].append(jx - ja); o[f"kd_{t}"].append(c[f"kl_{X}"][k].mean() - ka)
    o["kd_kc"].append(c["kl_kc_dedup_K3c"][k].mean() - ka)
    r = rd(s, "d2_rollouts.csv"); A = rd(s, "d2_actions.csv"); P = len(A["pos"]); keep = A["pos"] >= 12
    g = {kk: r[kk].reshape(P, 24)[keep] for kk in ("e", "H", "beta", "kl")}
    mix = np.r_[0:8, 16:24]; E, H, B, KL = (g[kk][:, mix] for kk in ("e", "H", "beta", "kl"))
    y = -KL.ravel(); o["st"].append(sp(B.ravel(), y)); ss = sp(-H.ravel(), y); o["d_ov"].append(o["st"][-1] - ss)
    o["wt"].append(np.nanmean([sp(B[p], -KL[p]) for p in range(len(B))]))
    Hr = H.ravel(); n = Hr.size; st = np.empty(n, int); st[np.argsort(Hr, kind="stable")] = (np.arange(n) * 10) // n
    o["d_strata"].append(np.nanmean([sp(B.ravel()[st == q], y[st == q]) for q in range(10)]) - np.nanmean([sp(-Hr[st == q], y[st == q]) for q in range(10)]))
    q = A["qstar_eff"][keep]; mse = lambda col: ((A[col][keep] - q) ** 2).mean()
    mu, ms = mse("q_unif"), mse("q_stiff2")
    for X, t in (("q_trust_gate", "gate"), ("q_trust_pool", "pool"), ("q_trust_resample", "res")):
        o[f"u_{t}"].append(mu - mse(X))
    o["r_gate"].append(1 - mse("q_trust_gate") / ms); o["r_res"].append(1 - mse("q_trust_resample") / ms)
for kk, v in o.items():
    m, lo, hi = ci(v); print(f"{kk}: {m:.5f} [{lo:.5f}, {hi:.5f}]")
```

### prereg_hash (check A)

```python
import hashlib
P = "/home/agent/workspaces/mats/work/micro-stratego/r7/PREREG-R7-E3.md"
b = open(P, "rb").read()
new = (b"time is in `PREREG-R7-E3.sha256`). `outbox/experiments-round-7-amendments.md` (09:24Z) exists: it amends R7-E2 (readings 4-7) and R7-E1 (reading\n"
       b"9) only and contains nothing for R7-E3. Work\n")
old = b"time is in `PREREG-R7-E3.sha256`). No `outbox/experiments-round-7-amendments*.md` existed when this was written. Work\n"
assert b.count(new) == 1
print("reconstructed first version sha256:", hashlib.sha256(b.replace(new, old)).hexdigest())
S = open("/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md", "rb").read().split(b"\n")
spec = b"\n".join(S[138:147]) + b"\n"
sec1 = b.split(b"## 1. Decision rules, copied verbatim from section 2.2 of the spec\n\n")[1].split(b"\n\n## 2. Operational")[0] + b"\n"
print("prereg section 1 body == spec 139-147 byte for byte:", sec1 == spec, len(sec1), len(spec))
print("spec sha256:", hashlib.sha256(open("/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md", "rb").read()).hexdigest())
```

Output: 9824d81bfcd3691a44bcb30e4e6cdb850c1b89a72313984c7c715ef346aa62aa (= the recorded superseded hash); True 1892 1892;
cbece88d738601336e5a1fe70a71de7805cfd7087455f39dc83249a35aa24e88.

### gate_diff (the 3.6e-5 trust_gate case)

```python
import csv, json, numpy as np
RUN = "/home/agent/workspaces/mats/work/micro-stratego/r7/runs/s101/eval"
with open(f"{RUN}/d2_rollouts.csv") as fh: r = list(csv.DictReader(fh))
B = np.array([float(x["beta"]) for x in r]).reshape(-1, 24)[:, np.r_[0:8, 16:24]]
E = np.array([float(x["e"]) for x in r]).reshape(-1, 24)[:, np.r_[0:8, 16:24]]
with open(f"{RUN}/d2_actions.csv") as fh: a = list(csv.DictReader(fh))
qg = np.array([float(x["q_trust_gate"]) for x in a])
tr = json.load(open(f"{RUN}/trust.json")); q, tau = tr["gate_q"], tr["gate_tau"]
Qq = np.quantile(B, q, axis=1, keepdims=True); sd = B.std(1, ddof=1, keepdims=True)
w = 1 / (1 + np.exp(-(B - Qq) / (tau * sd + 1e-6))); mine = (w * E).sum(1) / w.sum(1)
d = np.abs(mine - qg); i = int(d.argmax())
print("worst pair", i, "diff %.2e" % d[i], "beta std %.3e" % sd[i, 0], "beta range %.6g..%.6g" % (B[i].min(), B[i].max()))
print("pairs with diff > 1e-5:", int((d > 1e-5).sum()))
```

Output: worst pair 1648, difference 3.56e-05, beta std 0.0307, betas 19.9032-19.9998; 2 pairs above 1e-5; trust_gate MSE
0.01488245 from the rebuilt pool, from d2_actions.csv and from d2.json.
