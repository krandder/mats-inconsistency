# AUDIT R5-E1 (E-INFO): independent audit

Auditor: board investigator, Opus 5.5 (claude-opus-5-5[1m]), read-only. Audit run 2026-10-17 05:26-05:55Z. I did not run the
experiment. Versions audited: `RESULTS-R5-E1.md` sha256 dfdfd3bf... (mtime 05:34:09Z; section 11 still `@@AUDIT@@`),
`PREREG.md` 50413cd9... (05:45:07Z; deviation 6 was appended during the audit), `PREREG-AMENDMENTS.md` 4ce6667d..., raw files
as hashed in `audit/rederive-output.txt` section 0. Copies of the two audited documents: `scratch/r5e1-audit/inputs/`.
File location: my role allows writes only under `/home/agent/workspaces/fable-strategy/scratch/`, so this file and
`audit/` live in `.../fable-strategy/scratch/r5e1-audit/`; the paths requested in the brief (`li-net/r5e1/AUDIT-R5-E1.md`,
`li-net/r5e1/audit/`) need a copy by the owner.

## Overall verdict

- **Results: reproduced.** My own code (`audit/rederive.py`; it imports neither run.py, partA2.py nor analyze.py) re-derives
  every headline statistic from the raw per-board files. Per seed against `summary.json`: largest absolute difference
  4.3e-6 (606 Part A comparisons) and 2.3e-6 on the Part B rule statistics. That is the 5-decimal rounding of the raw arrays.
  My own exact-Bayes enumeration matches the raw `pstar` to 5.0e-6.
- **Verdicts: identical.** Re-deciding all 10 primary rules and the 5 secondary ones from my numbers gives the same
  PASS/FAIL as `results/final/verdicts.md`: primary PASS D0.1, D0.2, D1.2, D3.2, D3.3, D3.4; FAIL D1.1, D2.1, D2.2, D3.1.
  **No rule decided differently.** One reading caveat: E1-D0.1's PASS rests on the amendment's redefinition of clause 1. Read
  literally (the spec's 0-step belief mu0, here the v2 root on note + 2 anchors), Pearson is 0.299 < 0.65 (see check 4).
- **Integrity: holds.** Registration preceded the first GPU line by 41 s and the amendment preceded the first v2 GPU line by
  9 s. All hashes match. PREREG.md differs from its registered copy only inside section 10. Every executed file in every
  launch carries a declared hash. Exact Bayes is never used in training or selection. The LIR mask checks are 0.0 in every
  seed. Cost is $2.62 of the $6 ceiling, and no r5e1 VM is alive (05:39:54Z).
- **Write-up: accept after wording changes.** Every number I checked matches the files, apart from minor rounding and
  source slips. Failures are reported as failures. The amendment, the registered Part A, the ladder and the exploratory
  analyses are labelled, and the write-up says that a v2 rollout is a local evidence read. Several interpretive sentences go
  beyond the numbers: the D1.1 "why it fails" attribution, the key "shared component caps the correlation" explanation, the
  "same belief" deduplication claim, the sycophancy-board cause, "every pool shrinks", the E1-D0.1 gate reading, and the TL;DR
  headline for Part A. Fixes are listed at the end. None of them changes a verdict.

**Paste-ready line for RESULTS section 11:** Independent audit (`AUDIT-R5-E1.md`): every headline number re-derived from the raw
files with independent code (largest per-seed difference 4.3e-6, i.e. rounding); all 10 primary and 5 secondary verdicts
reproduced; registration and amendment precede their GPU launches (41 s, 9 s); no exact-Bayes use in training or selection;
LIR mask checks 0.0 in every seed; cost $2.62 of $6, no VM alive at 05:39:54Z; 15 wording changes requested, none changing a verdict.

## Check 1. Timing and integrity: PASS

Run `audit/rederive.py`, output section 9 (05:51:59Z). Also `diff`, `stat` and `sha256sum` at 05:28-05:48Z.
- `PREREG.md` "Registered: 2026-10-17T03:11:33Z". The first `soup-gpu-r5e1-` line is GPU-LOG.md:1997, "2026-10-17T03:12Z
  PENDING-CREATE vm=soup-gpu-r5e1-smoke-1017031214-9343". The VM-name stamp and the launch log both say 03:12:14Z, so
  registration came 41 s earlier.
- `PREREG-AMENDMENTS.md` "Registered: 2026-10-17T03:36:58Z". The first `soup-gpu-r5e1-v2` line is GPU-LOG.md:2025, "03:37Z
  PENDING-CREATE vm=soup-gpu-r5e1-v2-1017033707-24514", stamp 03:37:07Z. The amendment came 9 s earlier. File mtimes agree:
  PREREG.sha256 03:11:33.736Z, AMENDMENTS 03:36:58.888Z.
- Hashes:
  - `PREREG.sha256` f41e89c8...a4fd = sha256(`results/PREREG-as-registered-0311Z.md`).
  - `PREREG-AMENDMENTS.sha256` 4ce6667d...5ce9 = sha256(`PREREG-AMENDMENTS.md`) = sha256(`results/PREREG-AMENDMENTS-as-registered.md`).
  - Spec 7958f340...b8f, as quoted.
  - The spec 1.3 excerpt (spec lines 73-90) hashes to ec9a56cb..., as quoted, and the PREREG section 8 verbatim block is
    byte-identical to it.
  - The amendments file is d04269d4...355f, as quoted.
- PREREG.md vs the registered copy: lines 1-226 and the final 3 lines (the Registered stamp) are identical. Only line 227
  "(none yet)" in "## 10. Deviations log" was replaced, by 37 lines (deviations 1-6). The entries sit before the trailing
  Registered line rather than at end of file, which is harmless.
- Launch blocks in `results/stage-manifests.txt`. There are 6 blocks and 4 GPU launches; I matched them by VM-name stamps and
  `/tmp/r5e1/logs/launch-*.log`.

  | block | launch | executed files vs declared |
  |---|---|---|
  | 03:03:36Z | none (pre-registration staging) | run.py 850c1ed8 (pre-registration development) |
  | 03:09:51Z | none (registration manifest) | all = PREREG hashes |
  | 03:12:14Z | r5e1-smoke | all = PREREG hashes |
  | 03:25:29Z | r5e1-main | run.py 0a6a897a (declared by deviations 2-3; hash given in PREREG-AMENDMENTS s4); rest = PREREG |
  | 03:37:07Z | r5e1-v2 (preempted) | partA2 474c7bbd, remote.sh 91513560, test 8818b15c, run.py 0a6a897a = AMENDMENTS s4; env6/wm/rollouts/netdef = PREREG |
  | 04:33:51Z | r5e1-v2b | executed files identical to v2 |

  Non-executed staged files changed: analyze.py edfbd867 -> 795d13e0, and site_patch.py was new. Deviation 5's "staged hashes
  unchanged" is literally true only for the executed files. Deviation 6 (05:45Z) now says so. `gpu-run.sh:78` syncs
  `GPU_SOURCE_DIR` (= /tmp/r5e1/src). Its log line "rsync .../logical-soup -> ..." prints the wrong directory, which is
  cosmetic.
- I could not diff the run.py change from the registered 4561c8ea to the launched 0a6a897a: no copy of 4561c8ea is preserved.
  Indirect check: the smoke ran the registered code and main ran the new code. Comparing their summary.json schemas (s101),
  the only differences are the declared ones: 4 `kl_*_UN_tau1` keys, `syc_false_note_bayes_rate` and
  `syc_false_note_unwarranted`, and `cfg.ladder` "100,250,500,1000". Every other cfg default is identical. The inserted
  tau = 1 bag shifts the Part A random stream used by the later controlled bags and test reference. That changes the random
  draw, not the computation.
- analyze.py: the verdict logic used for all verdicts is the 795d13e0 version. Its staged copy has mtime 03:40Z, before any
  result: main Part A raw is stamped 04:22Z (VM clock) and v2 05:24Z. The post-result edits (05:25Z, 05:32Z; now 8f917fbf)
  change plotting and CSV labels only, per `diff /tmp/r5e1/src/analyze.py analyze.py`. The verdict blocks in
  `results/final/results.json` are identical to those in the per-run `analysis/results.json`: 0 differences.
- Provenance of the v2 results: every file in `results/v2/s*` is from the v2b run (mtimes 04:34:49-05:24:25Z). The preempted
  v2 run synced nothing ("final sync failed", `/tmp/r5e1/logs/launch-r5e1-v2.log`; "remote rc=255",
  `/tmp/r5e1/logs/v2/driver.txt`).
- Test-split exposure before launches, a disclosure point rather than a violation:
  - The priced GPU smoke ran the registered design on the full 500-board test split. Its Part A test numbers are stamped
    03:23:25Z (VM clock; `/tmp/r5e1/logs/smoke/s101/log.txt:11`) and Part B 03:23:51-03:24:18Z.
  - run.py's last pre-launch edit was 03:23:57.6Z (stat). The 240 s sync would bring smoke numbers local at about 03:25Z or
    later.
  - Post-smoke changes are descriptive only (deviations 2-3). Deviation 3 uses the smoke's exact-Bayes monitor KL, which is
    disclosed.
  - Three CPU code-path smokes ran after registration and are not mentioned in PREREG s0 or the deviations log:
    `/tmp/r5e1/smoke-cpu3` (03:18-03:26Z), `/tmp/r5e1/smoke2-cpu0` (03:34Z) and `results/cpucheck` (03:38Z). All used d 32
    models, 120-150 steps and 40 test-stream boards.
- Unit tests: 10 passed at 05:43:28Z on hash-identical copies in scratch
  (`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider test_r5e1.py`).

## Check 2. No oracle in training or selection: PASS

`grep -nE 'bayes_marginals|Pstar|Pfull|Proot|Pstd6|mon_bayes|mon_b' run.py partA2.py rollouts.py wm.py env6.py`, then read in context.
- Exact Bayes is computed in exactly four places:
  - run.py:132, the monitor curve, used only at :144 ("descriptive curve only").
  - run.py:344, the test split. It feeds test metrics (:361-451), Part B evaluation `eval_b` (:484-529) and the Bayes ceiling
    (:627-629).
  - partA2.py:127, the monitor curve, used at :140.
  - partA2.py:355-357, the test split: Pfull, Proot and Pstd6, used at :365-484.
- Inside `part_b`, Pstar reaches only `eval_b` and the ceiling. The training loop (:556-609) uses the RL boards (stream
  10000+seed), their realised `occ` and sampled claims.
- Training targets:
  - theta_real: realised occupancy (run.py:116-119; partA2.py:150-152).
  - theta_trust: excess surprise against the realised board (run.py:236-237 `es_target(..., cal["occ"])`; partA2.py:251).
  - Aggregators: theta_real's own K = 128 reference. In run.py:221-229 it is the mean of sigmoid(lr) over 128 rollouts; in
    partA2.py:242-246 it is `union_call`, theta_real on the clean readings of the union of 128 reads.
  - eta and u: fitted on the same bags and reference (run.py:296-309; partA2.py:298-311).
  - Part B: realised rewards plus the truth loss on fresh torch boards (run.py:556-591).
- No selection anywhere. All loops have fixed length and final checkpoints are used. The Part B evals at steps 100-300 and
  `mon_loss` (held-out calibration bag BCE to the reference) are logged only. All hyperparameters are fixed in `Cfg`.
- Streams:
  - Test boards come from numpy `default_rng([2, 5501])` and `([22, 7702])`.
  - Calibration 1/21, validation 3/23, monitor 4/24 and RL 10000+seed all have distinct seeds.
  - Training and sampling use torch generators and manual seeds `seed`, `seed*1000+{11,21,31,41,51,61,71,211,221,231,271}`
    and 777.
  - Empirical check (rederive section 1): 0 identical full observations (occupancy, evidence, note) between test(2) and cal(1),
    val(3), mon(4) or RL(10101, 10102, 10103), and between test(22) and cal(21), val(23) or mon(24). Shared occupancy
    patterns occur only at chance level (3-66).
  - The raw test boards equal the regenerated streams 2 and 22 (occ, note, note_true, anchors).
- Notes, none of them a violation:
  - Deviation 3 used the smoke's exact-Bayes monitor curve to place the descriptive ladder snapshots. It is disclosed, affects
    no rule, and contradicts only PREREG s2's "never used for anything".
  - v2 theta_trust receives the read kind (log tau), a registered design flag.
  - The v2 aggregator target, the union call on clean readings, contains evidence that some bag reads lack. It is registered
    and is theta_real's own output.

## Check 3. Re-derivation from raw files: PASS

Commands: `audit/rederive.py`, output in `audit/rederive-output.txt` (05:51:59Z). Per-seed numbers are in
`audit/rederive-numbers.json`.
- Exact Bayes (own enumeration of 49,504 patterns, equal to env6's in the same order; own likelihood). Against the raw
  `pstar`: max 5.02e-6 for main and 5.03e-6 for v2. Against `pstar_root`: 5.02e-6.
- Part A, all 3 seeds x 2 runs: KL(P*||mu0), KL of all 4 pools on the UN, NW and SE bags, JSD of all 4 pools, key Pearson,
  same-patch and across-sector cosines, cosine by overlap, and the controlled-bag W(N) and MSE per N and tau. That is 606
  comparisons with `summary.json`.
  - Largest absolute difference 4.33e-6 (main s102 `key_cos_same_patch`, 0.47837479 vs 0.47837046).
  - Largest relative difference 8.3e-5 (main s103 `jsd_mmd_secells`, value 4.4e-6).
- Selected 3-seed means (mine):

  | run | KL P0 | KL mix UN | JSD mix | JSD attn | key r | same-across | W(8)/W(1) t1.5 | W(16)/W(1) t0.3 |
  |---|---|---|---|---|---|---|---|---|
  | v2 | 0.27967 | 0.20462 | 0.003309 | 0.022520 | 0.3749 | 0.2177 | 1.9139 | 1.0728 |
  | main | 8.05e-5 | 1.445e-3 | 7.03e-4 | 2.04e-5 | 0.1564 | 0.2596 | 3.9725 | 1.7840 |

- Integrity checks inside the raw files:
  - Every patch lies in its stated half (0 violations).
  - Controlled-bag backgrounds never overlap A* (max overlap 0 cells).
  - Key norms are 1 within 1.1e-4.
  - Realised NW shares are 0.79-0.80 / 0.20 / 0.50.
- Part B from `partB_raw.json` plus the main `partA_raw.json`:
  - LIR drift L-inf: 0 / 0 / 0.
  - Token-Masked mean drift: 0.05123 / 0.05150 / 0.05547.
  - False-note sycophancy: Fused 0.8287 / 0.8343 / 0.8287; LIR 0.0552 / 0.0525 / 0.0525.
  - LIR faith flip 1 / 1 / 1. Same-cell: Fused 0 / 0.008 / 0.004; LIR 0 / 0 / 0.
  - Fused Delta_desire change: 0.4906 / 0.4870 / 0.4346.
  - Bayes ceiling 0.830 in every seed.
  - Largest difference on a rule statistic 2.3e-6 (Token-Masked literal L1 per board); on any Part B statistic 1.5e-5
    (Fused `kl_bayes`, descriptive).
- Gap ladder: KL and JSD from the `partA_ladder_s*_raw.json` files match `summary.part_a_ladder` to 3.4e-8.
- Against `analysis/results.json` (3-seed means and CI bounds): largest difference 2.9e-4, on a CI bound of the secondary
  W(8)/W(1) (5.0898 vs 5.0901), from t = 4.303 vs 4.30265 plus rounding. On the primary rule statistics the largest is 6.9e-5
  (lower CI bound of JSD(attn)/JSD(mix), 5.78852 vs 5.78859); on means it is 2.7e-6.

## Check 4. Rules re-decided from my numbers: same as `results/final/verdicts.md`

Rules: PREREG s8 as amended by PREREG-AMENDMENTS s3; 3-seed means; seed-t, t = 4.303. Source: rederive section 6.

| rule | auditor | final | decisive auditor numbers |
|---|---|---|---|
| E1-D0.1 (v2) | PASS | PASS | clause 1 (amended; summary.json only) 0.9994 [0.9992, 0.9997]; cut mix UN 0.2683 [0.2644, 0.2723] |
| E1-D0.2 (main) | PASS | PASS | gain Fused 0.8763 [0.7752, 0.9773], LIR 0.9024 [0.8791, 0.9258]; same-cell 0.0040 / 0.0000 |
| E1-D1.1 (v2) | FAIL | FAIL | JSD ratio 6.8072 [5.7885, 7.8258]; mix - attn -0.0192 [-0.0223, -0.0162] |
| E1-D1.2 (v2) | PASS | PASS | cut vs P0 0.7405 [0.7374, 0.7436]; paired CIs vs mix, mmd, kc all > 0 (lowest lower bound 0.0387) |
| E1-D2.1 (v2) | FAIL | FAIL | Pearson 0.3749 (bar 0.55); gap 0.2177 (bar 0.30) |
| E1-D2.2 (v2) | FAIL | FAIL | only W(8)/W(1) at tau 1.5 = 1.9139 < 2 fails (per seed 1.847 / 1.905 / 1.990); the other 5 clauses hold |
| E1-D3.1 (main) | FAIL | FAIL | LIR syc 0.0534 > 0.03; drift 0 in every seed; R(LIR)/R(Fused) 1.011 |
| E1-D3.2 (main) | PASS | PASS | dd change 0.4707 [0.3928, 0.5486]; Fused syc 0.8306 |
| E1-D3.3 (main) | PASS | PASS | 0.0527 [0.0468, 0.0586] (literal L1 1.898 passes too) |
| E1-D3.4 (main) | PASS | PASS | flip 1.000 in every seed; swap belief change 0.0 (summary) |

Secondary, registered Part A: D0.1 FAIL (cut -17.27), D1.1 PASS (gate failed: descriptive; 0.0290), D1.2 FAIL, D2.1 FAIL,
D2.2 FAIL (the cuts vs kc and vs mmd fail). All agree with the final file.

Robustness:
- No verdict is within rounding of its bar.
- Readings that do not change any registered verdict:
  - E1-D0.1 literal clause 1: v2 root Pearson 0.2994 gives FAIL, which would make E1-D1 and E1-D2 descriptive. The amendment
    fixed the clause before the v2 launch, so the registered verdict is PASS.
  - E1-D3.1 with the descriptive unwarranted-agreement metric: 0.000, which would pass. The registered metric governs.

## Check 5. Mask checks: PASS

`summary.json` `part_b.lir.mask_check` = {max_reward_grad_real 0.0, max_param_change_real 0.0} in s101, s102 and s103.
- The LIR per-step `grad_real` is 0.0 at all 400 steps in every seed.
- The measurement is live: Token-Masked `grad_real` is 0.033-8.45 and Fused's is at least 0.685.
- The raw LIR pre/post beliefs are identical, and the swap belief change is 0.0.
- The zero is structural (`h.detach()`, `l.detach()`, run.py:563-564). The check confirms the mask; it does not test a
  learned property.

## Check 6. Cost: PASS

rederive section 10; `ledger.py` print mode; gcloud at 05:39:54Z.

| VM | GPU-LOG line | age_h x $0.87 | cost line |
|---|---|---|---|
| r5e1-smoke | 2014 DELETE | 0.32 h, $0.2784 | $0.28 |
| r5e1-main | 2058 DELETE | 1.43 h, $1.2441 | $1.24 |
| r5e1-v2 (preempted) | 2050 COST-AUDIT | 0.42 h, $0.3654 | $0.37 |
| r5e1-v2b | 2063 DELETE | 0.84 h, $0.7308 | $0.73 |

- Total: $2.62 by cost lines and $2.6187 by rate x age, against the $6 ceiling.
- Spans from the launch stamp to the end time (preemption at 04:02:50Z for v2) are 0.329, 1.442, 0.429 and 0.853 h. That is
  a $2.66 upper bound at minute resolution.
- `gcloud --project=GCP_PROJECT compute instances list --filter='name~r5e1' --format='value(name,status)'` returned empty
  output with rc 0 at 05:39:54Z. At 05:40:05Z there were no `soup-gpu` instances, and 14 other instances were visible, so the
  listing works.
- Note: GPU-LOG.md:2037 says "RUN-END ... rc=0" for the preempted v2 VM, because `driver.sh` always exits 0 (the remote rc was
  255). The COST-AUDIT line and deviation 5 record the preemption correctly.

## Check 7. RESULTS-R5-E1.md read critically

Method: I checked numbers by hand against the files, plus `audit/check7.py` (`audit/check7-output.txt`), which recomputes the
exploratory and descriptive numbers from the raw arrays. My coverage split equals `explore_coverage.json` to 4.6e-15, and my
centred-key statistics equal those cited. I did not use any automatic text classification, so the 30-item hand-label rule
does not apply.

Coverage: every number in sections 1, 3, 4, 4.1, 5, 6, 7, 7.1, 8 and 10, and the numbers in 2, 9 and 12. Not checked:
- R4-E1's "0.99 vs 0.99" (no source path).
- The mail ids and times.
- The owner's own 05:27Z gcloud result (my 05:39:54Z check agrees).

What is right:
- Numbers match the files, except the slips listed in change 9.
- The TL;DR table equals `results/final/verdicts.md`.
- "6 of 10 pass" is correct.
- Failures are called failures, including "fails as written" for D2.2 and D3.1.
- The header and section 2 say that a v2 rollout is a local evidence read and not a hypothetical sample.
- The amendment, the registered Part A (secondary, descriptive), the ladder (descriptive) and sections 4.1 and 5 (exploratory)
  are labelled, with one exception in the TL;DR (change 1).
- The secondary D1.1 "pass" is correctly called uninformative (shrinkage).
- The sycophancy section reports the registered metric as the decider.

What goes beyond the numbers is listed below.

## Requested wording changes (quote -> fix; evidence)

1. TL;DR: "**Why D1.1 fails: the 24% of cells only one search read.** There the policy dependence is in the evidence: ... With
   mix as the yardstick, the rule penalises using the evidence at all."
   -> "**Why D1.1 fails (exploratory, post hoc; section 4.1).** About half of attn_L3's JSD (49%) is on the 24% of cells
   only one search read. There the evidence itself differs: the union call is at 0.044 and attn_L3 at 0.046. The other half
   (50%) is on cells both searches read. There attn_L3's JSD is still 4.2x mix's and 3.6x the union call's, so the rule would
   fail on shared cells alone. mix's low JSD comes with the worst KL of any pool (0.207). In this design the 1/3-of-mix bar
   is met by ignoring the evidence (P0 has JSD 0) and missed even by the clean-evidence union call (4.1x mix)."
   Evidence (check7 A): attn share of JSD on shared cells 0.504 and on one-bag cells 0.486; JSD attn/mix on shared cells 4.19
   (3.93 / 4.19 / 4.45); attn/union 3.64 (ratio of means); union/mix overall 0.01364 / 0.00331 = 4.12.
2. TL;DR: "On cells both searches read, attn_L3's merged belief does not depend on how unequally the two searches repeated
   them; every other pool's does (exploratory, section 4.1)."
   -> "On cells both searches read, attn_L3's NW-vs-SE disagreement does not grow with the read-count imbalance (JSD
   0.0172 / 0.0158 / 0.0150 for count differences 0 / 1-2 / >= 3). mix's grows 5.5x, and ataraxos_mmd's and kc_dedup's
   3.5x (means of per-seed ratios). attn_L3's level there is still 3.6x the union call's (exploratory, cross-sectional bins; section 4.1)."
3. Section 12: "It deduplicates: identical reads add no weight, and cells read many times by one search and once by the other
   get the same belief."
   -> "It deduplicates: 16 identical clean reads get 1.07x [0.96, 1.19] the weight of one. On cells both searches read, its
   disagreement does not grow with the read-count imbalance (exploratory), but the two beliefs still differ there (JSD 0.015,
   3.6x the union call's)."
4. Section 5: "Disjoint reads still have cosine 0.71, which caps the correlation."
   -> "Disjoint reads still have cosine 0.71. This shared component compresses the same-minus-across gap (0.218). It hardly
   affects the Pearson: removing it (the exploratory centring below) raises the gap to 0.845 but the Pearson only from 0.375
   to 0.406. What limits the Pearson is the shape: cosine is flat for 0-2 shared cells (0.71, 0.68, 0.73) and rises steeply
   only at 4-6."
   Section 12: "The keys separate the same patch from other patches but carry a large shared component, so their correlation
   with overlap (0.375) misses the bar."
   -> "The keys separate the same patch (and 4-cell overlaps) from other patches but are not graded at small overlaps. Their
   correlation with overlap is 0.375 (0.406 after centring, exploratory) and misses the bar."
5. Section 8: "On 5.52% of those (20 boards), exact Bayes itself puts P(ship at the note cell) >= 0.5, because a probe there
   reported a hit."
   -> "... >= 0.5. On 12 of those 20 boards a probe at the note cell reported a hit. On the other 8 there is no probe at the
   note cell, and the other probes plus the note itself push P* above 0.5."
   Also, "LIR's 0.053 equals that rate" -> "LIR's 0.053 (0.0552 / 0.0525 / 0.0525) is at or just below that rate (0.0552)".
   The TL;DR "at exact Bayes's own rate" -> "at or just below". Evidence: check7 C (evidence regenerated from stream 2: hit 12,
   miss 0, none 8).
6. Gate reading, TL;DR E1-D0.1 row and section 3: add "Clause 1 is the amended one (PREREG-AMENDMENTS s3): the one-pass belief
   on 6 standard readings + note against Bayes on the same evidence. Read literally (spec 1.3: the pre-RL 0-step belief mu0,
   here the note + 2 anchors root, against full-board Bayes), the Pearson is 0.299 < 0.65. The gate would then fail and
   E1-D1 and E1-D2 would be descriptive. The amendment was registered before the v2 launch, so the registered verdict is
   PASS. The clause-1 statistic is in summary.json only; the raw files do not keep the 6-reading beliefs."
7. TL;DR: "every pool learns to shrink toward P0" -> "every fitted pool learns to stay near P0 (mix has no parameters and is
   18x worse than P0)". Section 7: "All of them learned to shrink toward P0." -> "The three fitted pools learned to stay near
   P0." Evidence: main eta 0.0016-0.0017, u 0.024; KL on preference bags: mmd 9.0e-5, kc 1.6e-4, attn 2.2e-4, mix 1.5e-3
   (mix 1.4e-3 on the unskewed bag, 18x P0's 8.0e-5).
8. Section 7.1: "From step 500 on, sampling noise outweighs any gap." -> "From step 500 on, mix is 27-29% worse than P0.
   attn_L3's +2.7% at step 1,000 is not consistent across seeds (+2.7 / -0.7 / +6.0%)." TL;DR: "helped only at training step
   250 (KL -7 to -8%)" -> "helped consistently only at step 250 (KL -7 to -8%, every seed)".
9. Rounding and source slips:
   - Section 4 table, union call KL "0.0335" -> "0.0334" (results.json `kl_union_pref` 0.033449).
   - Section 5 table, tau 1.5 W(2) "0.818" -> "0.817" (`controlled_bags.csv` 0.81745).
   - Section 10:
     - "Part B 4-23 s per arm" -> "4-25 s" (truth-only 25.2-25.4 s).
     - "the whole run 4,614-4,618 s (... results.json, compute)" -> "4,611-4,615 s" (`compute.total_sec`; 4,614-4,618 s is
       `rc.json` wall time).
     - "each ladder point about 6.5 min" -> "about 7 min" (fit + test 415-421 s).
   - Section 4: "about twice the weight" -> "2.3x (U_NW bags) to 2.5-2.8x (U_SE bags)".
10. Section 2: "trained jointly, and eta and u fitted, on 4,000 calibration boards" -> "on the 3,200 fit boards of the
    4,000-board calibration split (800 held out, logged only)" (run.py:232; partA2.py:248).
11. Section 9: "Nothing was retuned after any test number existed." Keep it, and add: "The priced GPU smoke (full 500-board
    test split, 1,000-step model; Part A numbers at 03:23Z VM time) and three CPU code-path smokes after registration
    (`/tmp/r5e1/smoke-cpu3` 03:18Z, `smoke2-cpu0` 03:34Z, `results/cpucheck` 03:38Z; d 32 models, 40 test boards, numbers
    meaningless) produced test-split numbers before the main and v2 launches. After them only descriptive metrics and ladder
    points changed (deviations 2-3); deviation 3 placed the descriptive ladder using the smoke's exact-Bayes monitor KL."
12. TL;DR Part A headline: "attn_L3 is the best pool by far and deduplicates as the spec predicts" -> "1 of 4 Part A rules
    passes (E1-D1.2). attn_L3 is the most accurate pool and passes the deduplication clauses, but fails policy independence
    (D1.1), key overlap (D2.1) and the mutual-support clause W(8)/W(1) >= 2 (D2.2)."
13. Scope the Token-Masked claim. TL;DR and section 8: "So spillover through shared weights is desire projection, not random
    noise." -> "Here, in all 3 seeds, Token-Masked's drift is in the goal-favoured direction: +0.14 to +0.17 on the 4
    most-likely-ship cells and -0.04 to -0.05 on the 4 least-likely, with Delta_desire utility component +0.100 [0.081,
    0.119]. It is not symmetric noise (descriptive, no rule)." The direction itself is supported (check7 D).
14. Section 6, E1-D3.4: add the pre-RL baseline. "Before RL, the same untrained head already flips on 0.644 of boards
    (0.000 / 0.982 / 0.950); RL brings it to 1.000 in every seed." Source: results.json `lir_faith_flip_rate_pre`. Without it
    the reader cannot tell how much of the 0.85 bar the head's u input gives for free.
15. Minor:
    - Section 5: "attn_L3's error falls with N (0.0398 to 0.0347), unlike kc_dedup's" -> "attn_L3's error falls from N = 1 to
      8 (0.0398 to 0.0347) and stays there (0.0350 at 16); kc_dedup's falls only to N = 4 (0.0411), then rises to 0.0484".
    - Section 3: "the clean-evidence ceiling for that bag" -> "the clean-evidence reference for that bag" (theta_real is not
      exact; KL 0.018).
    - Section 10 ledger: footnote that the v2 "deleted 04:33Z" is the COST-AUDIT time; GCE preempted and deleted the VM at
      04:02:50Z.
    - Section 3: "training reward rose from 0.50 to 0.79-0.82" -> "training env reward (25-step mean) rose from 0.49-0.51 to
      0.79-0.83 (Fused's rater reward, with the note bonus: 1.23-1.25)".
    - Section 5: give a source path for R4-E1's "0.99 vs 0.99".
    - Section 12: "E1-D3.1's 0.03 bar is below the Bayes rate of its own metric" -> add "on this test split (0.055)".

## What I could not check

- E1-D0.1 clause 1 (`clause1_corr_std6`, 0.9994): the model's 6-reading beliefs are not in the raw files. I computed exact
  Bayes on that evidence but used the summary.json value for the model's side.
- Expected eval env reward under pi_act (R_pre, R_post): the policy probabilities are not saved, so I used summary.json.
  The greedy-cell rewards and the Bayes ceiling, which I re-derived, match.
- The swap belief change: only one belief pass is saved. It is 0 by construction, since the belief has no U input.
- The registered run.py (4561c8ea) source: no copy is preserved. I checked it only through the output schemas.
- The preempted v2 run's logs, overwritten by v2b. Deviation 5's "about step 30,000" is unverifiable.
- Whether anyone read the smoke VM directly (ssh) before 03:23:57Z.
- The mails (#3688, #3692, #3693, #3696), and the R4-E1 comparison.

## Files (absolute)

- This audit: `/home/agent/workspaces/fable-strategy/scratch/r5e1-audit/AUDIT-R5-E1.md`
- Re-derivation: `/home/agent/workspaces/fable-strategy/scratch/r5e1-audit/audit/rederive.py`, output
  `.../audit/rederive-output.txt`, per-seed numbers `.../audit/rederive-numbers.json`
- Check-7 helper: `.../audit/check7.py`, output `.../audit/check7-output.txt`
- Audited copies: `.../inputs/RESULTS-R5-E1.audited-dfdfd3bf.md`, `.../inputs/PREREG.at-audit-50413cd9.md`
- Re-run: `cd .../audit && OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 rederive.py > rederive-output.txt && python3 check7.py > check7-output.txt` (about 10 s each)
