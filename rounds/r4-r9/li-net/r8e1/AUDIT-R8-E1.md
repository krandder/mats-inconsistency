# AUDIT R8-E1 (E-INFO-ANYTIME): independent read-only audit

Auditor: board investigator subagent (Opus 5.5, claude-opus-5-5[1m]); did not run the experiment. Written 2026-10-17T17:27Z.
Location: my role may write only under `/home/agent/workspaces/fable-strategy/scratch/`, so this file and `audit/` sit in
`/home/agent/workspaces/fable-strategy/scratch/r8e1-audit/`, not in `li-net/r8e1/` (the brief's fallback).
Audited versions: `RESULTS-R8-E1.md` sha256 d4029373...a575a (mtime 17:05:54Z; unchanged at 17:24:14Z); `PREREG-R8-E1.md` sha256
51dc94c6...924f (6 deviations, mtime 17:01:38Z). Scripts and outputs: `audit/a1_*` ... `audit/a8_*` (paths below are relative to
`li-net/r8e1/` unless absolute).

## Overall verdict: PASS with requested changes

- Registration preceded launch; every post-registration change is hashed and logged; no oracle in training or selection; no
  world-model training; PUCT never reads unread cells' real readings; the anchor token is exact at K = 0.
- Every headline number re-derives from raw.json with my own code. The verdicts in `analysis/verdicts.md` are reproduced part
  by part: E1-D0 PASS, E1-D1.1 PASS, E1-D1.2 FAIL, E1-D2.1 FAIL, E1-D2.2 FAIL. I would not decide any rule differently.
- Spend: $0.62 of the $0.80 ceiling. The scored VM's delete was issued 5 s after the run ended.
- RESULTS: 246 hand-mapped numbers match their files at printed precision. The record still needs corrections: the deviation-3
  story is wrong in two places (run 1 did produce test-board numbers; the retry did not crash). Some labels are wrong ("one-step"
  for looped_lin_T4), and a few mechanism and "never/cannot" sentences need a scope or an interpretation label. None of this
  changes a verdict.
- Largest numeric discrepancy: 3.45e-4. s103 ataraxos_mmd sparse-root KL at K = 32 is 0.17601 from raw.json vs 0.17567 in
  summary.json (4.35e-4 on the upper CI bound of its 3-seed mean). It is a descriptive pool, caused by 5-decimal rounding in
  raw.json (4.2% of mmd's K = 32 probabilities round to exactly 0 or 1). Over every decisive E1-D1 input the largest difference is
  1.3e-6.

## Check 1. Timing and integrity: PASS (deviation 3's narrative is inaccurate; see changes 1-3)

Evidence: `audit/a1_integrity.py` -> `a1_integrity.json` (run 17:16:42Z); `audit/a1_run1_vs_rerun.txt`.
- Registration: `PREREG-R8-E1.sha256` and `PREREG-R8-E1-AMENDMENTS.sha256` mtime 16:07:15.915Z, both "registered 16:07:15Z".
  - The first soup-gpu-r8e1- line of `logical-soup/GPU-LOG.md` (line 2127) is "16:07Z PENDING-CREATE" (minute resolution).
  - Launch log `/tmp/r8e1/logs/launch-r8e1-main.log` line 1: 16:07:22Z, 7 s after registration. The GCE insert of that VM
    started at 16:07:37Z (`audit/a7_gcloud.txt`).
- Hashes: `results/PREREG-R8-E1-as-registered.md` = 12034c21... = the .sha256. `results/PREREG-R8-E1-AMENDMENTS-as-registered.md`
  = current AMENDMENTS = 369b2194...
- PREREG vs as-registered: a single difflib hunk, inside section 5. The "(none yet)" placeholder (registered line 253) became
  the 6 deviation entries; the last line "Registered: 2026-10-17T16:07:15Z" is unchanged. Nothing above section 5 differs.
- Staged hashes (`results/stage-manifests.txt`, 4 blocks):
  - 16:07:22Z (first launch) equals the 16:07:15Z registration block in all 20 entries.
  - 16:35:21Z (refused try, rc 3, no VM) and 16:37:15Z (scored rerun) differ in exactly e1v5.py (deviation 3), ledger8.py and
    launch8.sh (deviation 4). They add numbers8.py, render_page8.py and site_patch8.py (deviation 2; staged, not run by
    remote8.sh). All imports and the 3 checkpoints are unchanged.
- e1v5.py: removing current lines 785-791 (the deviation-3 try/except that calls `deregister_op_overrides()` and logs) gives
  exactly the registered hash ace2bc1e. So nothing else changed. It is a 7-line block with one functional call, not literally
  "one line".
- After the scored run: analyze8.py changed (deviation 5; `diff` shows only the plot label, line 212). numbers8.py changed
  (deviation 6, 17:01:38Z). Every other file still equals the scored run's stage (`current_vs_scored_run_stage`).
- The rerun is the identical configuration: the cfg in `results/main-run1/s101/summary.json` and `results/main/s101/summary.json`
  has 0 differing keys. In all 3 seeds, the first 12 log lines of run 1 and of the rerun are identical once timestamps and
  elapsed seconds are removed (12 of 12). Those lines cover calibration, the scalar fits, the held-out BCEs, the root KLs, the
  Pearson and the K = 0 identity.
- Amendments folded in before launch: the outbox file mtime is 16:01:43Z (spec 15:42:28Z, sha256 b19a33e0 as registered). Its
  first 45 lines hash to d7b193a7... and are verbatim in the registered PREREG (section 2b). Spec 1.2 lines 48-62 hash to
  a9a8c597... and are verbatim in section 2a. At 17:16Z no other `experiments-round-8-amendments*` file exists.
- Inaccurate record (not a rule breach: no choice changed and the rerun reproduced these values exactly):
  - Run 1 did compute and log test-board numbers before crashing: root KL by M_0, the clause-1 Pearson and the K = 0 identity
    (E1-D0's inputs), at 16:24:22-35Z (`/tmp/r8e1/logs/main-run1/s*/log.txt` lines 11-12).
  - remote8.sh's retry did not crash. Each seed's `stdout-retry.txt` ends at 16:31:40-48Z with "trained anytime attn_L3-K3" and
    contains no traceback. The run was stopped at about 16:31:56Z (driver "remote rc=255", launcher rc=130).
  - Deviation 3 names `/tmp/r8e1/logs/main/` for run 1's partial outputs; they are in `/tmp/r8e1/logs/main-run1/` and
    `results/main-run1/`. `/tmp/r8e1/logs/main/` still holds run 1's three `stdout-retry.txt` files (byte-identical to
    main-run1's) and run 1's first 5 lines of `driver.txt`.

## Check 2. No oracle in training or selection: PASS

Evidence: code reading of e1v5.py and every function it calls; `audit/a2_unit_tests.txt`; `audit/a2_puct_noleak.py` ->
`a2_puct_noleak.json`; `audit/a4b_oracle_root.json`.
- Exact Bayes: `env6.bayes_marginals` is called only in `test8` (e1v5.py 632-633). It feeds KL, the clause-1 Pearson (the gate)
  and the D3 MSE. In the imported modules it occurs only in test2/test6/test7/test_part_a and train_world_model(2), none of
  which e1v5 calls (grep).
- Targets:
  - The anytime pools train with BCE to `REFS[Mi, b]` = ref(M_0) (lines 302, 356, 369, 380). M_0 = 2 uses the standard
    reference; 6/12/18 use `union_call` with root M_0 on the same 128 reference patches.
  - The R7-style pools train with BCE to `ref` (line 421). All 6 scalar fits use BCE to `ref` (std_sampler, d3_sampler).
  - theta_trust is trained by MSE to `RO.es_target`: real excess surprise against calibration occupancy (stream 21 only).
  - The only optimizers are over trust, pool + key-head parameters, and the scalar log-params (lines 326, 374, 415, 453).
- Frozen world model: parameters have `requires_grad_(False)` and `load_wm_npz` puts the model in eval mode. My CPU replay of
  the checkpoints, read in place with their sha256 re-checked, reproduces raw.json's `root_mu` for all 4 root strengths within
  4.0e-4 (GPU TF32 vs CPU FP32), the root KLs within 1.5e-6 and the clause-1 Pearson within 1.5e-7.
- PUCT: candidate keys come from `E3.prospect`, which imputes readings as mu >= 0.5. The variance term uses mu = the attn_L3-K3
  output on reads already taken. Both are checked three ways:
  - test_r8e1.py: 13 passed at 17:12:50Z (CPU, no caches written).
  - My own test: random unread readings redrawn before each wave, 3 RNG seeds, 10 boards. Picks were unchanged in all 4
    cells, and `prospect` output was bitwise unchanged when every non-root reading was redrawn.
  - Positive control: redrawing the read cells too changed later waves' picks on 1-3 boards per cell, so the test can detect
    a leak.
- Amended items, implemented as stated:
  - Interleaved stream (`fs_stream`). In raw.json for all 3 seeds: tile slots are shared by U_NW and U_SE; slots 0-20 hold the
    6-tile partition; slots 24 and 28 repeat its first 2 tiles; the first 24 reads cover all 36 cells; 89.8-90.9% of the
    skewed reads fall in the preferred half.
  - ref(M_0) target as above.
  - KS_TRAIN = (0, 1, 2, 4, 8, 16, 24, 32), pool_any = 32 and `sample_bags(nb, pa, g_, dev, 32, 32)`.
  - `crowding_prec` matches c_r = sum_b v_b (beta_b / beta_clean) relu(<k_r,k_b>)^2 within 1.2e-7, anchor included.
  - Strong-root tau: raw.json sr18 tau values are {0.3, 1.5} at 49.8-50.2%; sr18t3 values are {0.3, 1.5, 2.5} at 32.9-33.9% each.
- Code nit (no change requested; the registered code is frozen): the comments at e1v5.py lines 350 and 515 and the module
  docstring (48 reads) predate the amendment. The code itself is correct.

## Check 3. The anchor token: PASS

Evidence: `audit/a6_params_coef.json` (`anchor_token`), `audit/a4_rederive.txt`, `audit/a4b_oracle_root.json`.
- At K = 0 both anytime pools (LoopedAttnT3P, AttnL3P) return l0 bitwise, at M_0 = 2, 6, 12, 18, with random output heads.
  `summary.json` `identity_K0` is 0.0 for every pool, root and seed. In raw.json the K0 arrays of all saved pools equal
  `root_mu` exactly (M_0 = 2 and 18, both tau modes, 3 seeds).
- The anchor's Delta l is exactly 0. With 9 reads, out - l0 - sum_{r>=1} w_r Delta l_r <= 7.6e-6 (float32).
- beta_0 / beta_clean = M_0 / 6 (0.333, 1, 2, 3). `fit.beta_clean` equals `beta_by_tau["0.3"]` in every seed
  (3.3355 / 3.3989 / 3.5157). k_0 = normalize(W_K s_0) exactly.
- Anchor cells: my own re-implementation of make_boards2 (stream 22, n = 500) reproduces raw.json's `occ` and `anchor_order18`
  exactly in all 3 seeds. order[:, :2] are make_boards2's anchors by construction, and e1v5 asserts this at run time.
  anchor_order18 is identical across seeds.

## Check 4. Re-derivation of the headline numbers from raw.json: PASS

Evidence: `audit/a4_rederive.py` (numpy only; no e1v5/e1v4/e1v3/analyze8/wm/env6 import) -> `a4_rederive.txt/.json`;
`audit/a4b_oracle_root.py` -> `a4b_oracle_root.json`.
- 1,980 comparisons with summary.json (per seed) and analysis/results.json (3-seed mean, lo, hi). Largest |difference| by family:
  - sparse-root KL: looped_attn and attn_K3 2.2e-7; scaled_mix 4.2e-5; mmd 3.45e-4 (K = 32, rounding as above). JSD 3.5e-7.
  - M_0 = 18 curves, looped_attn/attn_K3/mmd/mix by K: decisive tau mode 1.7e-5 (mmd K24; mmd K16 1.3e-6); three-tier 9.7e-7.
  - Root KL by M_0: 2.2e-7. D3 W(N) of jacobi_prec, econ_D3, econ_R7, kc_K3, looped_attn: 5.1e-7; W ratios on results.json
    1.9e-6. D3 MSE of the saved pools (jacobi_prec, econ_D3, kc_K3, attn_K3): 2.6e-8.
  - 2x2 PUCT coverage from the saved patch lists: 1.2e-7. Wave-1 preferred share: 3.8e-8. attn_K3 KL/JSD: 7.2e-8.
  - Re-derived values: KL(32) = 0.0303 [0.0287, 0.0319], JSD(24) = 0.00575; M_0 = 18 KL(0) = 0.1444, mmd KL(16) - KL(0) = -0.0665.
    jacobi_prec W(16)/W(1) at tau 0.3 = 1.164 and at 2.5 = 2.816; econ_D3 = 1.417 / 2.455. PUCT coverage 0.9200 / 0.9580 /
    0.9651 / 0.9876.
- The evaluation oracle is correct: my own placement enumeration (49,504) and exact Bayes reproduce raw.json `pstar` within
  5.03e-6 (5-decimal rounding). The CPU replay gives clause-1 Pearson 0.999421 / 0.999483 / 0.999462 against summary.json's
  0.9994209 / 0.9994828 / 0.9994622.
- Not re-derivable from raw.json (not saved): P-FullSupport KL of jacobi_prec, looped_lin_T4 and kc_K1 (E1-D2.1's KL clauses),
  and the M_0 = 6 and 12 curves. Those come from summary.json only.
- The published `results.json`, `verdicts.md` and 4 CSVs are byte-identical to what both the current (33b8daad) and the
  registered (d4d1714f) analyze8.py produce from copies of the summaries (`audit/a5b_reanalyze.txt`, 17:15:56Z; copies deleted).

## Check 5. Rules re-decided from my numbers (PREREG section 3): all agree

Evidence: `audit/a5_rules.py` -> `a5_rules.txt/.json` (seed-t quantile from the closed form for 2 df, 4.302653).

| rule | auditor | verdicts.md | failing parts (auditor) |
|---|---|---|---|
| E1-D0 | PASS | PASS | none: sha256 3/3; Pearson 0.99946 [0.99938, 0.99953]; K = 0 diff 0.0 |
| E1-D1.1 | PASS | PASS | none: largest step -0.00029 (s103, K 24->32); KL(32) 0.0303; JSD(24) 0.00575 |
| E1-D1.2 | FAIL | FAIL | KL(0) 0.1444 > 0.050; mmd KL(16)-KL(0) = -0.0665 [-0.0700, -0.0630] (needs >= +0.015) |
| E1-D2.1 | FAIL | FAIL | P-FullSupport KL 0.0414 [0.0396, 0.0432] > 0.040 (summary.json input) |
| E1-D2.2 | FAIL | FAIL | econ_D3 W(16)/W(1) at tau 0.3 = 1.417 [1.395, 1.439] > 1.25 |

- Part-by-part flags are identical to analyze8's for all 5 rules. E1-D1 and E1-D2 are FAIL in both.
- I would not decide any rule differently. E1-D1.2's FAIL is mechanical and correct. Its two failing clauses (root strength;
  naive degradation) show that the approved reading 1 (M_0 = 18 anchor cells, KL 0.144) did not produce the strong root the
  spec assumed (0.035-0.050), which mail #3738 had flagged. The anchor-pool clauses pass.

## Check 6. Parameter counts and jacobi_lin_T4_prec's W: PASS

Evidence: `audit/a6_params_coef.py` -> `a6_params_coef.json` (CPU; imports e1v5 as the object under test).
- Counts by my closed-form arithmetic, by instantiating the code's classes, and as logged in the run (`fit.params`):
  looped_attn_T3 39,428; attn_L3-K3 106,180; attn_L3-K1 107,652; looped_lin_T4 2,789; jacobi_lin_T4 812; jacobi_lin_T4_prec
  814. The last is 222 (36 lambda~ + 4 eta + 144 P + 36 s + a + b) plus the K3 head Linear(36, 16) = 592. All three sources agree.
- W exactness on 6 random bags with random, non-init parameters (K 1-24, random masks, |Delta l| < 8):
  - `coef` vs the forward map: 5.7e-6 (float32). `coef` vs the autograd Jacobian diagonal: 4.8e-7. The off-diagonal (cross-cell)
    Jacobian is exactly 0.
  - The code's forward vs my float64 implementation of the PREREG formula (lambda_rc = softplus(lambda~_c) sigmoid(a - b
    beta_r) in the residual and in D): 4.4e-6. `lam_rc` vs the formula: 4.8e-7.
  - Identical reads: W(N) = N beta / (lambda(beta) + N beta) within 3.0e-7. lambda = 1 at init for beta 0.5, 2 and 4.
- Caveat: the trained pools were not saved, so W was checked on random parameter values. The trained D3 W values were
  re-aggregated from raw.json's per-board W arrays (check 4).

## Check 7. Cost and teardown: PASS

Evidence: `audit/a7_cost.py` -> `a7_cost.json`; `audit/a7_gcloud.txt` (read-only gcloud, 17:15:08-10Z).
- GPU-LOG DELETE lines: r8e1-main $0.36 (age 0.41 h) + r8e1-main2 $0.26 (0.30 h) = $0.62 <= $0.80 ($0.18 headroom).
- At the $0.87/h ceiling with GCE times from insert start to delete end (16:07:37-16:32:49Z; 16:37:24-16:56:25Z) the total is
  $0.64, still within the ceiling. The spot VM's real price is lower. Billing is not visible (Billing API disabled per GPU-LOG).
- Teardown of the scored run: `/tmp/r8e1/logs/main/driver.txt` records "remote rc=0" at 16:55:25Z, after the final sync. The
  delete was issued at 16:55:30.2Z (+5.2 s) and was done at 16:56:24.9Z (+59.9 s).
- Run 1: the crash came at 16:24:27-40Z. The registered automatic retry then kept the VM busy until about 16:31:56Z. The delete
  was issued at 16:32:00Z, 7.3 min and about $0.12 (ceiling) after the crash. That was within protocol (remote8.sh retries once),
  but it was the avoidable part of run 1's cost.
- No r8e1 VM alive: `gcloud --project=GCP_PROJECT compute instances list --filter='name~r8e1' --format='value(name,status)'`
  returned nothing at 17:05:51Z and at 17:15:08Z. One unrelated soup-gpu-r8e2 VM was RUNNING at 17:05:51Z.
- Group r8 (context): deleted-VM costs so far are r8e1 $0.62 + r8e3 $0.46 = $1.08 of $2.20; R8-E2's VM was still running.

## Check 8. RESULTS-R8-E1.md: numbers correct; wording changes requested

Evidence: `audit/a8_results_numbers.py` -> `a8_results_numbers.txt/.json`; `audit/a8_hand_labels.json`.
- Hand-mapped structured checks: 246 numbers, each tied to its results.json key. They cover every cell of the sparse-root table,
  the strong-root table, the D3 jacobi table and the PUCT table, plus the TL;DR and section 3, 5 and 6 values. 0 mismatches at
  printed precision.
- Automatic screen: every decimal token (448) was looked up among all values of results.json, summary.json, numbers.txt and
  GPU-LOG, plus registered constants.
  - Hand-labelled random sample first (40 tokens, seed 20261017): 40/40 correct or constant. Agreement with the screen is 40/40.
  - The screen is weak evidence: in 4 of the 40, its first hit was a coincidental key.
  - Screen totals: 441 matched a data value, 4 matched a constant, 3 were unmatched. All 3 are correct by hand: "5.5" is the
    model name; 0.0228 and 0.00029 are sign-flipped values verified above.
- Also verified by hand:
  - PUCT contrasts and interaction (numbers.txt lines 187-191, and my own per-seed coverage).
  - Fitted (u, lambda) means 1.21/1.13 and 1.26/1.41; jacobi_prec a, b and the per-tier lambda; compute 665-673 s and 106-109 s.
  - Strong-root curve shapes: mmd and scaled_mix improve through K = 16 and rise at K = 24 on every strong root, three-tier
    included. The anchor pools are monotone through K = 24.
  - JSD at K = 1 is 0 for all 13 pools. "R7's jacobi_lin_T4 ... 0.0405": 0.04053. "0.0009 higher": 0.00086.
- Failures are reported as failures. E1-D1.2, E1-D2.1 and E1-D2.2 are FAIL in the TL;DR and in sections 4-6.
- Problems: two factual errors carried over from deviation 3, one mislabel, one unscoped number, and several mechanism or
  absolute sentences without a label or scope. See the requested changes.

## Requested changes (quote, then fix)

PREREG section 5 is append-only. Fix the PREREG items by appending deviation 7 ("correction to deviation 3"), not by editing
entry 3.

1. PREREG deviation 3: "**the first run crashed before any test number**". Also RESULTS section 7: "The first run's partial
   outputs (calibration only, no test number) are kept in `results/main-run1/`."
   Fix: "The first run crashed after the E1-D0 values and the root KL by M_0 had been computed and logged on the test boards
   (`/tmp/r8e1/logs/main-run1/s*/log.txt` lines 11-12, 16:24:22-35Z), before any pool was scored. Those values fed no choice, and
   the rerun reproduced them exactly. `results/main-run1/` holds only the start-up summaries."
2. PREREG deviation 3: "remote8.sh's one retry crashed the same way. I stopped the run at 16:32Z". Also RESULTS section 8: "the
   run crashed in all 3 seeds at the first K = 1 test prefix, and remote8.sh's retry crashed the same way."
   Fix: "remote8.sh's automatic retry restarted all 3 seeds at 16:24:30Z. I stopped the run at about 16:31:56Z, while the retries
   were still calibrating (their logs end at 'trained anytime attn_L3-K3', no traceback). They ran the same code and would have
   reached the same path."
3. PREREG deviation 3: "The partial outputs (calibration logs only, no test number) stay in `/tmp/r8e1/logs/main/`."
   Fix: "Run 1's text logs are in `/tmp/r8e1/logs/main-run1/` and its partial summaries in `results/main-run1/`.
   `/tmp/r8e1/logs/main/s*/stdout-retry.txt` and the first 5 lines of `/tmp/r8e1/logs/main/driver.txt` are leftovers from run 1;
   the scored run did not retry (rc.json retry 0)." Add the same note to RESULTS section 12 ("Text logs: `/tmp/r8e1/logs/main/`
   and `/tmp/r8e1/logs/main-run1/`, temporary.").
4. RESULTS section 1: "The naive one-step pools are not anytime-safe." Also section 11: "One-step pools are not anytime-safe:
   ataraxos_mmd, scaled_mix and R6's looped linear pool turn back up as skewed reads accumulate".
   Fix: looped_lin_T4 is a 4-step looped pool, not one-step. Write "The pools without the anchor token (ataraxos_mmd, scaled_mix,
   R6's 4-step looped_lin_T4) are not anytime-safe on these streams". Drop "as skewed reads accumulate", or label it
   "(interpretation, not tested)" (see 5).
5. RESULTS section 4: "ataraxos_mmd rises after K = 16, and scaled_mix and looped_lin_T4 after K = 24, as skewed duplicate reads
   pile up;" This sits under a bullet marked "(descriptive)", but it states an untested mechanism.
   Fix: end the clause at "after K = 24", or add "(interpretation: after K = 24 the stream adds only repeated tiles and skewed
   reads; no unskewed control was run)".
6. RESULTS section 1: "At M_0 = 18 every K >= 1 was below K = 0 by at least 0.0228."
   Fix: "At M_0 = 18 with tau 0.3/1.5 reads, every K >= 1 was below K = 0 by at least 0.0228 in every seed (three-tier reads:
   at least 0.0156; numbers.txt lines 114-119)."
7. RESULTS section 1: "The anchor-aware attention pools never ended above their root in these tests, at any K, root strength or
   seed." Also section 11: "The anchor-aware attention pools never ended above their root in any measured setting".
   Fix: scope the claim. "In mean test KL over the 500 boards, neither anchor pool ended above its root at any tested K, on any
   of the five anytime curves (sparse root; M_0 = 6, 12, 18; three-tier), in any seed (numbers.txt lines 105-134)." "Any measured
   setting" overreaches: D3 and PUCT made no root comparison, and no per-board check was made.
8. RESULTS TL;DR: "E1-D1.2 fails on its two premise clauses" and "**FAIL** (premise clauses)".
   Fix: name the clauses. "FAIL on its root-strength clause (KL(0) <= 0.050) and its naive-degradation clause (mmd KL(16) - KL(0)
   >= +0.015); the anchor-pool clauses pass." The mmd clause is the spec's prediction, so calling both "premise" is an
   interpretation. Keep that word only with an "(interpretation)" label.
9. RESULTS section 1 (bold): "The precision-aware ridge does what it was designed for." Also section 5: "So clean reads get
   almost no ridge and noisy reads a large one".
   Fix: "The precision-aware ridge passes all four weight-law clauses of E1-D2.1; the rule fails on KL (0.0414 vs 0.040)." Then
   "clean reads get a small ridge (0.35, about a sixth of the tau-2.5 ridge)". W(16)/W(1) at tau 0.3 is 1.164, not about 1.
10. RESULTS section 11: "A one-step econ pool cannot get clean deduplication from a mixed-tier fit alone." Section 1 labels this
    "(interpretation)"; section 11 states it as fact.
    Fix: keep the label, or give the algebra. With one shared lambda and N identical reads, W(16)/W(1) = 16(lambda + beta) /
    (lambda + 16 beta). The tau-0.3 bar (<= 1.25 at beta about 3.4) needs lambda <= 0.92; the tau-2.5 bar (>= 2.2 at beta about
    1.5) needs lambda >= 2.09. This ignores background reads.
11. RESULTS section 4: "Gemini's gloss of '3 full 6-tile partitions' read in one call would have given about the R7 union call's
    KL (0.002)". The number has no path.
    Fix: cite `li-net/r7e1/RESULTS-R7-E1.md` (union call 0.0022, line 147), or summary.json
    `theta_real_train_curve_last.monitor_kl_full` (0.00228 / 0.00223 / 0.00241).
12. RESULTS section 1: "kc_dedup with raw keys ends up worse than the root after one read (0.3523 vs 0.2797)."
    Fix: "is worse than the root at K = 1 (0.3523 vs 0.2797) and back at the root's level at K = 2 (0.2791)". "Ends up" suggests
    a final state; its K = 32 value is 0.0940.
13. PREREG deviation 3 and RESULTS sections 1 and 8: "one-line backend fix" / "a one-line fix".
    Fix: "a 7-line block (one call, `deregister_op_overrides()`, inside try/except with a log line); removing it restores the
    registered e1v5.py hash exactly". This is minor and is evidenced in `audit/a1_integrity.json`.
14. RESULTS line 3: "(15:44-17:xxZ)". Fix: fill in the end time. Section 10 "@@AUDIT@@": point it to this audit's location
    above.

## What I could not check

- Agent-mail items (#3734, #3738, #3741, #3743, #3746): not read. I sent no message to anyone.
- Execution on the VM. I verified the staged source hashes and that the logs show the deviation-3 line. I did not see the
  copy on the VM itself.
- The 16:57Z first scoring outputs were overwritten at 16:58:13Z. I verified that both analyze8 versions reproduce the
  published outputs instead.
- Billed (not ceiling) cost.
- Trained pool weights were not saved. W exactness was checked on random parameters.
- jacobi_prec, looped_lin_T4 and kc_K1 sparse-stream outputs, and the M_0 = 6 and 12 curves, are not in raw.json. Their numbers
  are summary.json values.
- "R6-E1's 0.9494 used K4 keys": checked only against `e1v3.puct_search` and RESULTS-R6-E1.md line 29.
