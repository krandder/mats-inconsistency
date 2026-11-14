# AUDIT R6-E1 (E-INFO-V3)

Independent audit, 2026-10-17 07:50-08:25Z, by a Board investigator subagent (Claude Opus 5.5, model id claude-opus-5-5[1m]). I
did not run the experiment. Read-only on it: no GPU, no edits to code, results or write-up, no messages. My role can only write
under `/home/agent/workspaces/fable-strategy/scratch/`, so this file and `audit/` are at
`/home/agent/workspaces/fable-strategy/scratch/r6e1-audit/`, for the owner to copy into `li-net/r6e1/`. Paths below are
relative to `li-net/r6e1/` unless absolute; `audit/...` means the copied audit folder. RESULTS line numbers refer to
`RESULTS-R6-E1.md` sha256 7259317b... (mtime 07:50:42Z).

## Overall verdict

**Sound, publishable after corrections. All five verdicts confirmed from my own numbers:** E1-D0 PASS, E1-D1 PASS, E1-D2.1
PASS, E1-D2.2 FAIL (coverage 0.94938 < 0.95), E1-D3 FAIL (MSE cut vs kc_dedup at tau 2.5 0.103 < 0.15). I would decide no
rule differently, with t = 4.30265 or 4.303.

- Registration (06:27:53Z) precedes the first launch (06:28:06Z). Every registered file was staged with its registered hash;
  launch 2 also staged two unregistered publishing scripts that the VM never runs. The oracle is used only for scoring. PUCT never reads unread cells' real readings: shown by the code, the owner's unit test and my own
  flip test.
- **Largest numeric discrepancy: 5.8e-5** (P-FullSupport KL of ataraxos_mmd, seed 101: mine 0.1192627 vs summary.json
  0.1192047). It lies inside the interval allowed by the 5-decimal rounding of the saved beliefs ([0.1190005, 0.1194256]), and
  all 63 recomputable KL statistics lie inside their rounding intervals. Apart from that KL (3.8e-5 to 5.8e-5 across the
  three seeds), every seed-level value agrees to within 1.42e-5. JSD, coverage, key, W(N) and MSE values agree to within
  2.5e-6.
- Cost: $1.10 of $2.50. No r6e1 VM alive (gcloud, 07:54:53Z and 08:19Z).
- To fix before publication:
  1. One post-registration code change is not logged: `analyze6.py` was edited at 07:48:57Z. I verified it is plot-only and
     changes no number. The only copy of the registered script is in `/tmp`.
  2. 14 numbers in RESULTS are wrong at their printed precision. All are rounding-level and none affects a verdict.
  3. Several sentences claim more than the numbers show. The most important is section 10's "Key-space PUCT reaches 95%
     coverage", which contradicts the reported FAIL.

## Check 1: timing and integrity: PASS, with one unlogged change

Evidence: `audit/integrity.sh` -> `audit/integrity-output.txt` (run 08:19:14Z).

- **Registration before spend.** `PREREG-R6-E1.sha256` reads "registered 2026-10-17T06:27:53Z"; its mtime is 06:27:53.80Z. The
  first launch came 13 s later:

  | evidence | time (UTC) |
  |---|---|
  | `/tmp/r6e1/logs/launch-r6e1-main.log` (LAUNCH) | 06:28:06Z |
  | VM name `soup-gpu-r6e1-main-1017062806-8653` | 06:28:06 |
  | GCE insert operation | 06:28:17Z |
  | first `soup-gpu-r6e1-` GPU-LOG line (`logical-soup/GPU-LOG.md` line 2064, minute stamp) | 06:28Z |

  No r6e1 GPU-LOG line exists before 06:28Z, so there was no GPU smoke.
- **Hash.** `sha256sum results/PREREG-R6-E1-as-registered.md` = 8f0a3ec5...b65c = `PREREG-R6-E1.sha256`.
- **Only appended deviations.** `diff results/PREREG-R6-E1-as-registered.md PREREG-R6-E1.md` shows one hunk (146c146,157). The
  section-5 placeholder line "(none yet)" is replaced by the deviations-log text (deviations 1 and 2). The `Registered:` line
  is intact.
- **Rules copied verbatim.** The registered rules excerpt (2,629 characters, sha256 eb175845...) is a byte-exact prefix of
  spec section 1.2. The spec file still hashes to 8be631f1...170c, as registered. The spec continues only with `---`.
- **Staged code.** `results/stage-manifests.txt` has 3 blocks: 06:27:03Z (registration manifest), 06:28:06Z (launch 1) and
  06:51:25Z (launch 2).
  - All 14 registered files are identical in all three blocks.
  - Launch 2 also staged `render_page6.py` and `site_patch6.py`. These are publishing scripts created at 06:29Z;
    `remote6.sh` runs only `e1v3.py`.
  - A hash-identical copy of the staged code passes the 6 unit tests: `audit/unit-tests-output.txt`, 6 passed, 08:05Z.
- **Current files.**
  - Unchanged: `e1v3.py`, `test_r6e1.py`, `ledger6.py`, `stage6.sh`, `remote6.sh`, `driver6.sh`, `launch6.sh`, the five
    r5e1 modules and `smallnet/netdef.py`.
  - **`analyze6.py` changed:** now fd65675d, registered 2f76562b. The mtime is 07:48:57.04Z, after the raw results synced at
    07:47Z and 0.2 s before `results/main/analysis/*` was written. The change is not in either deviations log.
  - `diff /tmp/r6e1/src/analyze6.py analyze6.py`: the edit is plot-only. In panel 3 of `policy_independence.png`, bars of
    KL/10 were replaced by bars of the union-call JSD.
  - I re-ran the registered script (`/tmp/r6e1/src/analyze6.py`, 2f76562b) on copies of the three summary.json files, in
    `scratch/r6e1-audit/analyze-registered/`. It reproduces `results.json`, `verdicts.md`, `keys.csv`,
    `policy_independence.csv`, `controlled_bags.csv`, `keys.png` and `controlled_bags.png` byte-for-byte. Only
    `policy_independence.png` differs (`cmp`, 08:09:36Z).
  - So no number is affected, but the change must be logged, and the registered script survives only in `/tmp`.
- **Other facts.**
  - No amendment file exists (`ls /home/agent/workspaces/mats/outbox/experiments-round-6-amendments*.md`, 08:09Z and 08:19Z).
  - No run was resumed: `rc.json` retry 0 for all seeds, `resumed_from_saved_theta_real` null, `cfg.load_wm` empty.
  - theta_real trained 40,000 steps: `train_curve` ends at step 40000 in all three seeds.
  - The two CPU smokes ran before registration (06:25-06:26Z) on the first 40 test-stream boards, with a d-32, 120-step
    model. This is disclosed in the PREREG header.

## Check 2: no oracle in training or selection: PASS

Evidence: reading `e1v3.py` (650 lines) and every r5e1 function it calls. A grep inventory lists 33 distinct imported
names: `A2.train_world_model2`, `union_call`, `boards2_t`, `ev_from`, `cells_mask`; `RO.sample_patches`, `geo`,
`forward_chunks`, `trust_features`, `es_target`, `draw_tau`; `wm.*` pools and metrics; `env6.bayes_marginals`,
`placements`; `run` helpers.

- **Exact Bayes.**
  - `env6.bayes_marginals` is called only at `e1v3.py:475-476` (Pfull, Pstd6, test boards). It is used for the KL-to-Bayes metrics
    (`kl`, line 480, used at 504, 531 and 536), the gate clause 1 (482-483), raw `pstar` (484) and the controlled-bag MSE
    (591). The JSD metrics compare the two searches' beliefs and do not use it.
  - In `partA2.train_world_model2`, `mon_b` (line 127) is used only at line 140 (`monitor_kl_*`, curve only).
  - Training runs a fixed 40,000 steps and returns the final model: no selection and no early stop.
  - `partA2.fit2/test2`, which also call Bayes, are not called by `e1v3.py`.
- **Fitted to the model's own reference.**
  - The reference is `ref = sigmoid(union_call(...))` on 128 unskewed calibration reads (271-275).
  - All five learned pools train by BCE to `ref` (`train_pool`, 314-317), plus the K4 alignment loss to the model's own
    salience.
  - eta, u and alpha are fitted by BCE to `ref` (`fit_scalar`, 346-360). alpha is fitted on unskewed K = 24 bags, as
    registered.
  - Two targets are ground-truth occupancy, not exact Bayes, and both are registered. theta_trust is trained on real excess
    surprise against the *calibration* boards' true occupancy (281). theta_real is trained on fresh boards' true occupancy
    (`partA2.py:146-152`).
- **PUCT never peeks.**
  - Code: `prospect()` (127-153) builds candidate evidence from the root plus readings imputed as `mu >= 0.5`.
    `puct_search()` (412-448) scores with the merged belief of earlier real reads and crowding against earlier reads.
  - The owner's unit test passes.
  - My behavioural test (`audit/puct_peek_and_params.py`, a hash-verified code copy, tiny random model, 24 boards, U_NW and
    U_SE): for each wave w, I flipped the real readings of every cell no earlier read covered (anchors excluded). Waves 1..w
    stayed identical for w = 1..4. The flipped-cell counts were 816 / 315 / 196 / 107 (U_NW) and 816 / 308 / 170 / 126
    (U_SE).
  - Replacing the true occupancy changed nothing.
  - Positive control: flipping the readings inside wave-1 patches changed waves 2-4 on 24 of 24 boards, so the test can
    detect a dependence.
- **Patch ids.** `grep -nE '\["patch"\]|\bpatch\b' e1v3.py` returns 48 lines.
  - Patch ids are used in:
    - the read definition: which cells' readings are shown, and which cells' hidden states `_read3` pools into hbar;
    - the union-call reference;
    - theta_trust's target;
    - scoring and outputs (`coverage`, `key_stats`, raw);
    - `.shape` for masks.
  - They are never an input to KeyNet, AttnL3, LoopedLinT4, the one-step pools or theta_trust's features.
- **Streams and the oracle itself.** Evidence: `audit/streams_bayes.py` -> `audit/streams_bayes-output.txt`.
  - My own enumeration gives 49,504 occupancy patterns.
  - Regenerating test stream 22 reproduces raw.json's `occ` and `anchors` for all 3 seeds.
  - These are the same boards as R5-E1 v2: occ, anchors and note are equal, and pstar is identical to
    `li-net/r5e1/results/v2/s101/partA_raw.json`.
  - My own exact Bayes matches raw `pstar` to 5.03e-6 (max), which is the 5-decimal rounding.
  - Calibration stream 21 vs test: 0 of 500 test boards are identical. 39 share an occupancy pattern with some calibration
    board (chance expectation 38.8), and none shares the sonar field too.
  - Training boards are fresh draws from a seeded CUDA generator each step, so they are disjoint by construction. They cannot
    be regenerated on CPU (not checked).

## Check 3: re-derivation from raw.json: PASS (largest difference 5.8e-5, explained by rounding)

Own code: `audit/rederive.py` (numpy only; imports neither `e1v3.py`, `analyze6.py` nor any r5e1 module). Outputs:
`audit/rederive-output.txt`, `audit/rederive-output.json`, `audit/headline-per-seed.txt`. Re-run:
`OMP_NUM_THREADS=2 PYTHONPATH=/tmp/r4e3a/pylib python3 audit/rederive.py` (3 s).

The script compares 462 seed-level values with `summary.json` and 105 three-seed statistics (means and CI bounds) with
`analysis/results.json`:

| group (re-derived per seed) | values | largest abs difference |
|---|---|---|
| P-FullSupport: KL and JSD of all 9 pools + union call, P0 KL, coverage | 69 | 5.80e-5 (s101 `fs_kl_mmd`) |
| P-ActivePUCT: attn_K4 / mix / scaled_mix / mmd / kc KL and JSD; coverage from the saved patch lists; wave and open preferred shares | 108 | 1.42e-5 (s102 `ap_open_kl_mmd`) |
| keys K3, K4: Pearson with overlap/6, same-patch, across-sector, gap, cosine at 0/1/2/3/4/6 shared cells | 60 | 2.52e-6 |
| controlled bags: W(N) for K4 at 3 tiers x 5 N | 45 | 3.9e-7 |
| controlled-bag MSE of the saved pools (attn_K4, mmd, kc, mix) | 180 | 4.3e-8 |
| 3-seed statistics vs results.json | 105 | 5.5e-5 (E1-D2.1 mmd KL-diff CI low bound) |

- **Geometry.** I re-implemented the patch geometry (20 horizontal 2x3 then 20 vertical 3x2, row-major; sector by >= 4 of 6
  cells).
  - It gives 0 pairs at overlap 5.
  - The key-geometry bags are the open_U NW and SE bags (identical patch lists) plus the unskewed bag: 414,000 pairs per
    seed.
  - Renormalising the 4-decimal keys changes the Pearson by < 1e-6.
- **Why the KL differences exceed 1e-5.** `audit/kl_rounding_bounds.py` -> `audit/kl_rounding_bounds-output.txt`.
  - In seed 101's P-FullSupport beliefs, 206 of 36,000 ataraxos_mmd cells are saved as exactly 0 or 1 (kc_dedup 31,
    scaled_mix 1, attn_L3-K4 0). For such a cell the unrounded q is anywhere in [1e-6, 5e-6], so log q, and with it the KL,
    is not pinned down. That is why the mmd and kc KLs differ most.
  - With p = my exact Bayes, each summary.json KL lies inside [KL at the most favourable, KL at the least favourable
    unrounded q]. This holds for 63 of 63 statistics; for `fs_kl_mmd` the interval width is 4.2e-4 to 5.0e-4.
- **Union call.** The zero JSD of the P-FullSupport union call is exact (check 5).

## Check 4: rules re-decided from my numbers: same verdicts

PREREG-R6-E1 section 3; 3-seed means, seed-t 95% CI; ratios and cuts are means of per-seed values. Source:
`audit/rederive-output.txt`.

| rule | my decisive statistics | verdict (mine / verdicts.md / RESULTS) |
|---|---|---|
| E1-D0 | scaled_mix KL cut 0.6315 [0.6212, 0.6418] >= 0.40; clause-1 Pearson 0.9994 >= 0.99 (from summary.json; not in raw.json) | PASS / PASS / PASS |
| E1-D1 | K4: Pearson 0.8628 [0.8581, 0.8675], gap 0.6291, Spearman 1.000 (per seed 1, 1, 1). K3: Pearson 0.8799 [0.8759, 0.8838], gap 0.9673, Spearman 1.000 | PASS / PASS / PASS |
| E1-D2.1 | JSD ratio vs scaled_mix 5.823, mmd 6.536, kc 2.846 (all >= 2.5; per-seed minimum 2.683); JSD-diff CI low bounds 0.0350 / 0.0420 / 0.0129 > 0; KL-diff CI low bounds 0.0658 / 0.0830 / 0.0297 > 0 | PASS / PASS / PASS |
| E1-D2.2 | **coverage 0.94938 [0.94708, 0.95168] < 0.95** (exact from the patch lists; per seed 0.94836, 0.94961, 0.95017); JSD ratio 2.000 >= 1.8 (diff CI [0.0126, 0.0135]); KL cut 0.3823 >= 0.20; wave-1 share 0.8241 >= 0.75 | FAIL / FAIL / FAIL |
| E1-D3 | W16/W1 at tau 0.3 1.075 <= 1.25; MSE cut vs mmd at tau 0.3 0.385 >= 0.35; W8/W1 at tau 2.5 3.536 (CI low bound 3.283 > 1.90); W16/W1 at tau 2.5 4.522 >= 2.80; **MSE cut vs kc at tau 2.5 0.103 < 0.15**; monotone in all 3 seeds | FAIL / FAIL / FAIL |

Readings that would move a clause. These are sensitivity results, not re-decisions: the registered definitions govern, and no
verdict changes.

- **E1-D2.2 coverage.** Counting the 2 root anchor cells as covered evidence gives 0.9531 [0.9510, 0.9551], which would pass.
  The registered definition is "read at least once", which fails.
- **E1-D2.2 wave-1 share.** Under the registered convention the 4 straddling row-2/3 patches count as NW, giving 0.824. Counting
  only reads wholly in the NW half (>= 4 of 6 cells) gives 0.666 [0.642, 0.691], which would fail. E1-D2.2 fails either way.
- **E1-D3 MSE cut vs kc at tau 2.5.**

  | MSE cut definition | value | vs bar 0.15 |
  |---|---|---|
  | registered: ratio of N-averaged MSEs | 0.103 | fail |
  | mean of the per-N cuts | 0.104 | fail |
  | A* cells only | 0.069 | fail |
  | N = 16 only | 0.157 | pass |

- **E1-D3 MSE cut vs mmd at tau 0.3.** As the mean of per-N cuts it is 0.353, still >= 0.35 but by a hairline.
- **E1-D2.1.** "Lowest KL" holds against the three baselines, which is the rule's scope. Over all pools, attn_L3-K3 has lower
  KL than attn_L3-K4 in seeds 101 and 102.
- **E1-D2.1 kc ratio.** Its CI is [2.485, 3.207]; the rule uses the mean and the difference CI.

## Check 5: parameters and the zero-JSD union call: PASS

Evidence: `audit/puct_peek_and_params.py` -> `audit/puct_peek_and_params-output.txt`; `audit/rederive-output.txt` (checks).

- **Parameters.** Analytic counts and instantiated module counts both equal `summary.json` `fit.params` in all 3 seeds.

  | model | parameters | breakdown |
  |---|---|---|
  | looped_lin_T4 | 2,789 | K4 key head 2,640 + 4 eta_t + 4 x 36 P_t + 1 log s |
  | attn_L3-K4 | 108,228 | AttnL3 105,588 + K4 head 2,640 |
  | attn_L3-K3 | 106,180 | |
  | attn_L3-K1, attn_L3-K2 | 107,652 each | |
  | theta_real | 669,633 | |

  looped / attn_L3-K4 = 1/38.8 < 1/5, so the parameter half of the secondary bar holds. Its KL half fails (2.04x > 1.15x),
  as RESULTS reports.
- **Zero JSD.**
  - `fs_jsd_union` = 0.0 exactly in all three summary.json files.
  - In raw.json, the U_NW and U_SE union-call beliefs are bitwise identical (max abs diff 0.0) in all 3 seeds.
  - Both bags start with tiles 0, 3, 8, 11, 16, 19 on every board, and coverage is 1.0.

## Check 6: cost and live VMs: PASS

Evidence: `audit/integrity-output.txt` (08:19Z).

- **GPU-LOG.** `logical-soup/GPU-LOG.md` lines 2064-2088 contain 14 r6e1 lines for 2 VMs:
  - line 2075 COST-AUDIT: spot VM, age 0.32 h, $0.28 (preempted 06:47:12Z);
  - line 2088 DELETE: on-demand VM, age 0.94 h, $0.82.
- **Ledger.** `python3 ledger6.py` gives a total of **$1.10 of the $2.50 ceiling**.
- **Independent recomputation from GCE operations** (`gcloud compute operations list --filter="targetLink~soup-gpu-r6e1"`):
  - VM 1: insert 06:28:17Z to preempt 06:47:12Z = 0.3153 h, $0.274;
  - VM 2: insert 06:51:32Z to delete 07:47:41Z = 0.9359 h, $0.814;
  - total $1.09 at $0.87/h.
  - The most conservative reading bills VM 1 to its 06:51Z COST-AUDIT line, giving $1.16. Every reading is under $2.50.
  - Pre-launch checks in `/tmp/r6e1/logs/launch-main*.nohup`: $0.00 + $2.12 and $0.28 + $1.97, both <= $2.50.
- **Live VMs.** `gcloud --project=GCP_PROJECT compute instances list --filter='name~r6e1' --format='value(name,status)'`
  returned nothing (rc 0) at 07:54:53Z and at 08:19Z. A control listing in the same project returns
  `soup-gpu-r6e2-main-1017072822-15483 RUNNING`, so the query works.
- **Not checkable.** Actual billing: the Billing API is disabled and $0.87/h is an unverified list-price ceiling.

## Check 7: RESULTS-R6-E1.md read critically: numbers mostly right, several claims too strong

- **Numbers.** `audit/claims_curated.py` -> `audit/claims-curated.csv`, `audit/claims_curated-output.txt`.
  - I mapped all 337 numbers in RESULTS by hand to the quantity each claims (318 measured values, 19 bars or design
    constants). The script checks the rounding at the printed precision against `results.json` / `summary.json` /
    `audit/rederive-output.json`.
  - **323 are correct and 14 are wrong** (list in requested changes B). 10 of the 14 match a value that was rounded twice
    (for example 0.7547 -> 0.755 -> 0.76). The other 4 (three beta values and alpha) are seed 101's values printed without
    a seed label.
  - A further 36 tokens found by automatic extraction are bars, design constants, "95% CI" labels, the model name or ledger
    values. I checked each by eye; all are right.
  - I first tried an automatic rule: a token matches if any file value rounds to it (`audit/claims_check.py`). Against my
    hand labels it agreed on 304 of 317 tokens (0.959), but it caught only 1 of the 13 wrong numbers in that set. I
    rejected it; all totals above come from the hand map (`audit/auto-vs-hand-agreement.txt`).
- **Sources.** Each section names its source files (`verdicts.md`, `results.json`, the figure CSVs, `rc.json`, `ledger6.py`).
  Three things lack a source:
  - beta (line 51);
  - alpha (line 102);
  - R5-E1's 0.375 / 0.218. They match `li-net/r5e1/results/v2/analysis/results.json` (0.3749 / 0.2177) and the spec.
- **Failures reported as failures.** Yes: in the TL;DR table, sections 5 and 6, and for the looped secondary. One exception:
  section 10's "reaches 95% coverage" contradicts E1-D2.2's FAIL.
- **Exploratory and interpretive labels.** "Possible next steps (none was run)" and the descriptive union-call rows are
  labelled. Four mechanism or causal sentences are written as findings without a label: lines 41, 120-121, 166, 246-247.
- **Claims beyond the numbers.** See requested changes C.
- **The published page and site entry repeat the errors.** `/home/agent/memo-serve/lir-agent/r6-e1.html` and
  `site-entry.json` (07:51Z) contain 0.868, 2.9x, "2.9-6.5x", "reaches 95%", "clearly", 0.0222, 0.038, 3.33, "alpha = 0.23",
  "weight-tied", "never saw", "learned key crowding", 0.78, 0.76 and 0.72 (`grep -c`, 08:17Z).

## What I could not check

- Clause-1 Pearson (0.9994), the K1/K2/looped key statistics, and the P-ActivePUCT union-call, looped and attn_K1-K3 numbers.
  raw.json does not keep these outputs, so I took them from summary.json.
- That the training stream is disjoint from the test stream: it comes from a CUDA generator, so it is disjoint by
  construction only.
- That no amendment file existed at 06:28Z and 07:48Z: I could only check that none exists now.
- Agent-mail times #3704 and #3708: I did not read mail.
- That the spot VM was "15 minutes into training" when preempted: its text logs never synced ("final sync failed"). RUN was
  at 06:29Z and the preemption at 06:47:12Z, which fits.
- Actual billing.

## Requested changes

### A. Integrity

1. **Log the analyze6.py edit.** Append to PREREG-R6-E1.md section 5, and mirror in RESULTS section 7: "3. **07:48:57Z,
   `analyze6.py` edited after the raw results existed (plot only).** Panel 3 of `policy_independence.png` now shows the
   union-call JSD instead of KL/10. The registered script (2f76562b) reproduces `results.json`, `verdicts.md`, the three CSVs
   and the other two PNGs byte-for-byte (audit check 1)." RESULTS section 7 says "Nothing was retuned after any test number".
   That is true, but the section should list this change.
2. **Preserve the registered analysis script and the logs.** The only copy of the registered `analyze6.py` is
   `/tmp/r6e1/src/analyze6.py`, which is ephemeral. Copy it to `results/analyze6-as-registered.py`. Also copy
   `/tmp/r6e1/logs/` (about 60 KB) under `results/main/logs/`, or say in section 11 that the text logs are ephemeral.

### B. Wrong numbers (fix each; regenerate from results.json at full precision)

3. Lines 25 and 116: "0.863 [0.858, 0.868]" -> "0.863 [0.858, 0.867]" (CI high bound 0.867463).
4. Line 26: "5.8x / 6.5x / 2.9x lower" -> "5.8x / 6.5x / 2.8x lower" (kc ratio 2.846). Line 242: "2.9-6.5x below the
   one-step pools" -> "2.8-6.5x below the one-step pools".
5. Line 51: "theta_trust grades the tiers: beta 3.33 / 2.16 / 1.50." -> "theta_trust grades the tiers: mean beta on the
   calibration reads 3.47 / 2.18 / 1.52 (3-seed means; `results.json` `fit.<seed>.beta_by_tau`)". The printed values are
   seed 101's.
6. Line 102: "alpha = 0.23 fitted to the model's own reference" -> "alpha = 0.22 (per seed 0.226 / 0.218 / 0.223; `results.json`
   `fit`) fitted to the model's own reference".
7. Line 113, K1 cosine at 1 shared cell: "0.72" -> "0.71" (0.714982).
8. Line 115, K3 cosine at 4 shared cells: "0.78" -> "0.77" (0.774673).
9. Line 116, K4 cosine at 4 shared cells: "0.76" -> "0.75" (0.754641).
10. Line 138, kc_dedup JSD: "0.0222" -> "0.0221" (0.022149).
11. Line 140, attn_L3-K2 KL: "0.037 / 0.038 / 0.033" -> "0.037 / 0.037 / 0.033" (0.037463). Line 152: "attn_L3 with any key is
    at KL 0.033-0.038" -> "attn_L3 with any key is at KL 0.033-0.037".

### C. Wording: claims beyond the numbers, or imprecise

12. **Line 246-247.**
    - Quote: "Key-space PUCT reaches 95% coverage in 4 waves, and the remaining disagreement is the evidence one search never
      saw."
    - Problem: coverage is 94.94% and the bar was missed. Also, attn_L3's JSD after PUCT (0.0131) is 3.2x the clean union
      call's on the same read cells (0.0041); per seed 3.19 / 3.24 / 3.13.
    - Fix: "Key-space PUCT reaches 94.9% coverage in 4 waves (bar 95%, missed). After it, attn_L3's JSD (0.0131) is about 3x
      the clean union call's on the same read cells (0.0041), so evidence one search never saw is not the main remaining
      source of disagreement."
13. **Line 166.**
    - Quote: "In 4 waves, learned key crowding redirects the search to unread cells (coverage +12 points)."
    - Problem: no ablation separates the crowding term from the variance term, which also falls on read cells.
    - Fix: "In 4 waves, the PUCT score (belief variance divided by key crowding) moves the search to unread cells (coverage
      +12 points); which term does it was not tested."
14. **Lines 187-188 and line 47.**
    - Quotes: "now holds clearly at all three tiers" and "The weight law holds in all three regimes."
    - Fix: "holds on the 3-seed means at all three tiers; at tau 0.3, W(16)/W(1) = 1.08 has CI [0.63, 1.52] and seed 102 is at
      1.246, just under the 1.25 bar".
15. **Line 26, and section 10 line 242.**
    - Quotes: "and its KL is lowest (all paired CIs > 0)" and "and it is the most accurate pool".
    - Fix for line 26: "and its KL is below each of the three baselines' (all paired CIs > 0); attn_L3-K3's is lower still
      (0.033 vs 0.034, lowest of all pools in seeds 101 and 102)".
    - Fix for line 242: "and attn_L3 (with any key) is the most accurate pool family".
16. **Lines 27, 45 and 162 (wave-1 share).** Add the convention after "wave-1 preferred share 0.82" / "82% of wave 1": "counting
    the 4 patches that straddle rows 2-3 (3 cells in each half) as in both halves, as registered; reads wholly in the preferred
    half are 0.67 [0.64, 0.69]".
17. **Line 85.**
    - Quote: "looped_lin_T4: 4 weight-tied Richardson steps with K4 keys, 2,789 parameters."
    - Problem: as registered, each step has its own step size and preconditioner.
    - Fix: "looped_lin_T4: 4 Richardson steps sharing one K4 key head and one attention matrix G, each with its own learned
      step size eta_t and diagonal preconditioner P_t (as registered); 2,789 parameters."
    - Line 253 ("an untied last step"): say what would be untied, for example "a separate key head or G for the last step".
18. **Line 93.** "Seeds 101, 102, 103 on one L4 (`r6e1-main`)" -> "Seeds 101, 102, 103 on one L4 (`r6e1-main2`, the
    on-demand rerun after `r6e1-main` was preempted)".
19. **Lines 77-79.**
    - Quote: "Learned pieces (...; target = the model's own unskewed K = 128 union-call belief, never exact Bayes):
      theta_trust (real excess surprise)".
    - Fix: "Learned pools and fitted scalars: target = the model's own unskewed K = 128 union-call belief, never exact Bayes.
      theta_trust: real excess surprise of each calibration read against that board's true occupancy (no exact Bayes)."
20. **Line 41.**
    - Quote: "So round 5's failure was the support gap plus the yardstick, not attn_L3's deduplication."
    - Fix: "This is consistent with round 5's failure coming from the support gap and the yardstick, not attn_L3's
      deduplication: with full support even round-5-style raw keys (attn_L3-K1) give JSD 0.0091, 5.0x below scaled_mix. Round
      5's protocol was not rerun."
21. **Lines 120-121.**
    - Quote: "so their footprints overlap in proportion to the shared cells".
    - Fix: label it as interpretation and write "so their footprints overlap more as the reads share more cells (K3 cosine
      0.01 / 0.27 / 0.50 / 0.67 / 0.77 / 0.96 at 0 / 1 / 2 / 3 / 4 / 6 shared cells)".
22. **Lines 146-147.**
    - Quote: "the summing pools let 16 repeated NW reads outvote 2 SE reads; attn_L3 does not."
    - Fix: "the summing pools let the about 16 NW reads outvote the about 2 SE reads; attn_L3 is much less affected (JSD
      0.0078 vs 0.022-0.051)". Mark it as interpretation.
23. **Line 3.** "(06:15-08:xxZ)": fill in the end time.
24. **Line 7.**
    - Quote: "13 s before the first r6e1 GPU line (06:28Z)".
    - Fix: "13 s before the first r6e1 launch (06:28:06Z, `/tmp/r6e1/logs/launch-r6e1-main.log`; GPU-LOG line 2064 is stamped
      06:28Z; GCE insert 06:28:17Z)".
25. **Republish.** After these fixes, re-render `/home/agent/memo-serve/lir-agent/r6-e1.html` and `site-entry.json`
    (its one-liner says "JSD 2.9-6.5x below one-step pools, lowest KL").

## Files (audit/)

| file | what it does |
|---|---|
| `rederive.py` | independent re-derivation and rule decisions; outputs `rederive-output.txt`, `rederive-output.json`, `headline-per-seed.txt` |
| `streams_bayes.py` | own placement enumeration, regenerated streams 21 / 22, own exact Bayes; output `streams_bayes-output.txt` |
| `kl_rounding_bounds.py` | KL rounding intervals; output `kl_rounding_bounds-output.txt` |
| `puct_peek_and_params.py` | parameter counts and the PUCT flip test; output `puct_peek_and_params-output.txt`. It uses the hash-checked code copy in `scratch/r6e1-audit/codecopy/` (`copy.sha256`) |
| `unit-tests-output.txt` | the owner's 6 unit tests, re-run |
| `integrity.sh` | checks 1 and 6; output `integrity-output.txt` |
| `claims_curated.py` | hand-mapped number check; outputs `claims-curated.csv`, `claims_curated-output.txt` |
| `claims_check.py` | the rejected automatic rule; outputs `claims-tokens.csv`, `claims_check-output.txt`, `auto-vs-hand-agreement.txt` |
| `scratch/r6e1-audit/analyze-registered/` | the registered `analyze6.py` re-run, for the byte-comparison in check 1 |

Auditor footprint outside scratch, disclosed:
- At about 07:51Z I created and immediately removed a zero-byte `li-net/r6e1/.audit-write-test` to test write access. This
  only changed the directory's mtime.
- At about 08:05Z I created and removed two temporary hash-list files in `/tmp`.
- The mandated gcloud checks write gcloud's own client logs under its config directory.
- Nothing else was written outside `scratch/r6e1-audit/`. A `find -newermt 07:50Z` over li-net/r6e1, r5e1 and smallnet at
  08:24Z lists only the owner's own writes (RESULTS and GPU-LOG.md at 07:50:42Z, site-entry.json at 07:51:09Z).
