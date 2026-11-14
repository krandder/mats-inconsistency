# AUDIT R7-E1 (E-INFO-V4)

Independent audit, 2026-10-17 10:55-11:25Z, by a Board investigator subagent (Claude Opus 5.5, model id claude-opus-5-5[1m]).
I did not run the experiment. I worked read-only on it: no GPU, CPU jobs at 2 threads, no edits to code, results, checkpoints
or write-up, no messages. My role can write only under `/home/agent/workspaces/fable-strategy/scratch/`, so this file
and `audit/` are at `/home/agent/workspaces/fable-strategy/scratch/r7e1-audit/`, for the owner to copy into
`li-net/r7e1/`. Paths are relative to `li-net/r7e1/` unless absolute; `audit/...` means the audit folder. RESULTS line numbers
refer to `RESULTS-R7-E1.md` sha256 196d7727...907c (mtime 11:02:50Z, unchanged at 11:19:21Z).

## Overall verdict

**Sound, publishable after wording corrections. Every registered verdict is confirmed from my own numbers:** E1-D0 PASS,
E1-D1.1 PASS, E1-D1.2 FAIL (jacobi_lin_T4 W(16)/W(1) at tau 0.3 = 1.449 > 1.25), E1-D1 FAIL, E1-D2.1 PASS, E1-D2.2 PASS,
E1-D2 PASS, E1-D3 FAIL (MSE cut vs kc_K1 at tau 2.5 = 0.148 < 0.20). I would decide no rule differently: t = 4.303 and
t = 4.302653 give the same verdicts, and both failures fail in every seed and under every reading of the spec.

- **Integrity.** Registration (09:39:09Z) precedes the launch (09:39:22Z), the first GCE insert (09:39:31Z) and the first
  `soup-gpu-r7e1-` GPU-LOG line (09:39Z). No registered file has changed, and every launch staged the registered hashes. The
  registered `analyze7.py` reproduces all 9 analysis files byte for byte. The one deviation (spot preemption, then a rerun from
  the synced checkpoints) follows the registered recovery rule. It was logged at 10:35:18Z, 19 s before the relaunch.
- **No oracle** in training or selection. PUCT never reads an unread cell's real reading. Shown by the code, the owner's unit
  test, and my own 40-case flip test with the real theta_real.
- **Checkpoints.** All three load strictly into `wm.WorldModel(128, 4, 4, 384)`: 669,633 parameters, file hash equal to its
  `.json`. On CPU they reproduce the clause-1 Pearson on all 500 test boards to within 1.5e-7.
- **Largest numeric discrepancy: 1.44e-4.** It is the seed-102 KL of ataraxos_mmd under open_U with clean reads, a descriptive
  number: mine 0.169684 vs summary.json 0.169540. It lies inside the bound allowed by the 5-decimal rounding of raw.json
  (1.7e-3).
  - Statistics that enter a rule agree within 6.3e-5 per seed, and within 8.5e-5 for 3-seed statistics with the owner's t.
  - All 504 numbers I mapped by hand in RESULTS match their files at the printed precision.
- **Cost:** $0.99 from the GCE operation times; the ledger says $0.98. The ceiling is $2.50. No r7e1 VM was alive at 11:08:19Z.
- **To fix before publication** (details under "Requested changes"):
  1. The key-attribution claim depends on the order of the swaps. Salience keys close 44-80% of the gap when swapped inside
     kc_dedup, the order RESULTS uses, but only 5-9% when swapped inside attn_L3. RESULTS lists the claim under "Settled".
  2. Three other sentences claim more than the numbers show:
     - "matches attn_L3 on every measure";
     - "2-3% over one precision-crowding step" (the CI includes 0 at tau 0.3);
     - "repairs R6's looped linear pool" (looped_lin_T4's W(N) was not measured here).
  3. One interpretation omits that kc_K1 has the highest MSE of the kc pools at every tier.
  4. One count is wrong: there are seven fitted scalars, not six.
  5. About 25 derived descriptive numbers have no saved script. They are all correct.

## Check 1: timing and integrity: PASS

Evidence: `audit/integrity.sh` -> `audit/integrity-output.txt` (11:01:09Z); `audit/gcloud-operations.txt` (read-only
`gcloud compute operations list`, 11:08:29Z); `audit/analyze-registered/` (11:08:14Z); `audit/unit-tests-output.txt` (11:09:03Z).

| evidence | time (UTC) |
|---|---|
| `PREREG-R7-E1.sha256` "registered 2026-10-17T09:39:09Z" (file mtime 09:39:09.93) | 09:39:09 |
| stage manifest block of launch 1; `/tmp/r7e1/logs/launch-r7e1-main.log` LAUNCH | 09:39:22 |
| first GCE insert operation (us-central1-a, stockout; us-central1-c created 09:39:52-09:40:04) | 09:39:31 |
| first `soup-gpu-r7e1-` line of `logical-soup/GPU-LOG.md` (line 2094, PENDING-CREATE, minute stamp) | 09:39 |

- **Before registration.** No r7e1 GPU-LOG line exists before 09:39Z. The CPU smokes ran 09:29-09:36Z with d 32, 120
  training steps, 40 pool steps and 40 test boards, as disclosed.
- **Hash.** sha256 of `results/PREREG-R7-E1-as-registered.md` = 1f9179bc...710e, which equals `PREREG-R7-E1.sha256`.
- **Only appended deviations.** The diff has one hunk (217c217,228). The section-5 placeholder "(none yet)" is replaced by the
  deviations-log text (deviation 1). The `Registered:` line is intact.
- **Verbatim excerpts.**
  - Section 2a is byte-equal to the spec's tail from "### 3.2" (sha256 0877c163..., as registered). The spec file still hashes
    to cbece88d....
  - Section 2b is byte-equal to the amendments file's tail from "## Part 2" (ba418115...). The file is c377811d..., mtime
    09:24:14Z, and it is still the only `experiments-round-7-amendments*.md`.
- **Staged code.** `results/stage-manifests.txt` has 4 blocks: 09:38:44Z, 09:39:09Z, 09:39:22Z and 10:35:37Z.
  - All 15 registered files are identical in all 4 blocks.
  - The 10:35:37Z (rerun) block adds the three checkpoints (aef34272 / abb0fffc / f9842c1e, as in deviation 1).
  - It also adds `render_page7.py` and `site_patch7.py`: unregistered publishing scripts, created 09:40:52Z, that
    `remote7.sh` never runs.
- **Current files.** All 15 registered files still have their registered hashes. Created after registration:
  `render_page7.py`, `site_patch7.py`, `site-entry.json` and `GPU-LOG.md` (publishing and ledger; none computes a result).
- **Analysis.** I ran the registered `analyze7.py` (8263475f) on links to the three `summary.json` files. It reproduces
  `results.json`, `verdicts.md`, the 4 CSVs and the 3 PNGs byte for byte (`cmp`).
- **Unit tests.** Hash-identical copies of `test_r7e1.py` and `e1v4.py`: 9 passed.
- **Recovery (deviation 1).**
  - GCE operation `compute.instances.preempted` at 10:32:07.9Z.
  - The checkpoints were created 10:26:53-10:27:03Z (`.json` "created"). The VM-side reload check is 0.0
    (`results/main-preempted/s*/summary.json`).
  - The rerun's configuration differs from the preempted run's only in `load_wm`. `loaded_theta_real.sha256` equals the
    checkpoint hashes. `rc.json` reads rc 0, retry 0, preloaded true for every seed.
- **Tooling observation (no effect on results).** `driver7.sh` always exits 0. So the launch log ("END r7e1-main rc=0") and
  GPU-LOG line 2107 ("RUN-END ... rc=0") report success for the preempted VM, whose driver log reads "remote rc=255"
  (10:34:01Z).

## Check 2: no oracle in training or selection: PASS

Evidence:
- reading `e1v4.py` (656 lines) and every e1v3 / r5e1 function it calls;
- grep inventory `audit/oracle-inventory.txt` (11:09:18Z);
- my flip test, `audit/flip_test.py` -> `audit/flip-output.txt` (11:07:00Z);
- the owner's tests (9 passed).

- **Exact Bayes.** `env6.bayes_marginals` is called at `e1v4.py` lines 453-454 only. The two results (Pfull, Pstd6) feed only
  the test KLs, the clause-1 gate, `raw["pstar"]` and the controlled-bag MSE (line 586). Elsewhere it is called once, in
  `partA2.train_world_model2` line 127, for the monitor curve. The model returned is the step-40,000 model; nothing selects on
  the monitor.
- **Targets.**
  - Every learned pool (`train_pool`) and fitted scalar (`fit_logs`) is fitted by BCE to `ref`. `ref` is theta_real's own
    union call on the clean readings of 64 NW + 64 SE unskewed reads of calibration boards.
  - theta_trust is fitted by MSE to `es_target(lr, patch, C["occ"])`: the real excess surprise against the calibration board's
    true occupancy, not exact Bayes.
  - The held-out bag BCE is a monitor only.
- **PUCT reads.** `puct_search7` reads no unread cell's real reading, in either selection mode or read setting.
  - Candidate keys come from `E3.prospect`, which imputes readings as mu >= 0.5 and never touches `B["s"]`.
  - Real readings enter only through `bag_reads3`, for patches already picked.
  - My test used the real theta_real (`ckpt/s101`, CPU) with random pools on 8 test boards. For power and greedy selection,
    mixed and clean reads, and waves 1-5, I flipped, or replaced with random bits, every unread non-anchor reading. Waves
    1..w were unchanged in 40 of 40 cases.
  - Positive control: flipping the cells wave 1 read changed a later wave in 4 of 4 settings. Every wave picks 6 distinct
    patches.
- **Amended reading 9 is implemented as written.**
  - Wave 1 (w = 0) scores (1 + 3 x 1[half]) x Var with crowd 0 (lines 420-421) and samples without replacement.
  - Waves w >= 1 score (1 + 3 / (1 + w) x 1[half]) x (Var / (1 + crowd)^1.5)^2 (lines 423-424). The primary arm samples in
    proportion without replacement; the greedy arm takes the top 6.
  - The run's configuration has lam_U 3, alpha_nov 1.5 and power 2.
  - crowd sums relu(<k_p, k_r'>)^2 over all earlier reads, using attn_L3-K3's K3 head. k_p is that head applied to the
    imputed prospective salience, as the amendment states.
- **Patch ids** are never a key or pool input.
  - Inputs: keys see (hbar, Delta l); pools see (l0, lr, Delta l, beta, hbar, valid, k); theta_trust sees (hbar, entropy, log
    tau, swing).
  - Patch geometry enters only the read definitions (which 6 cells a read shows; hbar pools theta_real's hidden states at the
    read's own cells, as in R5 and R6), the search policies (`sample_patches`, PUCT's Var and half), the choice of background
    patches, and scoring (coverage, `key_stats`, W over A*'s cells).
- **Streams.** Calibration uses stream 21 and test uses stream 22 (`default_rng([stream, 7702])`); streams 23 and 24 feed
  only the training curves.
  - theta_real's training boards come on the fly from a separate torch generator (seed x 1000 + 211).
  - Streams 21 and 22 share no identical board (ship placement, readings and note) and no identical reading-and-note pattern.
    They share 39 ship placements, the chance level (expected 38.8 of 49,504).
  - The CUDA training stream cannot be regenerated on CPU, so I did not check it.

## Check 3: checkpoints: PASS

Evidence: `audit/ckpt_check.py` -> `audit/ckpt-output.txt`, `ckpt-output.json` (11:02:00Z). My own loader (np.load,
`load_state_dict(strict=True)`), my own placement enumeration, stream generator and exact-Bayes code; all 500 test boards, CPU,
2 threads.

| seed | sha256 = `.json` | parameters | missing / unexpected keys | clause-1 Pearson, CPU | summary.json | abs diff |
|---|---|---|---|---|---|---|
| 101 | aef34272... yes | 669,633 | none / none | 0.9994208 | 0.9994209 | 1.0e-7 |
| 102 | abb0fffc... yes | 669,633 | none / none | 0.9994829 | 0.9994828 | 1.3e-7 |
| 103 | f9842c1e... yes | 669,633 | none / none | 0.9994624 | 0.9994622 | 1.5e-7 |

- Each file has 58 float32 arrays. `e1v4.load_wm_npz` gives identical tensors.
- My placements (49,504) equal `env6.placements()`. My stream-22 boards equal `partA2.make_boards2(22, 500)` and raw.json's
  occ and anchors.
- The CPU root beliefs differ from raw.json's mu0 by at most 1.9e-4; the run used TF32 on the GPU.

## Check 4: re-derivation from raw.json: PASS

Evidence: `audit/rederive.py` -> `rederive.json`, `rederive-diffs.json`, `rederive-output.txt` (11:03:04Z; direct MSE / W
check 11:19Z). The code is numpy only, with its own patch geometry, KL, JSD and coverage. It imports no owner module.

| per-seed statistic vs summary.json | n | max abs difference | where |
|---|---|---|---|
| KL (P-FullSupport, all 13 pools + union; P-ActivePUCT open / puct / greedy / K30, saved pools) | 135 | 1.44e-4 | seed 102, clean open_U ataraxos_mmd (descriptive); all 135 within their first-order rounding bounds |
| KL of rule statistics | | 6.3e-5 | P-FullSupport ataraxos_mmd, seed 102; attn_L3-K3, looped_attn_T3, jacobi_lin_T4 <= 7.4e-8 |
| JSD | 132 | 1.6e-7 | |
| coverage from saved patch lists (first 24 reads), wave-1 shares, coverage by wave | 204 | 1.3e-7 | |
| controlled-bag W(N) of attn_L3-K3 and jacobi_lin_T4 (saved per board) | 90 | 4.4e-7 | |
| controlled-bag MSE of the saved pools (5 pools x 3 tiers x 5 N) | 225 | 4.0e-8 | |
| K3 key Pearson (keys saved at 4 decimals) | 3 | 9.7e-7 | |

- **Per-tier values.** Per-tier MSE (mean over N) and W(N) against results.json (150 values): max 4.4e-7.
- **Wave-1 preferred share** (mixed, U_NW): 0.8157 / 0.8207 / 0.8310; wholly in half 0.6533 / 0.6587 / 0.6830.
- **Structure.** In all 3 seeds:
  - every P-FullSupport bag starts with the 6 tiles;
  - the NW and SE union beliefs are identical;
  - every saved PUCT and greedy wave has 6 distinct patches;
  - no controlled-bag background patch overlaps A*.

## Check 5: rules re-decided from my numbers: same verdicts

Evidence: `audit/rules.py` -> `rules-output.txt`, `rules.json` (11:06:02Z). Inputs: my raw.json numbers, my CPU Pearson, my
parameter counts; t = 4.303.

| rule | clause | my value: mean [95% CI] (per seed) | bar | mine | results.json |
|---|---|---|---|---|---|
| E1-D0 | Pearson; scaled_mix KL cut | 0.99946 [0.99938, 0.99953]; 0.6295 [0.6195, 0.6394] | 0.99; 0.40 | PASS | PASS |
| E1-D1.1 | parameter ratio; KL ratio | 0.3713; 0.967 [0.927, 1.007] (KL 0.0317, reported against 0.038) | 0.38; 1.10 | | |
| | JSD ratio vs scaled_mix, vs mmd | 7.05 [4.84, 9.25]; 7.84 [5.14, 10.55] | 4.5 | | |
| | 8 paired CIs (KL and JSD vs scaled_mix, mmd, kc_K1, kc_K3) | smallest lower bound 0.00539 (JSD vs kc_K3) | > 0 | PASS | PASS |
| E1-D1.2 | parameters; KL below looped_lin and kc_K1, every seed | 812; 0.0408 vs 0.0680 and 0.0939, yes | 3,000 | | |
| | W(16)/W(1) at tau 0.3 | **1.449 [1.392, 1.507] (1.454, 1.424, 1.470)** | <= 1.25 | | |
| | W(8)/W(1) at tau 2.5; monotone | 2.201 [2.113, 2.290]; every seed | 2.0 | **FAIL** | FAIL |
| E1-D2.1 | coverage (mixed); wave-1 U_NW share | 0.9863 [0.9855, 0.9871]; 0.822 (U_SE 0.815) | 0.955, CI lo > 0.950; 0.75 | PASS | PASS |
| E1-D2.2 | mixed: JSD ratio, diff CI, KL cut | 2.93 [2.74, 3.12], [0.0139, 0.0190], 0.513 | 2.0, > 0, 0.35 | | |
| | clean: JSD ratio, diff CI, KL cut | 5.38 [4.58, 6.18], [0.0129, 0.0210], 0.531 | | PASS | PASS |
| E1-D3 | D3.1: W(16)/W(1); cut vs mmd (diff CI); cut vs kc_K1 (diff CI) | 1.024; 0.386 ([0.0183, 0.0233]); 0.251 ([0.0086, 0.0137]) | 1.25; 0.35; 0.20 | | |
| | D3.2: W(8)/W(1) [CI]; W(16)/W(1); cut vs mmd; kc_K3 diff CI | 3.591 [3.020, 4.162]; 4.306; 0.227; [0.0026, 0.0040] | 2.20, lo > 1.90; 2.80; 0.20; > 0 | | |
| | D3.2: MSE cut vs kc_K1 at tau 2.5 | **0.148 [0.114, 0.182] (0.152, 0.159, 0.133)** | 0.20 | | |
| | D3.3 monotone | every seed | | **FAIL** | FAIL |

- **Agreement with results.json.** Across the 3-seed statistics (means, CI bounds, per-seed values) the largest difference
  is 2.4e-4: the upper CI bound of the JSD ratio vs mmd, caused by my t = 4.303 against the owner's 4.302653. With the owner's
  t the largest is 8.5e-5 (KL difference vs mmd, from the rounding of mmd's saved beliefs). The gate passed, so no
  "descriptive" label applies.
- **No rule decided differently.** I checked the readings that could matter.
  - Parameters: the spec's "~36k parameters" against 39,428 with the head (38,836 without). The binding clause, <= 38%, passes
    either way (0.371 / 0.368).
  - KL <= 0.038 also passes (0.0317).
  - E1-D2.1 uses the mixed setting and U_NW. The clean coverage (0.9870) and the U_SE wave-1 share (0.815) also pass.
  - E1-D1.2 against R6's reported 0.069 instead of the retrained 0.0680 passes either way.
  - The two failures hold under every reading, in every seed.

## Check 6: parameters, the zero-JSD union call, jacobi_lin_T4's W: PASS

Evidence: `audit/params_jacobi.py` -> `params-jacobi-output.txt`, `params_jacobi.json` (11:05:54Z).

- **Parameters.** I instantiated each pool with its key head and also counted by hand from the layer shapes:
  - attn_L3 with K1 / K3 / K4 head: 107,652 / 106,180 / 108,228;
  - looped_lin_T4 with K4 head: 2,789;
  - jacobi_lin_T4 with K3 head: 812 (core 220: lambda 36, eta 4, P 144, s 36; head 592);
  - looped_attn_T3 with K3 head: 39,428 (core 38,836; it has exactly one MultiheadAttention).

  All equal `summary.json` `fit.params` in all three seeds. Ratio 0.37133 (cores only 0.36781).
- **Union call on P-FullSupport.** The NW and SE union beliefs in raw.json are identical in all 3 seeds: JSD exactly 0.0,
  `fs_jsd_union` 0.0, coverage 1.0 for both bags. This holds by construction (both bags read the 6 tiles).
- **jacobi_lin_T4's W.** I tested 15 bags: 5 parameter settings × 3 bag types.
  - Parameter settings: the initial values; the three seeds' trained lambda, eta and s, with P_tc set to its saved per-step
    mean because the per-cell P is not saved; and random values.
  - Bag types: random K = 24, and 8 or 16 identical reads plus background.
  - Results: the owner's `JacobiLinT4.coef` equals the autograd derivative of the forward output with respect to Delta l
    within 3.0e-7, and my own float64 construction within 2.4e-7. It reconstructs the forward output within 7.6e-6.
  - So W is the exact coefficient of the 4-step linear map, as registered (reading 4).
  - The spec's per-read s_c x_rc / Delta l_rc differs from it whenever cluster and background reads mix: median relative
    difference 1-37% on my synthetic random-key bags. The trained pools were not saved, so I could not measure the
    difference on the real controlled bags.
- **RESULTS' closed form is right.** For N identical reads alone, x^(T) = [1 - prod_t (1 - eta_t P_tc)] beta Delta l /
  (lambda_c + N beta), so W(16)/W(1) = 16 (lambda + beta) / (lambda + 16 beta). Per seed, from the cell-mean lambda and the
  calibration beta at tau 0.3: 1.428 / 1.391 / 1.438, against the measured 1.454 / 1.424 / 1.470.

## Check 7: cost and live VMs: PASS

Evidence: `audit/cost.py` -> `cost-output.txt` (11:08:55Z); `gcloud-output.txt` (11:08:19Z); `gcloud-operations.txt`
(11:08:29Z).

| VM | lifetime (GCE operations) | hours | cost at $0.87/h |
|---|---|---|---|
| soup-gpu-r7e1-main-1017093922-28650 (spot) | insert done 09:40:04Z -> preempted 10:32:19Z | 0.871 | $0.758 |
| soup-gpu-r7e1-main2-1017103537-1091 | 3 insert attempts, all stockouts: never existed | 0 | $0 |
| soup-gpu-r7e1-main2-1017103729-163 (on-demand) | insert done 10:38:15Z -> delete done 10:54:00Z | 0.262 | $0.228 |

- **Total.** $0.986 from the GCE operations; $0.988 from GPU-LOG minute stamps with the 10:32:07Z preemption. The ledger
  (`ledger7.py`, `GPU-LOG.md`) says $0.98, because its DELETE line uses 0.25 h. Both are within the $2.50 ceiling.
- **Rate.** These use the on-demand list-price ceiling; the Billing API is disabled, so I could not check the real spend.
  Spot is cheaper.
- **Live VMs.** `gcloud --project=GCP_PROJECT compute instances list --filter='name~r7e1' --format='value(name,status)'`
  returned nothing at 11:08:19Z. The only soup GPU VM alive was `soup-gpu-r7e2-main-1017093146-24354`, another experiment's.

## Check 8: RESULTS-R7-E1.md: numbers right, failures reported, several claims too strong

Evidence:
- `audit/results_numbers.py` -> `results-numbers-output.txt`, `results-numbers.json` (11:14:47Z);
- `audit/uncovered_labels.py` -> `uncovered-labels-output.txt`;
- `audit/extra_checks.py` -> `extra-checks-output.txt` (11:16:48Z).

- **Numbers.**
  - I mapped 504 numbers by hand (158 entries on 134 lines) to their sources (results.json, summary.json, the checkpoint
    `.json` files, the ledger) and recomputed each at full precision. None differs at its printed precision.
  - An automatic scan finds 537 decimal tokens. I labelled by hand all 101 that no hand entry covers:
    - 57 design constants or labels (tau tiers, flip rates, t, exponents, ceilings);
    - 22 rule bars, all equal to PREREG section 3;
    - 9 result numbers already verified by an entry on the adjacent line;
    - 8 spec or R6 reference values, all present in the spec;
    - 5 ledger costs, all in `GPU-LOG.md`.
  - Hand spot check of 30 randomly drawn automatic comparisons, against the owner's factsheet and results.json: 30 of 30
    agree.
- **Sources.** About 25 derived descriptive numbers come from a computation that is saved nowhere: the attribution shares,
  the decomposition ratios with their CIs, the paired differences against attn_L3-K3, jacobi minus econ, the closed-form values
  and the learned-parameter summaries. The owner's `/tmp/r7e1/factsheet.py` prints none of them, and `/tmp` is not kept. The
  beta range "3.35-3.62" names no source; it is `results.json` `fit.<seed>.beta_by_tau`, a calibration mean.
- **Failures are reported as failures.** Both appear in bold in the TL;DR table, with the failing clause and its numbers, and
  again in sections 4 and 6.
- **Labels.** The decomposition and the closed form are labelled as interpretation. Key attribution is labelled descriptive
  in section 6, but not in the TL;DR bullet (line 47) or in section 11, which lists it under "Settled".
- **Claims beyond the numbers** (`audit/extra-checks-output.txt`).
  1. **Key attribution depends on the order of the swaps.** RESULTS swaps keys inside kc_dedup (kc_K1 to kc_K3) and credits
     the keys with 44-80% of the gap from kc_K1 to attn_L3-K3. The run also has attn_L3-K1, so the other order can be
     computed: swapping the keys inside attention (attn_L3-K1 to attn_L3-K3) closes only
     - 4.9% [3.5, 6.4] of the KL gap;
     - 5.0% [0.2, 9.9] of the JSD gap;
     - 8.1% / 9.4% / 5.3% [-1.6, 12.2] of the MSE gap at the three tiers.

     K3 keys matter a lot for kc_dedup and little for attention: a large key-by-pool interaction. So "salience keys account
     for 44-80%; the attention pool accounts for the rest" is not identified by these data.
  2. **"matches attn_L3 on every measure"** (listed as settled).
     - The registered E1-D1.1 passed, and no difference is detectable with 3 seeds.
     - Equivalence was not shown: the JSD difference is -0.00036 [-0.00179, 0.00107] on a JSD of 0.0069.
     - I confirm looped_attn_T3's weight-law clauses in every seed: W(16)/W(1) at tau 0.3 is 1.075 / 1.019 / 1.122; W(8)/W(1)
       at tau 2.5 is 3.740 / 3.352 / 3.880; monotone in every seed.
  3. **"2-3% over one precision-crowding step"** (listed as settled): at tau 0.3 the margin is 2.0% with CI [-0.1, 4.1],
     which includes 0.
  4. **"repairs R6's looped linear pool".**
     - looped_lin_T4's W(N) was not measured: it is not in WPOOLS, and summary.json has no `W_looped_lin`. The deduplication
       repair is therefore an argument from the formula: R6's update gives W(N) proportional to N for isolated identical
       reads.
     - The two pools also use different key heads (their own K3 vs their own K4), so the 40% KL gain is not attributable to
       the normal-equation fix alone.
  5. **kc_K1 interpretation.** RESULTS says "It hurts at tau 0.3, where kc_K1 has the highest MSE of the three kc pools".
     kc_K1 has the highest MSE of the kc pools at every tier (0.04425 / 0.04294 / 0.04797). What changes with noise is
     attn_L3-K3's margin over it.
  6. **"six fitted scalars"**: the run fits seven (eta, u_K1, u_K3, u_K4, econ_kc_K3's u and lambda, alpha) in six fits.

## What I could not check

- **Training stream.** The training boards come from an on-the-fly CUDA generator that cannot be regenerated on CPU. Their
  disjointness from the test boards holds by construction only.
- **Trained pools.** The pools, key heads and theta_trust were not saved, so I could not recompute any belief, weight or MSE
  from the models. Check 4 re-derives from the saved per-board beliefs and W values, rounded to 5 decimals. Check 6 uses
  jacobi_lin_T4's saved scalars, with P_tc replaced by its per-step mean.
- **Owner-only facts.**
  - mail times (#3717 09:15Z, #3722 09:22:01Z);
  - "found the amendment at 09:35Z";
  - "synced by 10:29Z";
  - the 10:30Z strict-load check;
  - the 10:54:46Z gcloud check.

  I did not read agent mail. Instead I confirmed the present state: the checkpoints load and match their hashes, no r7e1 VM
  at 11:08:19Z, one amendments file.
- **Real GPU spend** (the Billing API is disabled); I checked only the ceiling-rate cost.
- **R6 reference values** (0.9494; 0.069; K1 0.382, K3 0.880, K4 0.863) are checked against the spec text, not R6's files.

## Requested changes

Each item quotes RESULTS-R7-E1.md (sha256 196d7727...) and gives the fix. Items 1-6 change claims; 7-10 fix numbers and
sources; 11-13 are optional.

1. Line 47-48: "**Key attribution: salience keys account for 44-80% of attn_L3's margin over raw-key kc_dedup, depending on
   the measure; the attention pool accounts for the rest.**"
   Fix: "**Key attribution (descriptive; the split depends on the order of the swaps).** Swapping kc_dedup's keys from K1 to
   K3 closes 44-80% of the gap between kc_K1 and attn_L3-K3, depending on the measure. Swapping attn_L3's own keys from K1 to
   K3 closes only 5-9% of the same gap, because attn_L3-K1 is already close to attn_L3-K3. Salience keys matter far more for
   kc_dedup than for attention, so the gap does not split into a key share and a pool share."
2. Line 256-257: "So 44-80% of attn_L3's margin over kc_dedup with raw keys, depending on the measure, comes from the keys.
   The rest is the attention pool itself:"
   Fix: "In this order (keys swapped inside kc_dedup), K3 keys close 44-80% of the gap. In the other order (keys swapped
   inside attn_L3, attn_L3-K1 to attn_L3-K3) they close 4.9% [3.5, 6.4] of the KL gap, 5.0% [0.2, 9.9] of the JSD gap and
   8.1%, 9.4% and 5.3% [-1.6, 12.2] of the MSE gap. The interaction is large, so neither order gives the keys' share. With
   the same K3 keys:" (then keep the kc_K3 and econ_kc_K3 margins).
3. Line 350-351, under "Settled in this testbed": "44-80% of attn_L3's margin over raw-key kc_dedup, depending on the measure,
   came from its salience keys. On the same keys, attention keeps an MSE margin of 5-7.5% over kc_dedup and 2-3% over one
   precision-crowding step."
   Fix: move it out of "Settled" into a descriptive bullet: "Salience keys close 44-80% of the kc_K1 to attn_L3-K3 gap when
   swapped into kc_dedup, but only 5-9% when swapped into attn_L3 (descriptive, order-dependent). On the same K3 keys,
   attention's MSE is 5.1-7.5% below kc_dedup's (all CIs > 0) and 1.8-3.2% below one precision-crowding step's (CI includes 0
   at tau 0.3)."
4. Line 346-347, under "Settled": "A weight-tied 3-pass attention pool matches attn_L3 on every measure here (P-FullSupport
   accuracy and policy independence, the three-regime weight law in every seed, controlled-bag MSE) at 37% of the
   parameters."
   Fix: "A weight-tied 3-pass attention pool with 37% of the parameters passed its registered rule (KL within 1.10x of
   attn_L3-K3). On the other measures here no difference from attn_L3-K3 was detectable with 3 seeds: policy-independence JSD,
   the weight-law clauses in every seed, controlled-bag MSE. Equivalence was not tested; the JSD difference is -0.00036
   [-0.00179, 0.00107]."
   At line 38, "**Weight-tied attention matches attn_L3 at 37% of the parameters.**" becomes "**Weight-tied attention
   passes its rule at 37% of attn_L3's parameters, with no detectable difference from attn_L3-K3 (3 seeds).**"
5. Line 41: "**jacobi_lin_T4 repairs R6's looped linear pool but does not deduplicate clean reads enough.**"
   Fix: "**jacobi_lin_T4's KL is 40% below R6's looped linear pool's, but it does not deduplicate clean reads enough.**" Add:
   "looped_lin_T4's W(N) was not measured here (its formula gives W(N) proportional to N for isolated identical reads). The
   two pools use different key heads (K3 vs K4), so the KL gain is not attributed to the normal-equation fix alone."
6. Line 264-265: "That suits high noise, where support should grow. It hurts at tau 0.3, where kc_K1 has the highest MSE of
   the three kc pools."
   Fix: "kc_K1 has the highest MSE of the three kc pools at every tier (0.04425 / 0.04294 / 0.04797). What changes with noise
   is attn_L3-K3's margin over it: 25.1% at tau 0.3, 14.8% at tau 2.5. That a growing cluster weight suits high noise is a
   reading, not a test."
7. Line 334-335: "calibration fit (reads, reference, theta_trust, six learned pools, six fitted scalars) 535-541 s"
   Fix: "... six learned pools, seven fitted scalars in six fits ...".
8. Line 45-46: "With the learned ridge lambda (cell mean 1.55-1.61) and the clean-read precision beta (3.35-3.62) that is
   1.39-1.44, near the measured 1.42-1.47."
   Fix: "With the learned ridge lambda (cell mean 1.583 / 1.552 / 1.612, `results.json` `fit.<seed>.jacobi`) and the mean
   precision of the tau-0.3 calibration reads (3.365 / 3.622 / 3.351, `fit.<seed>.beta_by_tau`), that is 1.43 / 1.39 / 1.44
   per seed, near the measured 1.45 / 1.42 / 1.47."
9. Line 12: "Every number below names its source file". Today about 25 derived numbers have no re-runnable source:
   - lines 148-149, 160-162 and 168;
   - lines 44-46, 47-50, 55-59, 200-204, 253-256 and 350.

   Fix: commit the derivation as a script in `li-net/r7e1/` (for example `/tmp/r7e1/factsheet.py`, extended to print these
   numbers), cite it in those sections, and keep its output next to `results.json`.
10. Line 150: "Its JSD is the lowest of all pools except mix, which barely uses the reads (KL 0.1973)."
    Fix: "... except mix, whose belief stays far from exact Bayes (KL 0.1973, against the root's 0.2797)."
11. Optional, line 62 and line 332: "**Cost: $0.98**". Fix: "$0.98 by the ledger. The GCE operation times give $0.99: the
    rerun VM ran 0.262 h, not the 0.25 h in its DELETE line."
12. Optional, line 306: "**10:32:07Z** GCE preempted the spot VM during pool training; no test number existed." Add: "The
    launcher and GPU-LOG.md record rc=0 for this run because `driver7.sh` always exits 0; the driver log has remote rc=255."
13. Optional, line 52-53: "The amended score and the K3 keys changed together, so their shares were not separated."
    Fix: "The amended score, the K3 keys and the K3 pool's belief in the variance term (R6 used K4 for both) changed
    together, on a retrained world model, so their shares were not separated."

Also replace the `@@AUDIT@@` placeholder in section 10 with this audit's verdict paragraph.

## Files (audit/)

| file | check | what |
|---|---|---|
| `integrity.sh`, `integrity-output.txt` | 1 | times, hashes, diff, excerpts, stage manifests, rc lines |
| `analyze-registered/` | 1 | the registered `analyze7.py` re-run on links to the three summaries; outputs identical |
| `unittest/`, `unit-tests-output.txt` | 1 | hash-identical copies of the owner's tests: 9 passed |
| `oracle-inventory.txt` | 2 | every exact-Bayes, occupancy, readings and patch-geometry use |
| `flip_test.py`, `flip-output.txt` | 2 | PUCT leak test with the real theta_real; stream 21 vs 22 overlap |
| `ckpt_check.py`, `ckpt-output.txt`, `ckpt-output.json` | 3 | strict loads, hashes, parameter counts, CPU Pearson |
| `rederive.py`, `rederive.json`, `rederive-diffs.json`, `rederive-output.txt` | 4 | numpy re-derivation from raw.json |
| `rules.py`, `rules.json`, `rules-output.txt` | 5 | every rule re-decided from my numbers |
| `params_jacobi.py`, `params_jacobi.json`, `params-jacobi-output.txt` | 6 | parameter counts, union call, Jacobi coefficients |
| `cost.py`, `cost-output.txt`, `gcloud-output.txt`, `gcloud-operations.txt` | 7 | spend and live VMs |
| `results_numbers.py`, `results-numbers.json`, `results-numbers-output.txt` | 8 | 504 hand-mapped RESULTS numbers |
| `uncovered_labels.py`, `uncovered-labels-output.txt` | 8 | hand labels of the 101 unmapped tokens; 30-number spot check |
| `extra_checks.py`, `extra-checks-output.txt` | 8 | attribution in both orders, looped_attn_T3 weight law, kc MSE ranking |
