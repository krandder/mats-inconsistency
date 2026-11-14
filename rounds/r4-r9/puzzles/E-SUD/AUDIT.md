# E-SUD independent audit

Auditor: Claude Opus 5.5 (claude-opus-5-5[1m]), read-only. Checks run 2026-10-15 04:09-04:14Z on General.
Scratch scripts: `/tmp/esud-audit/recompute.py` (numpy only), `/tmp/esud-audit/check_exact_rerun.json`.

## Bottom line

**Supported with corrections.** I recomputed every D4 cut, seed-t CI and verdict, and every cell-KL cut, from
`perboard_*.npz` with numpy. All of them match RESULTS.md and ANALYSIS.json to the printed precision. The single
rounding difference (P2 vs P1 at M = 8, upper CI 8.4 vs 8.5) does not change the meaning. The verdicts are applied as
pre-registered. Round 1 (FAIL) is labelled as the pre-registered result and round 1b as post hoc. The addendum was
written before any round-1b VM or result existed. Standing rules hold in the code: training uses outcomes only, there
is no rule checker in the arms, and lambda, theta and kappa are fitted on validation only. Val, test and train are
disjoint, and I re-ran the exact marginals. Ceiling spend is about $4.7, against the $50 cap.

The corrections are all small, and none of them changes a verdict:
- RESULTS gives the false-positive range as "58-83%"; it is 39-83%.
- "Clue count matched" holds per board but not by distribution. Solvable test boards have 13.57 givens on average and
  unsolvable ones 14.83. "Mean clue count 14.3" is the mean over all boards, not over solvable boards as written.
- run.sh no longer matches PREREG.sha256 (disclosed). analyze.py was edited after round-1 results and was never hashed.
- esud-f was an on-demand VM, not spot.
- The ledger total of reserved-but-never-billed money is $22.26, not $10.26.
- esud-a was still RUNNING at 04:10Z although all its expected outputs were already pulled.

## Findings

### 1. Numbers reproduce from raw files: OK (2 wording ISSUEs)

Check: `python3 /tmp/esud-audit/recompute.py`, 04:11Z. results/r1 and r1b eval and perboard files are md5-identical
to the copies in vmout/.

- **D4, round 1 (M = 8).** All per-seed cuts, means and seed-t CIs match. Verdicts match: FAIL for P1, P2 and P3, with
  (a)-unsolvable and (b) failing.
  - P2: a_pos 0.963/0.992/1.000, mean 0.985 [0.937, 1.033]; a_neg mean -4.63 [-16.8, 7.5]; b_neg 0.32 [-1.04, 1.69].
  - Mean log losses: unsolvable P0 0.391, witness-only 2.604, null 2.495, count 2.482; solvable P0 0.606, witness
    0.006, null 0.011.
  - ECE (P2, M = 8): P0 0.027, null 0.278.
  - All of these match. Bootstrap CIs match ANALYSIS.json.
- **D4, round 1b.**
  - P2 a_pos is 1.000/0.309/1.000, mean 0.770 [-0.221, 1.760], FAIL. a_neg and b_neg are both 0.991 [0.953, 1.029].
  - P1 is 0.9999 / 0.981 [0.910, 1.051] / 0.981. P3 is about 1.000 on all three parts.
  - Guard and log-loss values match.
  - The single-board P2 failure is confirmed:
    - seed 1, found_pos = 0.9988, so 1 of 857 boards had no witness;
    - theta_voi = 79.43;
    - the maximum solvable-board LL is 13.816, which is the 1e-6 clip.
- **Cell KL, round 1b.** The KL-vs-M table means and sds match.
  - Cuts against P0: P1 27.35% [25.6, 29.1] / 44.53% [42.3, 46.8]; P2 29.76% [27.0, 32.5] / 40.75% [35.8, 45.7];
    P3 33.85% [32.2, 35.5] / 46.30% [41.7, 50.9].
  - P2 vs P1 is 3.30% [-1.8, **8.4**] (RESULTS prints 8.5; ANALYSIS gives 0.0845) and -6.88% [-19.5, 5.8].
  - P3 vs P2 is 5.82% [4.3, 7.3] and 9.37% [8.3, 10.4].
  - Weighted vs unweighted P1 at M = 16 is 0.0254 vs 0.0337. All match.
- **Cell KL, round 1.** The means 0.445 / 0.411 / 0.384 / 0.375 match. Seed 2 alone gives 0.205 / 0.128 / 0.066 /
  0.054, and its M = 16 cuts are 38% / 68% / 74%. These match too.
- **Other numbers that match:**
  - D0 per seed: r1 is 0.549 / 0.583 / 0.205 overall and 0.475 / 0.523 / 0.000 one-step; r1b is 0.0475 / 0.0467 /
    0.0433 overall and 0.000 one-step.
  - Best steps and validation CE/BCE.
  - Chain validity: P1 0.90-0.93, P2 and P3 0.99.
  - Fitted theta range in r1b: 7.08-79.43.
  - "37 passes per step": 1 + 6 x 6 = 37.
- **ISSUE (wording).** RESULTS says the check "accepted 58-83% of invalid completions" in the two failing seeds. The
  eval JSON chain_diag check_fp gives seed 0 0.394/0.642/0.738 and seed 1 0.582/0.768/0.834, so the range is
  **39-83%**. RESULTS' own table shows 0.39.
- **ISSUE (minor wording).** "VoI is 9% worse than confidence-first" is really P3's KL being 9.4% lower than P2's. In
  the other direction, P2's KL is about 10% higher than P3's.

### 2. Verdict application and labelling: OK, with 2 process ISSUEs

- **Rule application.** analyze.py `cut()` uses PASS = mean >= 0.20 and lower bound > 0, and STRONG PASS = lower bound
  >= 0.20, with t = 4.303. Overall PASS requires all three parts to be non-FAIL. This is exactly the pre-registered
  rule. The baselines are right: P0 at M0 for (a), witness-only at M8 for (b).
- **Labelling.** Round 1 is the pre-registered FAIL. Round 1b is labelled post hoc in the Verdict, its section header
  and Deviation 1. The /experiments label "Informative" is honest.
- **Addendum timing.** PREREG-ADDENDUM-1b.md has mtime 02:03:48Z. The first round-1b VM (esud-d) was created at 02:04Z
  per ledger line 1503, and all r1b result files have mtimes of 04:06-04:08Z. So the addendum predates any round-1b
  model, which is plausible and consistent.
  - Its header says "~02:08Z", later than the mtime, which is harmless.
  - It says "each 1b seed on its own VM (esud-d/e/f)". In fact seed 0 ran on esud-a and seeds 1-2 shared esud-f,
    which Deviation 3 partly discloses.
- **Hashes** (`sha256sum -c PREREG.sha256`, 04:10Z):
  - Match: PREREG.md, code/sud.py, code/boards.py, code/check_exact.py, PREREG-ADDENDUM-1b.md.
  - Mismatch: **jobs/run.sh FAILED** (mtime 02:08:17, after the hash file's 02:03:49). Deviation 2 discloses it as an
    added ROOT variable. I cannot check the old content.
- **ISSUE (process).** PREREG.sha256 was regenerated at 02:03:49 to add the addendum, so no hash survives from the
  original 01:01Z freeze. sud.py's mtime (01:10:48) is after PREREG.md (01:01:24) and after esud-a's creation (00:54).
  This fits the disclosed theta-grid widening, but it cannot be proven from hashes.
  - The widening mattered: round-1 fitted theta_voi for seed 0 is 281.8, above the original 100 bound.
- **ISSUE (process).** code/analyze.py, which produces the verdict, is not in PREREG.sha256 and was modified at
  02:20:44, about 1.5 min after the round-1 results were pulled (02:19). I read the current version and its statistic
  matches PREREG, but I cannot see what changed.

### 3. Standing rules: OK

- **Outcomes only in training.** boards.py drops `marg` from train files, and s6_train.npz holds only
  [board, solvable, outcome, nsol].
  - make_batch and losses use only the outcome and the solvable bit. val_outcome_loss uses outcome CE plus Solvable
    BCE. Checkpoints are chosen by that loss.
  - Note: on unsolvable boards, training states reveal cells of the hidden pre-corruption grid. This is
    generator-side information, but it is pre-registered and contains no marginals.
- **No rule checker in the arms.** `chains()` uses only model passes. `valid_full` appears only in chain_diag (the
  scorer), and `one_step_cells` only in D0 (the scorer). The network has learned position ids and no geometry.
- **Nuisance fits on validation only.**
  - lambda: `fit_lam` uses validation outcome CE (index [1]), not the exact marginals.
  - theta and kappa: fitted on validation solvable outcomes, pooled over M = 1-16.
  - The test split is used only for scoring.

### 4. Code bugs that could bias results: no bias-changing bug found (OK)

- **chains().** log q accumulates the log probability of the sampled digit only while a chain is active. rb stores the
  conditional each cell was sampled from. The final check runs with the Solvable token masked (`-one`).
  - Inactive chains (board already full) get dummy picks, but these are masked by `act`.
  - The VoI arm scores the expected squared change over the other masked cells with 6 x N extra passes.
- **sol_terms.** A completion is refuted when chk < 0.5. C_M sums exp(max log q) over distinct completions (keyed by
  bytes) and is capped at 1 - 1e-9.
  - Caveat (not a bug): under P1 the order is random, so q_sigma(z) depends on the order. C_M is therefore
    order-specific proposal mass, not the marginal mass of z.
- **cell_est / fit_lam.** These follow the formula. Given cells are excluded by msk, and the lambda grid includes 1e6
  (that is, P0).
- **D0 one-step.** An empty cell counts as one-step when its peers among the givens hold N - 1 distinct digits (a naked
  single). That is correct, giving n = 2,237 one-step and 16,983 multi-step cells on test.
- **cut() baseline M.** P0 is taken at M0 and witness-only at the same M = 8, as it should be.
  - Minor inconsistency: the bootstrap uses the ratio of pooled means across resampled seeds, while the point estimate
    is the mean of per-seed ratios. This changes neither verdict.
- **Splits.** The generator seeds are disjoint (gen.log: train 44, val 11, test 22; worker seeds are seed x 1000 + j).
  There are 0 duplicate boards between test and train, val and train, or test and val. All 2,000 test boards are
  unique.

### 5. Data sanity: OK, 1 ISSUE

- **Solvable fraction.** Test 857/2000 = 42.85%, val 43.55%, train 42.35%, which matches the 857 in gen.log.
- **Re-run cross-check.** I re-ran check_exact.py at 04:12Z, writing to /tmp. The output is identical:
  - 4x4: max |diff| 2.6e-8;
  - 6x6: 300 boards brute-forced, max |diff| 2.9e-8, 0 negatives with solutions, 0 invalid outcomes.
- **Extra checks on all test boards.** All 1,143 unsolvable test boards have 0 completions and 0 direct peer clashes.
  All 857 solvable boards re-solve with matching nsol, and their marginals match to 2.9e-8. Mean nsol is 12.11
  (range 2-348).
- **ISSUE.** Clue counts are matched per board by construction but not by distribution.
  - Mean givens on test: solvable 13.57, unsolvable 14.83. Train: 13.64 vs 14.89.
  - Clue count alone gives a Solvable LL of 0.658 against a prior LL of 0.683, which is a weak shortcut. It is
    irrelevant next to the measured P0 LLs, so the conclusions are unaffected.
  - RESULTS' "mean clue count is 14.3" is the mean over all boards, not over solvable boards as written.

### 6. Cost: cap holds (OK), 2 ISSUEs

Ledger lines are 1437-1586 of the shared GPU-LOG.md. The rate is $0.87/h, the on-demand ceiling.

| VM | event | ceiling cost |
|---|---|---|
| esud-a | created 00:54:44Z; still RUNNING at 04:10:51Z; age 3.27 h at 04:11Z | about $2.85 so far |
| esud-f | DELETE 04:08Z, age 2.00 h | $1.74 (from the ledger line) |
| esud-d | preempted within about 5 min; no DELETE line in the ledger | at most $0.10 |

- **Total ceiling spend: about $4.69**, against the $50 cap.
- The ledger `spend_usd overnight-sud` total is $28.00 (04:10Z). That figure counts esud-a's $4.00 reservation and
  reservations of $22.26 for VMs that were never billed (b, c, d, e x 2, first f). The cap holds either way.
- **ISSUE.** RESULTS says the never-billed reservations are $10.26. The correct figure is $22.26: it misses the 3
  failed creates at 02:04-02:06Z.
- **ISSUE.** esud-f was created with `model=STANDARD` (on-demand) per ledger line 1512, yet RESULTS and GPU-LOG.md say
  every VM was spot. For esud-f the ceiling rate is the real list price.
- **Live VMs** (`gcloud compute instances list --filter="labels.soup-run~esud"`, 04:10:51Z):
  `soup-gpu-esud-a-1015005430-29922`, us-central1-b, RUNNING, SPOT, labels {logical-soup: 1,
  soup-group: overnight-sud, soup-run: esud-a}.
  - All of its expected outputs are already in vmout/ (r1 and r1b-a both have DONE files), yet it keeps accruing about
    $0.87/h until its 6 h max-run, at about 06:55Z. I did not check whether anything is still running on it.

## What I could not check

- That the code on the VMs was byte-identical to the local hashed code. There are no hashes from the VMs.
- The original (hashed) content of run.sh, and the version of analyze.py from before 02:20.
- Exactly when 6x6 seed-0 training started relative to the 01:10 sud.py edit. The logs carry relative seconds only.
- Real billed spend. The Billing API is disabled, so all cost figures are rate-ceiling estimates.
- I did not re-run any model, because torch is not installed here. Every check works from the stored per-board
  outputs.
