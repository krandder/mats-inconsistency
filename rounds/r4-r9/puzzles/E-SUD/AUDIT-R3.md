# E-SUD round 3: independent audit (AUDIT-R3)

Auditor: board investigator subagent, model Opus 5.5 (claude-opus-5-5[1m]). Checks ran 2026-10-15 10:10-10:15Z.
The auditor's only write in the puzzle tree is this file. Scratch is under /tmp/esud-audit3. The scorer was reimplemented in
numpy without importing r3.py.

**Bottom line: SUPPORTED WITH MINOR CORRECTIONS.** The hard split matches the spec. All 1,000 unsolvable boards
were rechecked with an independent DFS and have zero completions. All have no peer clash, and the spec's verbatim
prop_depth is >= 3 on every one. All 1,000 solvable boards have 2-256 completions. The split has 10-15 givens, is
50/50 and has no val/test overlap. My scorer reproduces R3-ANALYSIS.json exactly: every fit (0 of 162 fit
parameters differ), R3-D1a, every R3-D1b cut and CI, every R3-D1c AUC and every mass-vs-count cut. The one exception
is a float-level difference in a ratio that is already degenerate (see 2f). The verdicts hold as pre-registered:
D1a PASS (1.8448 nats), D1b PASS for all three selectors, D1c FAIL with 0 of 6 conditions passing. Hashes still
match. The pre-registration was hashed 45 s before the first chain file landed locally. sud.py meets the spec. The
VM is gone. Corrections needed in RESULTS-R3.md, none of which changes a verdict:
- (a) M = 1 mass/count ratio is 1.5-4.0x, not 2-6x;
- (b) the best null_floor_prop cut is +38.5% (tau 1.5 P3, M = 2, CI [-42, 119]), not "at best +12%";
- (c) the P1 per-chain recall figures (83/78/71%) are below the 3-seed means (85/80/73%);
- (d) PREREG-R3.md's header says "written ~09:50Z", but the file's mtime is 09:39:45Z, after its own content and
  before the chains;
- (e) unresolved solvable counts at M = 4 go down to 0, not 1.

## 1. Hard split: OK

Script: `/tmp/esud-audit3/split.py`. It runs the spec's `prop_depth`, extracted verbatim from
overnight-followups-round-3.md into `specpd.py`, and my own MRV DFS (capped at 401), on **all** 2,000 boards (not a
sample).

| check | val | test |
|---|---|---|
| n / solvable / unsolvable | 1000 / 500 / 500 | 1000 / 500 / 500 |
| givens k range (recount = stored k) | 10-15 (yes) | 10-15 (yes) |
| unsolvable with >0 completions (own DFS) | 0 | 0 |
| unsolvable with direct peer clash | 0 | 0 |
| unsolvable with spec prop_depth < 3 | 0 (recomputed = stored pdepth) | 0 (recomputed = stored pdepth) |
| unsolvable depth histogram | 3:406, 4:42, 5:10, 6:2, 7:1, 8:1, 999:38 | 3:392, 4:69, 5:8, 999:31 (matches claim) |
| solvable completions min / max / median (= stored nsol) | 2 / 256 / 9 | 2 / 240 / 9 |
| solvable dead_end_rate min / mean | 0.30 / 0.393 | 0.30 / 0.394 |
| val/test board overlap; in-split duplicates | 0; 0 / 0 | |

- dead_end_rate (r3hard.py) is 10 random-MRV rollouts that pick a minimum-candidate cell (ties at random), fill a
  uniformly random legal digit, and count a dead end when some empty cell has zero candidates. This matches the spec's
  "10-trial random MRV rollout hits a dead end". It is reasonable.
- The spec's alternative clause "or prop_depth == 999" would have admitted every solvable board: all 500/500 have
  prop_depth 999, since singles cannot finish a multi-solution board. The implementation used only the stricter
  dead-end filter. OK.
- ISSUE (minor, already disclosed in the prereg): the spec asks for "matched clue counts", and they are not matched.
  Test means are 11.66 for solvable and 12.21 for unsolvable boards, and AUC(-k) on test is 0.61. P0 still fails
  badly on unsolvable boards, so this does not undercut D1a.
- Not met (disclosed): the spec's 20-60% per-chain recall target. Measured recall is 71-98%; see 2e.

## 2. Independent scorer: OK, numbers match

Script: `/tmp/esud-audit3/score.py`; output `/tmp/esud-audit3/mine.json`; comparison `/tmp/esud-audit3/cmp.py`.
- Refuted means the model's own check < 0.5.
- C^bel is the sum over distinct refuted completions (by Z bytes) of exp(max lb), capped at 1 - 1e-9, with a running
  prefix over chains.
- The floored likelihood is (1 - eta) + eta (1 - C)^theta, applied as a logit shift on p0; a found witness gives
  max(p0, best check).
- The count control is (1 - eta) + eta exp(-kappa M).
- Fits are argmins over the PREREG grids on val only, pooled over M in {1, 2, 4, 8, 16}, with log loss clipped at
  1e-6.
- The mixed bag interleaves rand chain j and conf chain j, j = 0..7.
- Seed-t uses t = 4.303.

**a. Fits.** All 27 condition-seed fits are identical to R3-ANALYSIS.json (theta_bel, eta_bel, theta_prop, eta_prop,
kappa, eta_count). theta_bel = 1000 in 25 of 27 fits; the other two are tau 1 rand seed 2 (223.9) and tau 1 voi
seed 2 (112.2). "Nearly every fit at the grid edge" is accurate.

**b. R3-D1a: PASS.** P0 log loss on unsolvable test boards is 1.9662 / 1.9404 / 1.6279, mean 1.8448 (>= 0.50).
Solvable: 0.019. p0s is identical across all 9 condition files per seed.

**c. R3-D1b (tau 1, M = 8): PASS, all three selectors.** Mean cut, seed-t CI:

| selector | unsolvable vs P0 | unsolvable vs witness | solvable vs P0 |
|---|---|---|---|
| rand | 0.9938 [0.9899, 0.9977] | 0.9938 [0.9899, 0.9977] | 0.9999 [0.9999, 1.0] |
| voi | 0.9958 [0.9859, 1.0056] | 0.9958 [0.9859, 1.0056] | 0.9999 [0.9999, 1.0] |
| conf | 0.9964 [0.9886, 1.0042] | 0.9964 [0.9886, 1.0042] | 0.9999 [0.9999, 1.0] |

- Witness-only equals P0 on unsolvable boards because there are 0 false witnesses: false_witness_neg = 0 in all 27
  condition-seeds at M = 16.
- Unsolvable null_floor log loss at M = 8, per seed: rand 0.0155 / 0.0093 / 0.0096; voi 0.0155 / 0.0093 / 0.0;
  conf 0.0 / 0.0093 / 0.0096. The 3-seed means are 0.006-0.012, as claimed.

**d. R3-D1c: FAIL, 0 of 6 (matches).**

| condition | AUC M=1 / M=2 (mine = theirs) | part1 | mass-vs-count M=2 | M=4 | part2 |
|---|---|---|---|---|---|
| tau1.5 rand | 0.7583 / 0.8501 | F | -0.1889 [-0.81, 0.43] | -0.0303 [-0.18, 0.12] | F |
| tau1.5 conf | 0.8742 / 0.9285 | T | 0.0482 [-1.61, 1.70] | -4154 (theirs -4101) | F |
| tau1.5 mixed | 0.7583 / 0.8559 | F | -0.2655 [-1.56, 1.03] | 0.0786 [-0.53, 0.69] | F |
| tau2 rand | 0.7103 / 0.8141 | F | -0.5014 [-1.43, 0.42] | -0.4433 [-1.47, 0.59] | F |
| tau2 conf | 0.8683 / 0.8913 | T | -1.215 [-2.52, 0.09] | -0.1198 [-1.24, 1.00] | F |
| tau2 mixed | 0.7103 / 0.8664 | F | -1.6319 [-4.94, 1.68] | -1.5007 [-5.51, 2.51] | F |

- The tau 1.5 conf M = 4 difference (-4154 vs -4101) is float precision on a near-zero count-floor log loss: the
  seed 0 and seed 2 denominators are about 1e-6. It does not affect the verdict, since the CI includes 0 either way.
- I checked every cell of the RESULTS totals table (M = 1, 2, 4, mass vs count, per seed). All match to the shown
  precision.
- The AUC table's unresolved-solvable counts also match:
  - M = 1: 102-123, 20-34, 138-153, 46-56;
  - M = 2: 26-31, 2-8, 48-53, 8-14.

**e. Corrections to RESULTS-R3.md text (ISSUE, minor; no verdict changes).**
1. "At M = 1 it is 2-6x worse": the per-seed mass/count ratio at M = 1 ranges from 1.52 (tau 1.5 rand seed 2) to 3.96
   (tau 2 conf seed 1). It should read 1.5-4x.
2. "null_floor_prop ... at best +12% at M = 4 for tau 2 P3": the largest mean is +38.5% (tau 1.5 conf, M = 2,
   CI [-42, 119]). The next are tau 1.5 mixed M = 4 at +12.2% and tau 2 conf M = 4 at +12.0%. No prop cut has a CI
   lower bound > 0, so the conclusion stands.
3. Per-chain recall: the 3-seed means over 16 chains are:
   - tau 1: P1 0.848, P2 0.975, P3 0.978;
   - tau 1.5: P1 0.801, P3 0.954;
   - tau 2: P1 0.729, P3 0.891.

   RESULTS gives P1 as 83 / 78 / 71%. Those are seed-1 values, not means. Either way the 20-60% regime was not
   reached.
4. "1-53 solvable unresolved per seed" at M = 2 and 4: the M = 4 range starts at 0 (for example tau 1.5 conf seeds 0
   and 2).
5. "By M = 4, 98.6-99.9% have a witness": that is true of the 3-seed means. The per-seed range is 98.2-100%.

## 3. sud.py chains(): OK

`code/sud.py` lines 154-212:
- Sampling uses `p = softmax(lg / tau)`. `lp`, which feeds into `logq`, is the log of the tempered p of the sampled
  digit.
- `pb = softmax(lg)[ar, z]` and `lpb = log pb[d]` feed into `logp_bel`. This is the untempered belief of the sampled
  digit, as the spec's item 3 requires.
- The conf and voi selection scores use the tempered p. This was pre-registered, and voi runs only at tau = 1.
- The check pass `fwd(model, x, -one)` with sigmoid has no tau, so it is untempered.
- `dump()` saves `<rule>_lb`. `ESUD_SPLIT_PREFIX` selects `s6_hard_*`.
- Note: exp(logp_bel) is the believed mass of a completion along that chain's cell order, so it depends on the order.
  The spec defines it this way. In the mixed bag, the same completion reached by P1 and by P3 keeps the max. That is
  a reasonable dedup choice.

## 4. Pre-registration timing and hashes: OK, with a header typo

At 10:13Z, `sha256sum -c PREREG-R3.sha256` gave OK for all 6 entries: PREREG-R3.md, code/sud.py, code/r3.py,
code/r3hard.py and both hard npz files. `results/sud.py.r3-frozen` is byte-identical to code/sud.py.

| file | mtime (UTC) |
|---|---|
| code/sud.py | 09:37:02 |
| code/r3hard.py | 09:37:19 |
| hard val / test npz | 09:38:00 / 09:38:22 |
| code/r3.py | 09:39:00 |
| PREREG-R3.md | 09:39:45 |
| PREREG-R3.sha256, sud.py.r3-frozen | 09:39:51 |
| first chain file (tau 1 rand) | 09:40:36 |
| last chain file (tau 1 voi seed 2) | 10:05:03 |
| R3-ANALYSIS.json / RESULTS-R3.md | 10:09:25 / 10:10:13 |

- The hashed r3.py is the scorer that produced the analysis, and it was frozen before any chain file existed
  locally.
- ISSUE (minor): the header of PREREG-R3.md says "written ~09:50Z", but its mtime is 09:39:45Z. RESULTS says it was
  hashed at 09:40Z, which agrees with the mtime. The header is a typo.
- Could not check: the chain start time on the VM. There is no local VM run log for esud-r3: the GPU-LOG has no RUN
  line for it, and vmout/ has no r3 dir. Local mtimes are copy times. The test data existed only from 09:38:22, so any
  pre-hash chain run would fall in a window of at most about 90 s. Tau 1 rand chains finishing 45 s after the hash
  is plausible.

## 5. Cost and cleanup: OK

GPU-LOG.md (/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md), lines 1645-1647 and 1659:
- PENDING-CREATE at 09:24Z;
- CREATE at 09:25Z: soup-gpu-esud-r3-1015092450-25443, us-central1-c, g2-standard-8, 1x L4, STANDARD (on-demand),
  rate ceiling $0.87/h, group overnight-sud;
- READY at 09:26Z;
- DELETE at 10:08Z: age 0.70 h, cost_usd 0.61 at the rate ceiling (0.70 x 0.87 = 0.609).

This is within the $5 cap. The log itself marks the rate "UNVERIFIED ceiling".

At 10:13:29Z, in project GCP_PROJECT, `gcloud compute instances list --filter='name~esud'` and
`gcloud compute disks list --filter='name~esud'` both returned empty, rc 0. The unfiltered instance list returned 23
rows, so the command works.

## Not checked

- The model checkpoints were not re-run, so chains were not regenerated (no torch here). Chain contents are taken as
  dumped.
- The chain start time on the VM; see section 4.
- Ground-truth refutation of chain completions. "Refuted" means the model's own check < 0.5, as pre-registered.
