# E-SUD round 2 audit (independent, read-only)

Auditor: board-opus subagent, running on Claude Opus 5.5 (claude-opus-5-5[1m]). Checks run 2026-10-15 05:35-05:40Z.

**Bottom line: supported with corrections.** I fitted and scored the chains myself, in numpy, without importing
r2.py (`/tmp/esud-audit2/indep.py`, `/tmp/esud-audit2/an.py`). All 15 validation fits match R2-ANALYSIS.json exactly
(theta, eta_mass, kappa, eta_count, theta_r1). All 9 R2-D1a cuts and both R2-D1b cuts match to 4 decimals: per-seed
values, means and seed-t CIs. Both verdicts reproduce: R2-D1a PASS (9/9) and R2-D1b FAIL (P1 and P3). Fits use
validation only. The code matches the spec. The tau handling in sud.py is as stated. Spend is under the cap, and no
esud VM or disk remains. Corrections:
(a) The "reproduces round 1b exactly" claim is overstated. The re-run chains are different random draws, and witness
rates differ from round 1b at M = 1-4. They agree at M = 8 only because witnesses saturate there.
(b) The code hashes in PREREG-R2.sha256 no longer match (this is declared), and the file is malformed.
(c) Small wording errors in deviation 2, the loss range, and the cost lines.
None of these changes a verdict.

## 1. Independent recomputation of R2-D1a / R2-D1b: OK

Method:
- Floored likelihood `L = (1-eta) + eta (1-C_M)^theta`, and count `L = (1-eta) + eta exp(-kappa M)`.
- On boards with no witness, `p = sigmoid(logit p0 + log L)`. A witness (own check >= 0.5) gives `max(p0, best check)`.
- The loss clip is 1e-6, as in round 1.
- Grid argmin of mean validation log loss, pooled over M in {1, 2, 4, 8, 16}, on the pre-registered grids (101 theta,
  62 kappa including 0, 11 eta).
- Test scoring: cut = 1 - LL_arm/LL_base. Seed-t CI with t = 4.303, ddof = 1.

Fits: 0 mismatches out of 15. My values: tau1 rand theta 141/1000/14.1, eta .99/.99/1. tau1 voi 199.5/251.2/44.7,
eta .999/.9999/1. tau1 conf 39.8/891/56.2. tau2 rand 794/501/56.2. tau2 conf 1000/398/35.5.

R2-D1a, tau = 1, M = 8 (mine = r2.py = RESULTS-R2.md):

| rule | comparison | per seed | mean | seed-t CI | verdict |
|---|---|---|---|---|---|
| P1 rand | pos vs P0 | .9999 .9999 .9999 | .9999 | [.9999, 1.0] | PASS |
| P1 rand | neg vs P0 | .9222 .8917 .9997 | .9379 | [.7995, 1.0762] | PASS |
| P1 rand | neg vs witness | same as neg vs P0 | .9379 | [.7995, 1.0762] | PASS |
| P2 voi | pos vs P0 | .9999 .8071 .9999 | .9357 | [.6591, 1.2122] | PASS |
| P2 voi | neg vs P0 / vs witness | .9898 .9972 1.0 | .9957 | [.9826, 1.0087] | PASS |
| P3 conf | pos vs P0 | .9999 .9999 .9999 | .9999 | [.9999, 1.0] | PASS |
| P3 conf | neg vs P0 / vs witness | 1.0 .9972 1.0 | .9990 | [.9951, 1.0030] | PASS |

- The VoI seed 1 cut on solvable boards goes from 0.3092 (round-1 unfloored refit on the new chains; round 1b's own
  eval JSON also gives 0.3092) to 0.8071, with eta_max = 0.9999. OK.
- Note, not an error: "vs P0, unsolvable" and "vs witness-only, unsolvable" are the same number in every case. No
  unsolvable board ever gets a witness (0 false witnesses in all 75 rule x seed x M cells), so the witness arm equals
  P0 on unsolvable boards. In effect there are 6 distinct comparisons, not 9.

R2-D1b, tau = 2, M = 4, mass vs count on all test boards (mine = r2.py = RESULTS):
- P1: per seed -0.2424, -2.6384, -0.0916. Mean -0.9908, CI [-4.5406, 2.5590]. FAIL.
- P3: per seed 0.7814, -0.0079, 0.9104. Mean 0.5613, CI [-0.6737, 1.7964]. FAIL.
- Other M: P1 M=1 +0.017 [-0.121, 0.155], M=8 -1.866. P3 M=1 +0.287 [-0.449, 1.024], M=8 -75.27. All match.
- Pooled LL table (tau = 2, M = 4): matches. Solvable boards with no witness at M = 4: P1 3/4/0 and P3 1/0/1 per seed.
  The large percentages come from these few boards.
- M = 8, tau = 1 LL table: all entries match (for example, VoI floored mass 0.00150/0.00017 and unfloored
  0.00537/0.00082).
- Witness rates at tau = 2: P1 .8405/.9572/.9973/.9996, P3 .9382/.9922/.9992/1.0. Match.
- Mean C_4 on unsolvable boards at tau = 2: P1 .257/.127/.420, P3 .398/.166/.573. Matches "0.13-0.42" and "0.17-0.57".
- Sensitivity check: how C_M merges duplicate completions. r2.py keeps the max log q over duplicates. I also tried
  first occurrence. Duplicates are common (tau1 conf 16,236 of 27,580 refuted chains at M <= 8; tau2 rand 1,951 of
  30,801). First occurrence gives identical fits and verdicts. The only change is the P1 neg lower bound
  (.7995 -> .7996).

Minor ISSUE (wording): "All of these losses are tiny (0.0005-0.006 nats per board)". For P3 seed 2 the losses are
0.00001 (mass) and 0.00015 (count). The range should start at about 1e-5.

## 2. Validation-only fits, pre-registration timing, hashes: OK, with ISSUEs

- **Validation only.** r2.py calls `fit(X['val'], r)`, and test arrays enter only `terms`/`arms` after the fit. My
  val-only fit reproduces every parameter exactly. OK.
- **Timing** (`stat`, 05:35Z):
  - PREREG-R2.md mtime is 04:19:10Z. PREREG-R2.sha256 is 04:19:49Z.
  - All 15 chain files are 05:15:07-05:29:36Z, and the rerun VMs were created at 05:05-05:10Z (GPU-LOG). OK.
  - The first VM (esud-r2) logged CREATE at 04:19Z and READY at 04:20Z, after the prereg. Its chains were lost.
  - The npz zip timestamps are fixed at 1980, so they say nothing about when the chains ran.
  - Minor ISSUE: the PREREG header says "written ~04:25Z", but the mtime is 04:19:10Z.
- **Hashes** (`sha256sum`, 05:35Z):
  - PREREG-R2.md matches (f0e4a97e...).
  - **code/sud.py does not match**: recorded eec72ac7..., now 119f83f0... (mtime 05:04:53Z).
  - **code/r2.py does not match**: recorded b0764d22..., now 6c836458... (mtime 05:04:53Z).
  - Both changes are declared in deviation 1.
- **ISSUE: PREREG-R2.sha256 is malformed.** Line 3 is a stray `date` output ("Thu Oct 15 04:19:12 UTC 2026"), and
  the r2.py line uses an absolute path. `sha256sum -c` will report an improperly formatted line.
- **Could not check:** the diff between the hashed 04:19Z code and the current code. I found no copy of the 04:19
  versions in the E-SUD directory; a broader `find` was stopped for time. Whether the change is "only file naming" is
  therefore unverified by diff. In the current code, the dump writes one file per rule (`a.rules` in the filename) and
  `load()` globs and merges, which is consistent with the stated change.
- Minor: the r2.py docstring still names the old `chains_tau{1,2}_s6_seed{s}.npz` layout.
- **Deviation 2 wording (minor ISSUE).** Exactly 2 mass fits sit at theta = 1000 (tau1 rand s1, tau2 conf s0). Two
  more are near the edge: tau1 conf s1 at 891 and tau2 rand s0 at 794. RESULTS says "3 of 15" and leaves out the
  794 fit.
- **Deviation 1 (generator restarts per rule).** This is accurate, and it breaks the prereg's "same generator seed
  and the same call order as round 1b". See section 4.
- **Checkpoints** (md5): r2models/model_s6_seed{0,1,2}.pt are identical to vmout/r1b-a/model_s6_seed0.pt and
  vmout/esud-f/model_s6_seed{1,2}.pt. OK.

## 3. Spec fidelity of r2.py and sud.py: OK

Spec: overnight-followups-round-1.md, Rank 1.
- **Likelihood:** the floored form and the count form are exactly as in the spec and the prereg. They are applied as
  an additive log-likelihood ratio on logit p0, which assumes P(null | unsolvable) = 1. OK.
- **Witness:** own check >= 0.5 gives max(p0, max check). OK.
- **C_M:** summed exp(log q) over distinct refuted completions among the first M chains, capped at 1 - 1e-9. When the
  same completion appears under different orderings, the max q_sigma is kept. The spec does not say how to merge
  these. It does not matter here (section 1).
- **Pooling and verdicts:** fits pool over M in {1, 2, 4, 8, 16}. The grids match the prereg exactly. The PASS rule
  (mean >= threshold and seed-t lower bound > 0) matches the prereg and the spec's "CI excluding 0".
- **sud.py `chains()`:**
  - `p = F.softmax(lg / tau, -1)` drives both the conf/voi entropy score and the multinomial sample.
  - `lp`, and therefore logq, is the log of the tempered pz. OK.
  - The check pass `fwd(model, x, -one)` -> sigmoid is untempered, and p0s in `dump` is untempered. OK.
  - Note: the VoI lookahead `qb` uses untempered softmax. That is irrelevant here, because VoI ran only at tau = 1.
- **Declared departure:** the spec asked to work "on the existing Round-1b chain outputs". Those were not saved, so
  the chains were re-run. This is declared and justified.

## 4. Reproduction claim: ISSUE (overstated)

The claim: "the re-run tau = 1 chains reproduce round 1b exactly where it can be checked".
- **What holds.** Witness-only LL (all boards) at M = 8 equals results/r1b/eval_s6_seed*.json to 1e-6 for all 9
  rule x seed cells: P1 .022374/.025034/.022575, P2 .022374/.025036/.022575, P3 same. Witness rate at M = 8 matches
  in all 9 cells, including VoI seed 1 at 0.9988.
- **Why that is weak evidence.** At M = 8 every solvable board but one (VoI seed 1) has a witness in both runs. The
  witness arm then equals P0 on unsolvable boards plus about 0 on solvable boards, so it matches whichever chains
  were drawn.
- **Where it fails.** At M = 1-4 the re-run differs from round 1b in many cells:
  - P1 seed 0, M=1: witness rate .9078 vs .9055, witness LL .023827 vs .022589.
  - P1 seed 1, M=2: .9802 vs .9872.
  - P2 seed 0, M=1: .9907 vs .9837.
  - P2 seed 1, M=4: .9977 vs .9988.
  - P3 seed 2, M=1: .9965 vs .9930.
- **Cause.** The chains are a fresh draw, not a bit-exact replay. This is consistent with deviation 1 (generator
  restarts per rule and per file) and with GPU nondeterminism. The prereg itself called the reproduction check
  "reported, not a rule". No verdict depends on it, because round 2 fits and scores only on the new chains.
- **Correction.** "Reproduction is distributional, not exact. M = 8 witness numbers match because witnesses saturate.
  Witness rates at M <= 4 differ by up to 0.7 points."

## 5. Cost and cleanup: OK, with a ledger ISSUE

Sources: GPU-LOG.md lines matching esud-r2 (grep 05:35Z), at a ceiling of $0.87/h.

| VM | time | ceiling cost |
|---|---|---|
| esud-r2 (spot) | CREATE 04:19Z, ~05:00Z preemption (no DELETE or PREEMPT line) | ~0.68 h, ~$0.59 (RESULTS: 0.75 h, $0.65, conservative) |
| esud-r2s0 | DELETE logged | 0.38 h, $0.33 |
| esud-r2s1 | DELETE logged | 0.46 h, $0.40 |
| esud-r2s2 | DELETE logged | 0.47 h, $0.41 (RESULTS: ~0.5 h, ~$0.43) |

- Total: $1.73-$1.79, under the $5 cap. Supported.
- **ISSUE:** GPU-LOG.md has no DELETE or PREEMPT line for soup-gpu-esud-r2-1015041904-11156. Any ledger sum built on
  DELETE lines misses about $0.6, and the 04:19Z reserve of $2.26 is never closed.
- **Cleanup:** `gcloud compute instances list --filter="name~esud"` returned "Listed 0 items." at 05:35:48Z, and
  `gcloud compute disks list --filter="name~esud"` returned "Listed 0 items." at 05:35:52Z. OK.

## Files

- Independent scorer: /tmp/esud-audit2/indep.py (argument `max` or `first` sets duplicate handling).
- Comparison: /tmp/esud-audit2/an.py.
- Pickles: /tmp/esud-audit2/indep_{max,first}.pkl.
