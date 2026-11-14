# P16 results: amortized in-context valuation of a newly listed bit

Status: **Problem** (pre-registered verdict: **inconclusive (undertrained): precondition not met**).

## Scope

- P16 tests a valuation tool. It tests **no economic mechanism**.
- A PASS would have supported a **valuation tool only**, and there is no PASS.
- P16 **does not bear on the Bansal falsifier**.

## Verdict in one paragraph

The meta-trained reader FR (axial transformer, 1.46M parameters, 1e16 counted FLOPs, 3 seeds) converged. Its validation NLL improved by only 0.29-0.35% over the last 20% of steps. But it reaches an in-distribution AUROC at k = 16 of only **0.711** (seeds 0.719 / 0.719 / 0.696), below the 0.80 the precondition needs. So the held-out result is "inconclusive (undertrained)", not a falsification. FR-small (1e15) is 0.067 lower, so the run is also flagged **compute-limited**.

Read past the precondition (every rule below was frozen before training), FR does not beat the prior-free baseline: tuned sparse-parent BMA (C4) is better on the held-out primary slice at k = 16. FR − best baseline = **−0.040 [−0.061, −0.020]**, negative for every seed. F1 (prior-free Bayes suffices) and F4 clause 1 fire.

C4 wins through its reverse term. Its score is a forward prequential gain plus a tuned reverse conditional-independence term. On the forward gain alone, the quantity FR's V^preq amortizes, **FR beats C4**:
- held-out primary: 0.705 against 0.660;
- in distribution: 0.711 against 0.641.

The reverse term alone scores 0.738 on the primary slice. This split is post hoc (below).

Four times the meta-training compute moves seed 0 only from 0.719 to 0.732 ID (exploratory). So more compute alone will not close the gap cheaply.

## What ran

- **Arms.**
  - FR: seeds 0-2, F_meta 1e16.
  - FR-small: 1e15.
  - FR-direct: 2 seeds, 1e16.
  - C1, C4, C5, S1, S2, CONS.
  - E1: TabPFN v2 on a paired subsample.
  - L-all and L-train: library matchers.
  - Adversarial slice: 60 episodes, about 95 CMI-0 candidates each.
- **Episodes.**
  - Test: 6000 episodes, 20 types × 3 strata × 100. The primary slice is 1200 held-out positives (HP) against 1200 held-out negatives (HN).
  - Tune: 504 ID episodes.
  - Pilot/gate: 1008 episodes.
- **Simplifications** (binding, listed in PREREG):
  - Dropped: C3, S3, FR-fact and CRT debiasing. F5 is therefore untestable.
  - F_meta is 1e16, not the spec's larger budget.
  - The test set is 6000 episodes, not 18k.
  - E1 ran on a subsample: 288 held-out + 120 ID episodes, k = 16 only.
  - The adversarial slice was reduced.
  - Stochastic baselines ran one seed each.
  - Not run: generator shift, menu set, the length check and V^KL.
- **Gate (Addendum A).** It passed. L-all − max(C4, C5) = +0.233 [0.204, 0.263]; on the E1 subsample, +0.200 [0.139, 0.263]; mean novelty 0.194.

## Main numbers (test, AUROC at k = 16; FR is the mean of 3 seeds)

| arm | ID | held-out primary | primary at k = 64 | primary at k = 128 |
|---|---|---|---|---|
| FR (1e16) | 0.711 | 0.705 | 0.884 | 0.928 |
| FR-small (1e15) | 0.644 | 0.570 | 0.702 | 0.752 |
| FR-direct (2 seeds) | 0.625 | 0.560 | 0.607 | 0.634 |
| C4 (tuned BMA, fwd + rev) | **0.777** | **0.745** | 0.885 | 0.921 |
| C5 (CI betting) | 0.614 | 0.649 | 0.820 | 0.893 |
| S1 / S2 (SGD readers) | 0.590 / 0.615 | 0.571 / 0.592 | 0.718 / 0.746 | 0.786 / 0.824 |
| C1 (KT counts) | 0.643 | 0.625 | 0.730 | 0.767 |
| E1 (TabPFN v2, subsample) | 0.575 | 0.599 | - | - |
| L-train (matcher, training types) | - | 0.766 | 0.840 | 0.856 |
| L-all (matcher, all types; oracle-like) | - | 0.971 | 0.999 | 1.000 |

## Decision rules

| rule | result |
|---|---|
| Gate | pass |
| Precondition | **fails**: ID 0.711 < 0.80; converged yes; compute-limited yes (FR-small gap 0.067) |
| (a) FR − max baseline, primary, k = 16 | fails: −0.040 [−0.061, −0.020] vs C4; E1 subsample −0.037 [−0.094, +0.016]; every seed negative |
| (b) ID >= .85 and primary >= .80 | fails (0.711 / 0.705) |
| (c) k90 | fails: every arm censored (no arm reaches 90% TPR at <= 10% FPR by k = 128) |
| F1, prior-free Bayes suffices | **fires**: FR is within 0.02 AUROC and 0.05 Spearman of max(C4, C5, S1, S2) at k = 8, 16 and 32 (it is below at each) |
| F2, library matcher | does not fire, because the precondition is unmet. Descriptively FR 0.705 < L-train 0.766 |
| F3, TabPFN suffices | does not fire: E1 0.599 vs FR 0.729 on the paired subsample |
| F4, hallucinated structure | **fires** (clause 1). FR's FPR at 80% TPR on pooled held-out nulls is 0.63 (bar 0.15); for comparison C4 0.54, C5 0.72, S2 0.75. Clause 2 does not fire (below) |
| F5 | untestable (S3 dropped) |
| F6, the pay is not the ingredient | does not fire: FR-direct 0.560 vs FR 0.705. But FR-direct had not converged (validation still improving 4-10% over the last 20%), so this is weak |

## Reported diagnostics (not part of PASS)

- **Settlement estimator favours FR.** Scored by realised log-score gain on the next 16 rows, FR's primary AUROC is 0.775 against C4's 0.711 (+0.064). This is the one pre-registered comparison FR wins.
- **FR's value is calibrated in nats; C4's is not.**
  - FR: the slope of the next-16-row settlement gain on V_16/16 is 1.05-1.10 (R² 0.41-0.45).
  - C4: slope 0.27 (R² 0.39).
  - So FR's V is a price in the right units; C4's ranks well but overstates the gain about 4x.
- **Winner's curse on null bits (adversarial slice, 60 episodes).**
  - When FR picks its highest-valued of about 95 CMI-0 candidates, that bit settles at −6.14 nats [−9.55, −3.77], against −0.83 for the average candidate.
  - C4's pick: −0.95 [−1.46, −0.50]; C5's: −0.98.
  - So F4 clause 2 (CI above 0) does not fire. But FR's top-valued null bits are the ones that cost most when used: ranking on FR's V over many candidates selects its own noise.
- **Where FR loses** (ID positives against all ID negatives, k = 16, FR vs C4):

  | ID positive type | FR | C4 |
  |---|---|---|
  | T1 | 0.62 | 0.82 |
  | T2 | 0.74 | 0.82 |
  | T4 | 0.68 | 0.77 |
  | T5 | **0.81** | 0.75 |
  | T6 | 0.64 | 0.77 |

  - By CMI bin (0.05-0.1 up to >= 0.4 bits): FR 0.60 → 0.83, C4 0.67 → 0.92.
  - By |B| = 2 / 6 / 12: FR 0.74 / 0.70 / 0.69, C4 0.80 / 0.78 / 0.76.
  - FR beats C4 only on T5.
- **Held-out types.** FR − L-train is negative on 7 of 8 held-out types (HP3a +0.04 is the exception). Fraction of oracle (primary): 0.44.
- **Other diagnostics.**
  - Spearman(V_64/64, CMI) on informative bits: FR 0.55, C4 0.62.
  - Consumer Spearman (V_16 vs CONS gain): FR 0.38, C4 0.39. On T3b: FR −0.01, C4 −0.06.
- **Cost per valuation.**
  - FR: 0.061 s per episode, all 6 k values; C4 is the same, 0.061 s.
  - TabPFN: 0.73 s per call, about 36 s per episode at k = 16.
  - The L matcher: 6.5 s per episode.
  - Meta-training cost 1e16 FLOPs per seed, about 47 min on one L4.

## Exploratory (PREREG Addendum B; enters no decision rule): the compute curve

Every number in this section is post hoc; it is in `results/explore_compute.json` and `charts/compute_curve.json`.

**FR seed 0, retrained from scratch at F_meta = 4e16** (19660 steps). It was preempted twice, and resumed from synced checkpoints at steps 4970 and 16211.

| F_meta | ID AUROC k = 16 | primary k = 16 | primary k = 64 | final val NLL |
|---|---|---|---|---|
| 1e15 (FR-small) | 0.644 | 0.570 | 0.702 | 0.534 |
| 1e16 (FR s0 / mean of 3) | 0.719 / 0.711 | 0.701 / 0.705 | 0.883 / 0.884 | 0.465 |
| 4e16 (FR s0) | 0.732 | 0.715 | 0.891 | 0.453 |

- **Diminishing returns.** 1e15 → 1e16 adds 0.075 ID AUROC per decade; 1e16 → 4e16 adds about 0.022 per decade.
- **Distance to the precondition.** A crude log-linear extrapolation puts ID 0.80 about three decades further, near 5e19 FLOPs.
- **Still behind C4.** At 4e16 FR is still below C4 on the held-out primary slice at k = 16 (0.715 against 0.745).

**The forward/reverse split of C4** (test set, k = 16; FR is the mean of 3 seeds at 1e16):

| score | ID | held-out primary |
|---|---|---|
| C4, forward gain only | 0.641 | 0.660 |
| C4, reverse CI term only | 0.776 | 0.738 |
| C4, tuned (forward + reverse) | 0.777 | 0.745 |
| FR (forward gain) | 0.711 | 0.705 |
| rank average, FR + C4 reverse | 0.784 | 0.758 |
| rank average, FR + tuned C4 | 0.781 | 0.759 |

- **FR adds information to C4.** Averaging FR's rank with C4's reverse term beats tuned C4 at every k >= 8 on the primary slice:
  - k = 32: 0.854 against 0.828;
  - k = 64: 0.908 against 0.885;
  - k = 128: 0.942 against 0.921.
- **A hypothesis for a new pre-registration.** A reader that also amortizes the reverse direction ("does b become predictable from (B, y)?") may close the gap. This is untested.

## Operations

- **VMs.** 13 L4 spot VMs (trader-b3-p16-1 ... -13). 10 were preempted: mostly in us-central1-c within minutes of creation, plus a 02:32Z wave and the exploratory run twice. One resume hit a zone stockout and was retried in us-central1-a. Every plan resumed from markers and the last synced checkpoint. The helper chain trainB2 (added by hand) ran trainB's own baseline steps in reverse, to finish sooner.
- **Failed step.** One step failed and was re-run: tune_C4, a CUDA OOM beside FR-direct training.
- **Cost.** $9.82, estimated at the $0.71/h rate ceiling (about 13.8 L4 spot VM-hours). The cap was $15, with the watchdog at $13. No paid APIs were used.
- **Teardown.** Every VM and disk is deleted.
- trader-b3-p16-13 was deleted after the final sha256-verified sync.
- `gcloud compute instances list` and `gcloud compute disks list` (project GCP_PROJECT, filtered on `trader-b3-p16`) both returned 0 items (checked 2026-09-30 about 06:22Z).
- The watchdog is stopped.

## Files

- `PREREG.md`: frozen, with sha256 of the first 12541 bytes in `PREREG.sha256`, plus Addenda A and B.
- Code: `gen.py`, `oracle.py`, `reader.py`, `train.py`, `baselines.py`, `larms.py`, `e1.py`, `eval.py`, `adv.py`, `analyze.py`, `analyze_final.py`, `analyze_x.py`, `tests.py`, `run_vm.sh`, `startup.sh`, `gpu.sh`.
- `results/summary.json`: the pre-registered analysis.
- `results/explore_compute.json`: exploratory.
- `charts/{auroc_vs_k,fpr_at_80tpr,calibration,adversarial,compute_curve}.json`.
- Raw per-episode scores: `results/vm_out/eval/` and `results/vm_out/x/eval/`.
- Training histories: `results/vm_out/runs/*/train.json` and `results/vm_out/x/runs/FR_s0/train.json`.
- Final reader weights (7 `final.pt` files, about 40 MB) are archived at `/home/agent/data/archive/p16-weights/`. Intermediate checkpoints were trashed.
- Commands:
  - `P16_OUT=results/vm_out python3 analyze.py final`
  - `P16_OUT=results/vm_out python3 analyze_x.py`
