# Item 2 (batch-2 follow-up): P10 bug check, results

**Status: No gain.** This follows the PREREG verdict rules: every Holm-adjusted p is 1.0. All gates passed.

Corrected 2026-09-30 ~01:55Z after the independent verifier, which recomputed every number from the raw files (all
match). The corrections are wording only: collapse timing in the warm-only arm, the "aged" auction group, the A/A
claim, and two post hoc details.

**Answer: yes, the learned events still collapse to constants.**

- With a centred init, a 4000-step rate warm-up, or both, 29 or 30 of the 30 learned cvae slots per arm end constant.
  The original init ends with 30/30 constant.
- From a centred init the slots go constant again within 50-100 steps, before the rate gradient is switched on (it is
  off for the first 4000 steps in the warm arms). With the original init and the warm-up alone it took up to 800 steps.
  No arm had the rate term off for the whole run (the norate arm was cut).
- So P10's collapse is not an artifact of the degenerate init or of the rate term: it happens from a non-degenerate init
  and before any rate gradient.

PREREG.md sha256 is `f29a87c1…a1b8`. `sha256sum -c PREREG.sha256` passes for PREREG.md, analyze.py, power.py, driver.py,
train_s6.py, auction.py and queue.txt. The run had 55 training jobs plus 20 step-0 jobs, and 0 of them failed.

## Gates (checked by `code/driver.py`, from `vmout/GATES.json`)

| gate | result | bar |
|---|---|---|
| G0: centred init is non-constant at step 0 (P10's test, 0.001, on test) | **30/30** slots, seeds 0-9. Original init: 9/15 constant on seeds 0-4, the same as P10, and 18/30 on seeds 0-9 | >= 80% |
| G1: A/A, the original init reproduces P10 | 9/15 constant at init. At the end, 0/15 slots are non-constant for cvae and 0/15 for auction | 7..11 of 15 at init; <= 1/15 non-constant at the end |
| G2: reg2 test NLL/cell | mean 0.551 (0.694, 0.694, 0.521, 0.422, 0.422). P10: 0.549 | mean in [0.35, 0.75], each seed <= 0.90 |

**A/A detail (seeds 0-4, this run minus P10, paired):**

- cvae NLL: -0.020 [-0.170, 0.129].
- auction NLL: -0.007 [-0.102, 0.087].
- reg2 NLL: +0.001 [-0.097, 0.100].
- Collapse: 0/15 non-constant here and 0/15 in P10, in both arms.

So the CPU rerun shows no clear difference from P10. This is not an equivalence test: the CIs are about ±0.1-0.15
nats/cell and single seeds differ by up to 0.17.

## Primary: fraction of the 3 learned cvae slots that are non-constant at the end

The test is on the test split with the 0.01 threshold. The CI is a seed-level t 95% CI over n = 10 paired seeds; the
p-value is the exact sign-flip permutation p, then Holm-adjusted over 3.

| arm | non-constant slots | mean frac | diff vs orig | perm p | Holm p | one-sided 95% upper bound per slot |
|---|---|---|---|---|---|---|
| cvae (orig) | 0/30 | 0.000 | — | — | — | 0.095 |
| cvae_init | 0/30 | 0.000 | 0.000 [0.000, 0.000] | 1.00 | 1.00 | 0.095 |
| cvae_warm | 1/30 | 0.033 | 0.033 [-0.042, 0.109] | 1.00 | 1.00 | 0.149 |
| cvae_both | 1/30 | 0.033 | 0.033 [-0.042, 0.109] | 1.00 | 1.00 | 0.149 |

**The one survivor** is the same in both warm arms: seed 0, slot 0.

- It is the non-cell bank atom `bl_row(0,4,0)` exactly: NMI 1.00.
- Its test G is 0.083 in cvae_warm and 0.060 in cvae_both, against a fee of 0.005.
- It was constant on val at steps 50-600 in cvae_both and then came back. So a collapse is not always absorbing.

## Secondary (descriptive)

**1. Collapse time.** This is the first val trace step (every 50 steps up to step 2000) at which all 3 learned slots are
constant.

| arm | median step | per-seed steps | never collapsed | collapsed before step 4000 (no rate gradient yet) |
|---|---|---|---|---|
| cvae | 50 | 50, 0, 50, 50, 0, 50, 0, 50, 50, 0 | 0/10 | — |
| cvae_init | 50 | all 50 | 0/10 | — |
| cvae_warm | 75 | 800, 0, 50, 100, 0, 750, 0, 100, 350, 0 | 0/10 | 10/10, but 4 (seeds 1, 4, 6, 9) were constant from step 0, so **6/10** |
| cvae_both | 50 | 50-100 | 0/10 | **10/10** |
| auction_both | 100 | 50-100 | 0/5 | **5/5** |

**Collapse does not need the rate gradient.** In the warm arms the encoder gets no rate gradient before step 4000; the
only gradient into it before then is the NLL through the straight-through estimator. Slots that started non-constant
went constant before step 4000 in cvae_both 10/10, auction_both 5/5 and cvae_warm 6/10 (its other 4 seeds had original-init
slots that were constant from step 0).

**2. Usefulness (test royalty G = gain_resample - rate, fee 0.005).**

| arm | seeds with a learned G > fee | best learned G, mean [95% CI] |
|---|---|---|
| cvae | 0/10 | 0.0000 [0.0000, 0.0000] |
| cvae_init | 0/10 | 0.0000 [0.0000, 0.0000] |
| cvae_warm | 1/10 | 0.0083 [-0.0104, 0.0270] |
| cvae_both | 1/10 | 0.0060 [-0.0076, 0.0196] |
| auction | 0/5 | -0.0034 [-0.0039, -0.0028] |
| auction_both | 5/5 | 0.0283 [0.0166, 0.0401] |

**3. Auction arms (seeds 0-4).**

- **Auction (orig).**
  - Non-constant at the end: 0.00. Respawns: 57 per seed, one per slot every 1000 steps.
  - Every learned slot is listed at the end, with a surviving G total of about -0.011.
- **auction_both.**
  - Non-constant at the end: **0.73 [0.39, 1.08]**, 11/15 slots. Respawns: 6 per seed.
  - G > fee in 5/5 seeds, even when only non-constant slots are counted.
  - Surviving G total: 0.014 to 0.064 per seed.
- **The pre-registered "aged" fraction is undefined (n = 0), for both arms.** It covers slots whose last (re)spawn was at
  least 7000 steps before the end. After its 7000-step learned grace, the auction checks a slot every 500 steps and
  respawns it only if its average is below the fee. Every auction_both slot failed that check at 7000 and again at
  14000, so every live slot was 6000 steps old at the end. That was the outcome, not forced by the design. See deviations.

**4. Test NLL/cell (e_hat), paired.** It is underpowered, as PREREG says.

| comparison | difference [95% CI] | n |
|---|---|---|
| cvae - reg2 | -0.034 [-0.185, 0.118] | 5 |
| cvae_init - cvae | -0.044 [-0.119, 0.031] | 10 |
| cvae_warm - cvae | -0.044 [-0.165, 0.077] | 10 |
| cvae_both - cvae | -0.002 [-0.112, 0.109] | 10 |
| auction - reg2 | -0.113 [-0.253, 0.027] | 5 |
| auction_both - reg2 | +0.004 [-0.190, 0.198] | 5 |

## Post hoc, not pre-registered: why do the auction_both survivors survive?

The source is `posthoc.py` → `results/posthoc.txt`. The survival depends on when the encoder was born.

| spawn epoch | encoder LR at spawn (cosine) | never constant on val |
|---|---|---|
| step 0 | 2e-3 | 0/15 (all constant within 50-100 steps) |
| step 7000 | 1.45e-3 | 0/15 (13/15 constant within 500 steps, all 15 by 5000) |
| step 14000 | 4.1e-4, falling to 0 at the end | **10/15** |

- **Most step-14000 survivors did move.** In 9 of the 10, val p1 changed by 0.05-0.34 in their first 4000 steps;
  the other one did not move at all. Some match bank atoms: seed 3 slot 1 is `eq(17,5)`, a copy of output cell 17, at
  NMI 0.96. (Seed 3 slot 2, `bl_col(1,5,3)` at NMI 0.96, is non-constant at the end but was constant on val at steps
  14500-15500, so it is not one of the 10 never-constant slots.)
- **Two of the step-14000 slots collapsed as the rate gradient ramped in** (ages 4500-5000). In seed 1, slot 2 had p1
  of about 0.28 until step 18000, then 0.16, then 0.0.
- **The rate term never had long to act.** The survivors ended at age 6000. Their rate gradient ramped in over their
  last 2000 steps and reached full strength only at the end, at an LR near 0.
- **Consistent with:** collapse is an early-training effect of the straight-through NLL gradient at a high LR against a
  young decoder. It is not an effect of the init or the rate term, and the rate term may cause a second collapse later.
- **Not tested:** the step-14000 result mixes up a low LR, a trained decoder and a young slot. It is not evidence that
  any of the fixes works.
- **A G artifact.** One slot has test p1 = 0.000 and still has G = +0.0074, because resampling from the prior gives
  values the decoder has never seen. This agrees with P10's verifiers: G measures decoder reliance, not information.

**Suggested next test (not run).** Give the encoders their own low LR from step 0 (for example 4e-4, the LR at step
14000), or damp the straight-through NLL gradient into them for the first N steps. Run long enough that every learned
slot is scored after a full rate phase (age >= 10000).

## Deviations

**Before the freeze (all in PREREG.md):**

- CPU instead of P10's L4 GPU. A/A equivalence was checked statistically, and step-0 p1 is bitwise equal.
- `s6.npz` was copied into `code/data/` (same sha256 as P10). It was trashed after the run: datasets live only under
  /tmp.
- Arms cut for CPU time: auction_init, auction_warm, auction and reg2 on seeds 5-9, norate and fix14.
- The G2 bar was widened to [0.35, 0.75] before the freeze.
- The val p1 trace records every 50 steps up to step 2000.
- The 17 fill jobs ran at the same time as the gate jobs.
- The pilot and smoke outputs (val only) are in `prefreeze/`.

**After the freeze:**

- There were no changes to code, arms, metrics or bars. The hashes verify.
- **The pre-registered "aged" auction secondary is empty (n = 0).** Every auction_both slot failed the auction's check
  at step 7000 and again at 14000 and was respawned, so none was 7000 or more steps old at the end. This is an outcome
  of the run, not forced by the design (a slot that passed at 7000 would have aged 13000 steps). It is reported as
  undefined.
- **"Seeds with a learned G > fee"** (the Gain bar, and secondary 1) counts constant slots too. It turns out G can
  exceed the fee for a constant slot (see above). This does not change any verdict: the primary fails before this bar
  applies, and for auction_both the count is 5/5 either way.
- `posthoc.py` and `results/posthoc.txt` were written after the test numbers. They are descriptive only.

## Cost

One spot c2d-highcpu-32 VM, trader-fu-p10-1, from 22:57:02Z to 01:27:49Z: 2.51 VM-hours. At the $1.20/h ceiling that is
**$3.02**, under the $4 cap. It includes about 14 min of pilot and smokes before the freeze.

The VM and its boot disk were deleted after a verified sync (284 files, 0 sha256 mismatches). `gcloud compute instances
list` and `disks list` with `--filter=name~trader-fu-p10` both return 0 items (checked at 01:28Z on 2026-09-30).

## Files

- `PREREG.md`, `PREREG.sha256`, `power.py`, `power.txt`: pre-registration and power.
- `analyze.py` → `results/summary.json` and `results/summary.txt`: the pre-registered analysis.
- `posthoc.py` → `results/posthoc.txt`: post hoc and descriptive.
- `code/`: P10's code plus `train_s6.py` (arms, centred init, warm-up, checkpoint, trace), `auction.py` (learned grace),
  `driver.py` (stages and gates) and `repro_init.py` (the 9/15 reproduction).
- `queue.txt` is the real run. `pilot_queue.txt`, `smoke_queue.txt` and `smoke_queue2.txt` were run before the freeze.
- `vmout/`: `results/` (55 m2 and 20 m2init result JSONs plus per-puzzle NLL arrays), `GATES.json`,
  `driver_status.json`, `DRIVER_DONE` and `logs/`.
  The mid-run checkpoints (`vmout/ckpt/`, 112 MB) were trashed after the analysis; `JOB_RUNNING` and `RUNNER.*` are stale
  markers from the VM.
- `prefreeze/`: pilot and smoke outputs (val only).
- `vm/LEDGER.md`: VM create, start, sync and delete records.
- `CATALOGUE-ENTRY.md`: the entry in the catalogue format.
