# Item 2 (batch-2 follow-up): P10 bug check, pre-registration

Written 2026-09-29, before any run that produces test numbers. Its sha256 is in PREREG.sha256, recorded before the VM
queue with the real runs starts. Everything after the freeze follows this file; each deviation goes into RESULTS.md.

## Question

P10's M2 learned-event arm (S6 6x6 puzzles, 8 slots: 3 learned encoders plus 5 reference slots) ended with every learned
event constant (0/15 non-constant at the end, in both the auction arm and the cvae arm). But 9/15 learned encoders were
already constant at step 0.

Is the collapse an artifact of that degenerate initialisation, or of the rate term acting before the encoder can learn
anything? To find out, the arm is rerun with (a) a non-degenerate encoder init, (b) the rate term off during a warm-up,
and (c) both. **Do the learned events still collapse to constants?**

## What was checked before this file (no trained model was evaluated on test)

- **P10's "constant at init" measurement was found and reproduced.**
  - p1 is the mean of (phi(sol, mask) > 0) on the test split (2000 puzzles). The model is built exactly as `m2` builds
    it (manual_seed(seed), S6(seed), Atoms(), M2). A slot is constant if min(p1, 1 - p1) <= 0.001.
  - `code/repro_init.py` gives the same p1 values as P10's `nmi.init` for seeds 0-4, to every printed digit.
  - Counts: 9/15 at 0.001 (P10's number), 8/15 at exactly 0 or 1, and 10/15 at 0.01.
  - Cause: at the original init the encoder output phi has mean about ±0.1 and an across-example SD of about 0.02, so
    its sign is the same for (almost) every input.
  - This is a step-0 reproduction of a published P10 number; no trained model was evaluated on test.
- **The init fix, on val only.** The centred init gives p1 ≈ 0.5 for all 15 slots (seeds 0-4).
- **Pilot observations (val only), which set expectations but not the design of the primary test.**
  - Smoke: auction_g200_both, WARM = RAMP = 100, 900 steps. All 3 learned slots were constant again by step 500.
  - Timing pilot: cvae_both and auction_both, 1000 steps, 16 seeds each, WARM = 4000. With the centred init and **no**
    rate gradient yet, the learned slots of the 4 runs inspected were already constant (p1 of 0 or 1 on 512 val
    puzzles) by step 500.
  - So collapse may not need the rate term. The val p1 trace is therefore recorded every 50 steps for the first 2000
    steps, and every 500 steps after that.
- **Code checks.**
  - Checkpoint/resume: a run stopped at step 599 and resumed wrote a result file identical, field for field (except
    timing), to an uninterrupted run, and the per-puzzle NLL array was identical too.
  - Driver control flow: a 1-seed, val-only smoke of `code/driver.py`, on General and on the VM, including a forced gate
    failure and a check that the stage-2 jobs start only after the gate check.
  - The pilot and smoke outputs (val only) are kept in `prefreeze/`, outside `vmout/`, which holds only the real run.

## Code

`code/` is P10's code (copied; P10/ is not edited) plus:

- `train_s6.py`: arm flags, centred init, rate warm-up, checkpoint/resume, the val p1 trace, and an env-selected eval
  split. With no new flag, an arm runs P10's code path.
- `auction.py`: a separate grace for learned slots.
- `driver.py`: stages and gates.

Data: P10's `s6.npz` (sha256 de06d1a5…720e7), copied to `code/data/` so a resumed VM has it.

## Arms (S6, T = 1, 20000 steps, fee 0.005, everything else as P10)

| token | meaning |
|---|---|
| (none) | original init, rate on from step 0: P10's arm, the A/A replication |
| `init` | **centred encoder init**. After construction, and again after every respawn, the last layer of each learned encoder is rescaled so that phi <- (phi - median) / SD on a 1024-example augmented train batch. The batch comes from its own generator (seed 1_000_003·seed + 1009·step + k), so the training data stream is unchanged. This is the data-dependent init of LSUV / weight-norm init. |
| `warm` | **rate warm-up**. For each learned slot, the rate term's gradient into its encoder is scaled by λ = clip((age - 4000) / 2000, 0, 1), with age counted from the slot's (re)spawn. The rate value, the prior and the auction's accounting are unchanged. In auction arms, learned slots get grace 1000 + 4000 + 2000 = 7000 steps before they can be de-listed. |
| `both` | init + warm |

Mechanisms: `cvae` (no auction; at the end, slots with val gain_resample <= rate are pruned) and `auction` (grace 1000,
respawn below the fee), exactly as P10.

Jobs (55 training runs plus 20 step-0 runs):

| arm | seeds | role |
|---|---|---|
| cvae (orig) | 0-9 | primary comparator; seeds 0-4 are gate G1 |
| cvae_init, cvae_warm, cvae_both | 0-9 each | primary |
| auction (orig) | 0-4 | gate G1, secondary |
| auction_both | 0-4 | secondary: the auction arm with both fixes |
| reg2 (decoder/baseline) | 0-4 | gate G2 |

**Cut from the first draft because of CPU timing** (see Compute):

- auction_init and auction_warm;
- auction orig and reg2 on seeds 5-9;
- the norate diagnostic (made redundant by the pilot: the warm arms have no rate gradient for 4000 steps, and their
  val trace shows whether collapse happens before then);
- the fix14 positive control, which the task asked for only "if cheap" and it is not. P10 ran it (3/4 seeds had G > fee).

## Seed count (power)

`power.py` → `power.txt`:

- In P10 the original arm had 0/15 learned slots non-constant (SD 0), so the paired SD is set by the fixed arm.
- **Assumed effect and SD.** A fix that works makes about half the slots non-constant: delta 0.5, per-seed SD 0.35 (a
  3-slot binomial at p = 0.5 has SD 0.29, plus seed variation). Alpha is 0.05/3 (Holm over 3 fixes).
- **Power.** 0.39 at n = 5, 0.79 at n = 8, 0.91 at n = 10. **n = 10 seeds for the primary.**
- **Secondary auction arm.** It gets P10's minimum of 5 seeds, which gives power 0.39 at the same effect; it is
  descriptive.
- **Null result.** If 0/30 slots are non-constant, the 95% upper bound on the per-slot rate is 0.095 per arm.
- **NLL is underpowered.** The P10 seed SD of (M2 - reg2) is 0.168 nats/cell, so detecting a 0.05 gap at 80% would need
  91 seeds. NLL is descriptive only.

## Primary metric and test

- **Per-seed metric.** The fraction of the 3 learned slots of the **cvae** arm whose final encoder is non-constant on
  the test split: min(p1, 1 - p1) > 0.01, with p1 = `nmi.live[k].p1`, over 2000 puzzles.
- **Why cvae.** cvae has no respawns, so a late fresh encoder cannot pass for a learned one.
- **Test.** For each fix F in {init, warm, both}: d_s = frac_F(s) - frac_orig(s), paired by seed over s = 0..9.
  - Effect size: mean d, with a seed-level t 95% CI.
  - p-value: the exact two-sided sign-flip permutation test over all 2^10 flips. If d is 0 for every seed, p = 1.
  - Holm correction over the 3 fixes, at family alpha 0.05.

## Secondary metrics (descriptive; no multiplicity claims)

1. **Usefulness.**
   - Per seed and arm, the number of learned slots with test royalty G = gain_resample - rate > fee (0.005). Also the
     best learned G, with a seed-level CI.
   - The max NMI of each non-constant learned event with the atom bank (`nmi.live`).
   - gain_vs_const and gain_resample.
2. **Auction arms** (orig vs both, seeds 0-4).
   - The non-constant fraction, both raw and restricted to slots whose last (re)spawn was at least 7000 steps before
     the end.
   - Respawns per seed, listed learned slots at the end, and surviving_G_total.
3. **Collapse time.** From the val p1 trace (512 held val puzzles): the first recorded step with all 3 learned slots
   constant at 0.01. Reported as a median over seeds, per arm.
   - For the warm arms, whether that step is before 4000 answers "does collapse need the rate gradient?".
4. **Test NLL/cell** (`test_nll_ehat_per_cell`), paired by seed. Each M2 arm is compared with cvae orig, and on seeds 0-4
   with reg2.
5. **A/A detail.**
   - Seeds 0-4, orig arms: per-seed collapse and NLL next to P10's.
   - P10 trained on an L4 GPU and this run is CPU-only, so equality is checked statistically, not bitwise. Step-0 p1 is
     bitwise equal.

## Gates (enforced by `code/driver.py`: it kills the running jobs, writes GATE_FAILED and exits 3)

- **G0 (stage 0, step-0 encoders, seeds 0-9).** The centred init is non-constant (P10's test: min(p1, 1 - p1) > 0.001
  on test) in >= 80% of the 30 learned slots.
- **G1 (A/A, seeds 0-4).** Both of these must hold:
  - the original-init step-0 constant count is in 7..11 of 15 (P10: 9/15);
  - both cvae and auction on the original init have <= 1 of 15 learned slots non-constant at the end (0.01 threshold;
    P10: 0/15).
- **G2 (seeds 0-4).** reg2 test NLL/cell is finite, with a mean in [0.35, 0.75] and every seed <= 0.90. P10: mean 0.549,
  seeds 0.30-0.70, seed SD 0.164, so the bar is about ±2.7 SE.

**Gate order.**

- The 15 gate jobs start first. The first 17 primary jobs (cvae_init seeds 0-9, cvae_both seeds 0-6) fill the VM's
  otherwise idle slots at the same time; they add no VM time.
- No other job starts until G1 and G2 pass.
- A failed gate means the run stops and nothing more is spent: the VM goes down, and the status is **Problem**.

## Verdict rules

| status | condition |
|---|---|
| **Problem** | a gate failed, or more than 10% of the non-gate jobs failed after 3 tries each |
| **Gain** | some fix F has a Holm-adjusted p < 0.05, mean frac_F >= 0.5, AND in at least 5 of 10 seeds a learned slot of that arm has test G > fee. That is, the collapse was an artifact and the fixed arm learns useful events. |
| **Informative** | some fix has a Holm-adjusted p < 0.05 but misses the Gain bar: the events stay non-constant but are useless (G <= fee), or the effect is only partial (mean < 0.5) |
| **No gain** | no fix differs from orig (all Holm-adjusted p >= 0.05). Collapse is not an init or warm-up artifact, and P10's finding stands. The upper bound on the per-slot non-collapse rate is reported. |

RESULTS.md answers the question in one line whatever the status: do learned events still collapse with the fixes?

## Compute and budget

- One spot c2d-highcpu-32 VM (16 cores, 2 threads each), 32 single-thread processes, $1.20/h ceiling. Item cap: $4 at
  the ceiling.
- **Timing pilot** (val only, 32 processes): 0.217 s/step per process, so a 20000-step run takes about 73 min. P10 ran
  these on a GPU in about 8 min.
- **Plan.**
  - Wave 1 is the 15 gate jobs plus 17 fill jobs; wave 2 is the other 23 jobs after the gates.
  - That is about 2.4 h of VM, about $3.2 at the ceiling including setup and the pilot.
  - The watchdog deletes the VM if item spend reaches $4. Any job still unfinished then is reported as missing: no
    rerun, no replacement.
- The per-job checkpoint is every 2000 steps (about 7 min); the watchdog syncs every 5 min, and a preempted VM resumes.

## Analysis

`analyze.py` implements the sections above. It reads only `vmout/results/` and `vmout/GATES.json`, and it is written
before any result file with test numbers is opened. A seed that is missing from an arm is dropped from that arm's
paired comparisons, and the drop is reported.
