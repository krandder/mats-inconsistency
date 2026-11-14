# Item 3: P8's pricing rules against a gain-over-base truth at N = 4096

Question: at N = 4096, do R4 (bits saved minus bits to learn Z) or R8 (R1 − 0.0063·L(Z|X,S)) rank P8's 32 maze
items better than R1 (importance only), against the truth T = gain over base of the deployed Y net (bits/maze)?

## Status: Problem

A pre-registered gate failed. G3 on seed 2 gave a price noise ratio of **0.106**, above the 0.10 bar. The driver
stopped the run automatically at 2026-09-30 01:48:18Z (`vmout/GATE_FAILED.json`, `vmout/logs/queue.log`). No verdict
is given. Only seeds 0 and 1 pass every gate, and the pre-registered minimum is 3 complete seeds.

The numbers below are **descriptive only**. The analysis script scores seeds 0-2, including seed 2 that failed G3.
Applied to those seeds, the bars would read "No gain" (a joint bar: both the R4 and the R8 upper bounds are below 0.05)
and "R4 hurts". That label is not the status of this item.

Corrected 2026-09-30 ~02:10Z after the independent verifier, which recomputed every gate and headline number from the
raw replicate, truth and B2 files (all match) and confirmed that Problem is the correct pre-registered status. The
corrections are wording: verdict words in descriptive text, the DEVIATION-1 description, and the gate's sd convention.

## Gates

| seed | G1 B2 held acc (≥ 0.90) | G2 rho r0 vs r1 (≥ 0.80) | G3 noise ratio (≤ 0.10) | U2 random | U3 null | copyY R4 (reported) |
|---|---|---|---|---|---|---|
| 0 | 0.9736 pass | 0.992 pass | 0.073 pass | pass | pass | −969 |
| 1 | 0.9375 pass | 0.974 pass | 0.057 pass | pass | pass | −970 |
| 2 | 0.9688 pass | 0.977 pass | **0.106 FAIL** | pass | pass | −969 |
| 3 | 0.9824 pass | 0.966 pass | not reached (6 of 8 price reps done when stopped) | | | |
| 4 | 0.9395 pass | 0.973 pass | not run (no prices, DEVIATION-1) | | | |

- G3a (a job failing twice) never fired.
- The deadline (02:35Z) was not reached.
- Seed 2's G3 inputs:
  - R4(S=∅:g2) = 60,900 bits over 8 replicates;
  - the noise ratio 0.106 means sd over replicates ≈ 6,100 bits;
  - for comparison, seed 0 had R4 57,875 (ratio 0.073) and seed 1 had 60,958 (ratio 0.057).
- **The failure is marginal and depends on the sd convention.** The frozen `code/jobs3.py` (line 122) uses the sample sd
  (ddof = 1), and the driver reads that value: 0.1058. With ddof = 0 it would be 0.0990 and pass. The frozen rule is
  what counts, so the gate failed.
- PREREG projected 0.05-0.075 at KREP = 8 (scaled from P8's 0.04-0.06 at 12 replicates). One seed in three went above that.

## Descriptive numbers (not a verdict; seeds 0-2, seed 2 failed G3)

Primary contrast: d = τ_b(R) − τ_b(R1), per seed, and the mean over seeds. The CIs are t-CIs. The p-values come from
P8's one-sided paired permutation test.

| contrast | seed 0 | seed 1 | seed 2 | mean [97.5% CI] | p(R > R1) | p(R1 > R) |
|---|---|---|---|---|---|---|
| R4 − R1 | −0.484 | −0.532 | −0.528 | −0.515 [−0.611, −0.419] | 1.000 | 0.0002 |
| R8 − R1 | +0.004 | −0.004 | −0.024 | −0.008 [−0.060, +0.044] | 0.910 | 0.094 |

On seeds 0-1 alone (every gate passed), the same per-seed values give R4 − R1 of −0.484 and −0.532, and R8 − R1 of
+0.004 and −0.004. With n = 2 the CIs are too wide to mean much: the verifier's 97.5% t-CIs are −0.508 [−1.124, +0.108]
and +0.000 [−0.103, +0.103].

Kendall τ_b against T over the 32 items, mean over seeds 0-2 with a 95% t-CI:

| rule | τ_b | per seed |
|---|---|---|
| R1 importance | 0.770 [0.750, 0.790] | 0.762, 0.770, 0.778 |
| R4 one-code price | 0.255 [0.204, 0.307] | 0.278, 0.238, 0.250 |
| R8 (λ 0.0063) | 0.762 [0.745, 0.779] | 0.766, 0.766, 0.754 |
| R8o (λ 0.0631) | 0.773 [0.758, 0.788] | 0.778, 0.774, 0.766 |
| R3 final-block importance | 0.767 [0.742, 0.793] | |
| R2 learnability (R1 > 0) | 0.252 [−0.092, 0.596] | |
| R5 last-block saving | 0.246 [0.202, 0.290] | |
| R7b wrapper | not run (R7b was queued after all price merges; the stop came first) | |

- **Pre-registered noisy-copy check** (S = ∅, bits(nz05 pipeline) − bits(g2 pipeline)):
  - −3.96 [−6.97, −0.95] bits/maze (95% t-CI over the 3 seeds, including the failed seed). Under the pre-registered
    rule this would read "nz05 > g2", but it is descriptive here. On the 2 gate-passing seeds it is −3.59
    [−16.7, +9.5], a tie;
  - per seed: −2.56 ± 0.51, −4.63 ± 0.51 and −4.69 ± 0.48;
  - the new truth still rewards the noisy copy of Y, as P8's truth did;
  - R1, R3, R8 and R8o order the pair the way the truth does in 3 of 3 seeds, and R4, R2 and R5 in 0 of 3.
- **Per context** (secondary):

  | rule | S = ∅ | S = {g2, g4} |
  |---|---|---|
  | R1 | 0.652 | 0.867 |
  | R4 | 0.377 | 0.168 |
  | R8 | 0.652 | 0.879 |
  | R8o | 0.706 | 0.873 |

- **In-sample λ curve** (exploratory):
  - the argmax is λ = 0.0631, τ 0.773 (P8's oracle λ, found again);
  - λ = 0 gives 0.770 and λ = 1 gives 0.255.
- **Against P8's control truth** (free-latent GStack − pipeline): R1 0.664, R4 0.300, R8 0.677, R8o 0.685.
- **P8's N = 1024 prices scored against this N = 4096 truth** (seeds 0-2):
  - R1 0.774 [0.705, 0.844], R4 0.234 [0.194, 0.274], R8 0.766, **R7b 0.941 [0.886, 0.996]**;
  - R4 − R1 −0.540 [−0.620, −0.460] and R8 − R1 −0.008 [−0.018, +0.002] (95% CIs).
- **Truth**:
  - top items are S=∅:g4 22.4, g3 21.9, g5 21.3, g7 21.3, then g2g4:g6 20.6, S=∅:g6 20.6 and S=∅:nz05 20.4 bits/maze;
  - the d2/d4/d6 items sit at about 0.

**Reading (descriptive, not the item's verdict):**
- On the 3 priced seeds, R4 stays about 0.5 τ below R1 in each seed, as at N = 1024. That is consistent with more data
  not rescuing the learnability charge, but it is not a gated result.
- R8 at the fixed λ is indistinguishable from R1.
- The seed-2 noise failure is on one item's R4 scale (a 3.5% relative standard error of the mean). A re-run with
  KREP = 12 would be needed to turn this into a verdict.

## Deviations (all disclosed; no retuning after seeing test numbers)

1. **DEVIATION-1 (00:23:56Z, before any price or truth number was read; the G1 results of all 5 seeds had been seen).** The frozen driver was replaced by
   `driver2.py`. It adopted the 31 running jobs and:
   - priced seeds 0-3 only (seed 4 still ran B2 and truth, with its G1/G2 gates);
   - raised workers from 31 to 34;
   - queued R7b after all price merges;
   - moved the deadline from 02:20Z to 02:35Z.

   Why: price replicates took about 115 min, not the 67 min in the time model. See `DEVIATION-1.md`.

   Consequence: even without the gate failure, at most 4 seeds could have been scored (power 0.72 rather than 0.91).

   It is more than a scheduling change: it cut the priced seeds from 5 to 4 and moved the pre-registered deadline. It
   changed no gate formula, metric, bar or analysis code (verifier diff). Its 115-min estimate was pessimistic (seed 2's
   r5-r7 took 79-84 min). The status would be Problem either way, because the G3 failure is on seed 2.
2. The timing probe (seed 99) was cut short by a spot preemption at 23:10Z. The VM was recreated automatically as
   trader-fu-p8-1-r1. This is disclosed in PREREG, and seed 99 is excluded.
3. Pre-registered design choices, listed again for the catalogue:
   - KREP = 8 (P8 used 12);
   - 4 R7b replicates (P8 used 12);
   - R7a not run.
4. The shared watchdog did not sync this VM from 23:20Z to 00:13Z; the parent session fixed it at 00:15Z (`vm/LEDGER.md`).
   A manual verified sync ran at 00:13:37Z. No data was lost.
5. At the stop, seed 3 had 6 of 8 price replicates done. The remaining 2 were killed, and seed 3 has no price merge.
   Seed 3's replicate files are kept but not scored.

## Cost (at the $1.20/h ceiling)

| VM | span | hours | cost |
|---|---|---|---|
| trader-fu-p8-1 | 22:41:50Z to its preemption at 23:10Z | 0.47 | $0.57 |
| trader-fu-p8-1-r1 | 23:13:16Z to delete at 01:50:27Z | 2.62 | $3.14 |
| **total** | | | **$3.71** (cap $5) |

- `GPU-LEDGER.md` lines 269 and 271.
- **Note: `vm.sh all spend` under-counted this item as $0.57 (fixed 01:58Z).**
  - Cause: its `item_spend` awk matched the ledger's "WATCHDOG-FIX … backstop" line, which contained the upper-case
    delete keyword. That line has no `vm=`/`epoch=` fields, so it reused the previous VM name and set p8-1-r1's stop
    time to its own create time.
  - Item 2's ledger had the same line.
  - Effect: from 00:15Z to the end, the watchdog's item-cap and total-cap checks under-counted items 2 and 3. Realised
    spend stayed under both caps.
  - Fix: the line was reworded in both ledgers. `vm.sh all spend` now reports item3 $3.71 and a total of $7.41.
- VM teardown:
  - A verified sync (115 files, sha256) ran at 01:49:18Z, then `vm.sh item3 down` at 01:50:27Z.
  - gcloud listed 0 instances and 0 disks matching `trader-fu-p8`.

## Files

- `PREREG.md`, `PREREG.sha256` (frozen 2026-09-29 23:19:31Z) and `DEVIATION-1.md`.
- `code/`:
  - frozen: `driver.py`, `jobs3.py`, `analyze3.py`, `power3.py` and P8's patched `jobs.py`/`p8lib.py`;
  - `driver2.py` (DEVIATION-1);
  - `tables3.py` (reporting only).
- `vmout/`:
  - `results/`: 48 result files;
  - `GATES.json` and `GATE_FAILED.json`;
  - `logs/queue.log`;
  - `item3_summary.json`: descriptive, seeds 0-2. Its `"status": "No gain"` field is `analyze3.py`'s bar reading, which
    ignores gates. **The item status is Problem** (see `vmout/STATUS.txt`).
- `vm/LEDGER.md` (every VM create, sync and delete) and `vm/autoresume.log`.
