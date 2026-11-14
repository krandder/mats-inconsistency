# P1: multi-horizon anytime-value heads for halting a frozen recurrent net

**Verdict: No gain.**
- **T2m mechanism (Q1): No gain.** The multi-horizon head R7 halts within 1% of the oracle, 33-35% below the simple-rule envelope. The same head restricted to its H=1 and cap outputs (R4c) does just as well, so S2 fails and the F2 non-myopia kill fires. The head capacity gate also fails as specified, so this is formally not a clean falsification (see below).
- **Mazes (Q2): No gain (no headroom).** The headroom gate failed (0.059 at 33, 0.053 at 59; threshold 0.30).
- **Q3 (label-free KL settlement after test-time refit, mazes 59 and 99):** **No gain.** S3 fails at both sizes and the F3 kill fires at both. R9 ranks well (Spearman 0.79-0.81 at 59, 0.86-0.90 at 99) and is level-calibrated in its top quintiles, but its low quintiles are off by 3-50x, and it is 9-10pp less accurate than the labelled refit R7TT at half the oracle budget. Where the maze is still changing, KL movement is information (sufficiency holds: realized G about equal to K); the martingale condition fails everywhere (entropy drop is 1-4% of K).

Pre-registration, deviations and dated addenda A-I: `PREREG.md`. Every number below is recomputed by `analyze.py` (`results/p1_results.json`; chart data `results/p1_chart.json`). Intervals are 95% bootstrap intervals over 2000 resamples: draw a unit (backbone seed x head seed on T2m, head seed on mazes), then examples.

## Q1: T2m (plateau of varying length plus 50% irreducible instances)

**Setup**
- Backbone: MarketResNet w=64, trained with recipe v2 (deviations 9 and 9b: a length curriculum with stage holds, 8000 iterations).
  - 2 backbone seeds x 3 head seeds = 6 units.
  - Both backbones pass the validity gate: resolvable exact 1.00 at L64 and L128; blocked mean |p-0.5| 0.003-0.016.
- Pools: 500 examples per backbone seed at L64 and at L128; L256 (200 examples) is descriptive only.
- a* = 0.98 x the best fixed-T log-score gain.

**Mean steps to reach a\***

| pool | oracle | R7 multi-H | R8 KL | R4c {1,cap} | R13 Q-ACT | envelope | R4 one-step | R4e | R3 best | R11 | R12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L64 | 14.1 | 14.2 [13.0, 15.5] | 14.3 | 14.4 | 18.8 [13.1, 28.9] | 21.3 | 46.9 | 46.6 | 46.5 | 46.8 | 45.9 |
| L128 | 29.0 | 29.1 [26.7, 31.6] | 29.1 | 29.5 | 36.1 [27.1, 60.0] | 44.7 | 101 (a* not reached in some unit) | 92.4 | 92.1 | 92.7 | 90.5 |
| L256 (descr.) | 40.5 | 67.1 | 65.9 | 83.8 | 76.5 | 79.9 | 141 (not reached) | 231 | 223 | 225 | 175 |

**Savings of R7 at a\* (1 - cost(R7)/cost(rule))**

| | L64 | L128 |
|---|---|---|
| vs best competitor | 0.003 [-0.010, 0.018] | 0.009 [-0.011, 0.025] |
| vs R4c | 0.011 [0.000, 0.021] | 0.016 [0.000, 0.026] |
| vs envelope | 0.328 [0.270, 0.387] | 0.347 [0.264, 0.423] |
| vs R13 | 0.173 [-0.010, 0.485] | 0.142 [-0.011, 0.503] |
| vs R4 | 0.63 [0.31, 0.79] | 0.71 [0.27, 1.00] |

**S2, primary: FAILS on both pools**
- Saving of at least 20% over the best competitor: fails. The best competitor is R4c or R8; R7's saving is 0.3-0.9%.
- Fewer blocked-instance steps than R3: passes. R7 uses about 1 step on blocked instances; R3 runs to T_cap (64 or 128).
- Higher resolvable exact match than R4 at a*: fails. The difference is -0.21 [-0.32, -0.08] at L64 and -0.37 [-0.68, -0.15] at L128.
  - R4 almost never stops early, so it solves nearly every resolvable instance, but at 3.3x the cost.
  - At matched accuracy, every efficient rule (the oracle included) gives up the longest resolvable plateaus.
- R8 (label-free KL) within 10% of R7: passes. The ratio minus 1 is +0.001 [-0.005, 0.005] at L64 and -0.001 [-0.017, 0.011] at L128.

**F2**
- The non-myopia kill fires: R7 is within 10% of R4c on both pools (1.1% and 1.6%). The intermediate horizons add nothing here.
- The other F2 clauses do not fire: R7 is 17% and 14% cheaper than R13 (intervals touch 0) and about 69% cheaper than R3 and R11.

**Why R4c ties R7**
- R4c is itself within 2% of the oracle, so no halting rule could be 20% cheaper than it.
- The {1, cap} pair already separates the two slices:
  - on blocked instances the cap head predicts about 0 remaining value, so the rule stops at once;
  - on resolvable instances the cap head predicts the full remaining gain through the plateau.
- A plain myopic gate (R4, R4e) cannot see across the plateau and runs 3x longer.
- So the non-myopic part that matters is the cap (remaining-value) horizon, not the intermediate horizons.
- Out of distribution at L256 (descriptive, one pool), R7 is 13.5% [-15%, 59%] cheaper than R4c. The interval is wide and this was not tested.

**Head capacity gate (validity): FAILS as specified**
- Gate: Spearman >= 0.8 and predicted/realized mean ratio within 10% at H in {4, 16} on held-out training-distribution data.
- R7 at H=4: Spearman 0.46-0.53, mean ratio 0.79-1.20.
- R7 at H=16: Spearman 0.58-0.63, mean ratio 1.29-1.50.
- R8 at H=4: Spearman 0.46-0.78.
- Under the pre-registered rule, E-P1-1 is therefore "invalid rather than a falsification".

**Post-hoc diagnostic** (`capdiag.py`, `results/capdiag_t2m.json`; PREREG Addendum H; not a criterion)
- Computed on the L64 and L128 pools, splitting R7's predictions by instance type.
- **Blocked instances:** the realized gain is pure noise by design, so Spearman is about 0 (-0.04 to 0.01).
  - The heads correctly predict about 0 there: |pred| < 0.5, against 2.8-17 on resolvable instances.
  - This alone roughly halves the pooled Spearman.
- **Resolvable instances:** Spearman is 0.72 (H=4) and 0.81 (H=16) at L64, and 0.67 and 0.73 at L128.
  - Mean ratios are 0.99 and 0.96 at L64, 1.02 and 1.54 at L128 (the H=16 head over-predicts at L128).
- So the gate as written cannot pass on a task with an irreducible half, and even on resolvable instances the heads are only moderately good.
- They are still good enough to halt within 1% of the oracle, so weak heads do not explain the tie with R4c.

## Q2: mazes, headroom gate: FAILED (PREREG Addendum C)
- Backbone: Bansal et al.'s released DT-Recall checkpoint (w=128). The reproduction gate passes:
  - 13x13: 1.00 at t=51 (paper 99.88%);
  - 59x59: 1.00 at t=637 (paper 97.30%);
  - TF32 on.
- On dev, headroom = 1 - oracle/envelope in steps at a*:
  - 0.059 at n=33 (envelope R2:2:1 104.6 steps vs oracle 98.5);
  - 0.053 at n=59 (280.3 vs 265.4).
  - The threshold was 0.30.
- DT-Recall solves every dev maze by T_cap. A KL-lag-2 threshold (Huginn-style) is already within 6% of the earliest-exact oracle, so there is nothing for a VOC head to win.
- E-P1-2 is therefore "No gain (no headroom)". As pre-registered, only 59 and 99 were run, for Q3.

## Q3: label-free KL settlement after test-time refit (mazes 59 and 99)
**Setup**
- Backbone: the DT-Recall checkpoint as in Q2. Heads were trained on 9x9 trajectories, then refit label-free on the
  unlabelled pool of each size (2 epochs; deviation 2). 3 head seeds.
- Pools: size 59 (T_cap 708, dev 100, eval 500) and size 99 (T_cap 1188, dev 50, eval 200). Heads decide every 4 steps.
- Realized K_H = summed per-pixel KL(b_{t+H} || b_t) over the next H steps; G_H = realized log-score gain;
  entropy drop = H(b_t) - H(b_{t+H}). Bins are quintiles of predicted K; ratio = mean predicted / mean realized.
- Matched compute: exact match at budgets of 0.5x, 1x and 2x the oracle's mean pixel-steps.

**R9 (KL heads after refit) against the S3 clauses**

| | size 59 (N=500) | size 99 (N=200; H <= 32, see caveat) |
|---|---|---|
| Spearman(pred, realized K), per H | 0.79-0.81 (lower CI 0.77-0.79): fails | 0.87-0.90 (lower CI >= 0.86): passes |
| bin ratios, low 3 quintiles | 0.02-0.66 at H <= 8; 2.1-12.3 in the middle quintile at H=16-cap: fails | lowest two 0.04-0.33 at H <= 8 (middle 0.74-1.02); 1.5 and 5.0 at H=16, 32: fails |
| bin ratios, top 2 quintiles | 0.79-1.15 (cap 0.57) | 0.80-1.32 |
| K mass on steps with G < 0 | 13.8% [13.1, 14.6] at H=2; 0-2% at H >= 4: passes | 15.9% [14.9, 17.1] at H=2; 1.8-2.3% at 4 <= H <= 32: passes |
| R9 - R7TT exact at 0.5x / 1x / 2x oracle budget | -0.10 [-0.18, -0.02] / -0.32 [-0.46, -0.05] / +0.01 [-0.01, 0.05]: fails | -0.09 [-0.15, -0.01] / -0.12 [-0.22, 0.06] / +0.10 [0.01, 0.23]: fails |
| S3 | fails | fails |

**Kills and secondaries**
- F3 fires at both sizes: R9 is more than 5pp worse than R7TT at matched compute, and populated bins are off by more than 2x.
  Its other clauses (Spearman < 0.5, harmful share >= 20%) do not fire.
- F4 as worded (R12TT within 2pp and 10% of R9) does not fire, but only because the refit consistency probe R12TT is
  better than R9: +0.16 [0.11, 0.20] exact at 0.5x and +0.40 at 1x the oracle budget at 59, +0.06 [0.02, 0.12] and
  +0.20 at 99, and R9 needs 23% (59) and 12% (99) more pixel-steps at a*.
- F1 (the expected maze outcome) holds: no VOC arm beats the simple-rule envelope. At a* R9 needs 537 steps at 59 vs
  267 for the envelope (a KL lag-2 rule) and 252 for the oracle; at 99, 1001 vs 679 and 529. Holm-adjusted p = 1.0 for
  the pre-registered secondaries (R7, R8, R9s vs the envelope) at both sizes.
- Headroom at 99 (descriptive, not a gate): 0.22 [0.13, 0.27]. The a* is lower at 99 (0.89) because 18 of 200 mazes
  are never solved by T_cap; every rule spends 1116-1188 steps on them (R9 1145).

**Which condition fails (sufficiency vs martingale)**
- Top two quintiles of predicted K (the mazes still being solved): realized G / K = 0.89-1.21 at every H at both
  sizes (H <= 32 at 99), so movement there is information and sufficiency holds. The entropy drop is only 1.2-4.4% of K, so the
  martingale condition fails: beliefs move by large KL without losing entropy. A reading consistent with this (not
  tested): per-pixel beliefs are near-deterministic and flip from confident-wrong to confident-right.
- Low three quintiles (settled mazes): G is far below K and sometimes negative (for example at 59, H=2: K 0.6-1.0,
  G -0.41 to 0.50), so there sufficiency fails too: the net keeps moving without gaining information. By the
  lag-2 criterion 487 of 500 (59) and 182 of 200 (99) eval mazes end in a 2-cycle. The refit heads under-predict this
  oscillation KL by 3-50x in the lowest quintiles and, at 59, over-predict it by up to 12x in the middle quintile at long
  horizons; that is where the bin clause fails.
- R8 (KL heads without refit) is worse calibrated (bins up to 9.5x); the signed pseudo-label arm R9s ranks worse
  (Spearman 0.18-0.83). R7TT (settled on realized G) ranks worse than R9 (Spearman -0.16 to 0.83); its low bins are unstable because realized G there is near 0.

**Caveat: float16 overflow at 99 (PREREG Addendum I)**
- Stored K and head outputs are float16; at 99 about 6.6% of realized cap K exceed 65504 and are stored as inf, so
  H=64 and cap are not computable at 99 and the table uses H <= 32. Size 59 and T2m are unaffected.
- `sens_f16.py` reran size 99 with inf replaced by 65504 (the opposite bound; `results/sensitivity_f16_n99.json`):
  R9 990 vs 1001 steps at a*, R9 - R7TT -0.09 / -0.24 / +0.11, harmful share at most 15.9% at every H. The verdict
  is unchanged.

## Compute, preemptions and cost
- GPU: 7 spot g2-standard-4 + L4 VMs in GCP_PROJECT (trader-cat-p1-1 to p1-7), 6.91 VM-hours, rows in `../../GPU-LEDGER.md` (P1 section); local log `gpu/LEDGER.md`.
  - The watchdog (`gpu.sh watchdog`) synced every 15 min with sha256 checks.
  - It would delete a VM at an estimated $7, an age of 5.8 h, or 20 min idle.
- Cost: $4.90 of the $8 cap, estimated at the on-demand ceiling of $0.71/h (spot is cheaper). After the teardown at 20:18Z, gcloud listed no trader-cat-p1 instance or disk (re-checked 21:44Z).
- Six spot preemptions: 14:32Z, 15:32Z, 16:31Z, 16:50Z, 18:31Z and 19:02Z. Each time the VM and its disk were auto-deleted, and the run resumed on a new VM from the last watchdog sync plus a 150-second checkpoint pull.
  - After the second one, every long step writes partial checkpoints (Addendum G).
  - A GPU test showed that a resumed maze run is bit-identical to an uninterrupted one (max diff 0.0).
- The CPU ran unit tests (a)-(e), the analysis and the smoke tests; General's load was 55-120 on 16 cores throughout.

## Deviations (details in PREREG.md)
- Location: puzzles/P1.
- Streamed test-time fitting.
- Backbone curriculum v2 with stage holds, 8000 iterations instead of 10000 (deviations 9 and 9b).
- Head-target normalization fix (Addendum F). It applies to the T2m heads only; the maze base heads keep the old per-time scale.
- Maze reproduction at 13 and 59, since the paper value for 33 is not available.
- 2 backbone seeds (the spec's number) x 3 head seeds on T2m.
- Q3 only on mazes, after the headroom gate failed.

## Files
- Code:
  - `data.py`, `models.py`, `acore.py`, `train.py`, `heads.py`, `traj.py`, `halting.py`, `train_heads.py`, `eval.py`: models, heads, pools;
  - `make_mazes_upstream.py`: maze generation with the upstream generator;
  - `tests.py`: unit tests (a)-(e);
  - `analyze.py`: bootstrap, criteria, chart;
  - `capdiag.py`: post-hoc diagnostic;
  - `sens_f16.py`: float16 sensitivity at maze 99 (Addendum I);
  - `gpu.sh`, `*run_vm.sh`: VM and watchdog; `gpu/push.sh`, `gpu/pull.sh`, `gpu/vm.sh`: resume helpers (run from a /tmp scratch dir; the VM IP in vm.sh is the last one used).
- Results:
  - `results/p1_results.json`, `results/p1_chart.json`, `results/capdiag_t2m.json`, `results/sensitivity_f16_n99.json`;
  - raw pools under `results/vm/trader-cat-p1-7/` (`t2m/` and `maze/`): the final VM's /root/out, which carries every earlier stage forward (T2m pools byte-identical to the p1-3 sync, maze 59 to the p1-5 sync), plus the per-VM maze step logs (`logs/msize*_on_p1-*.log`).
