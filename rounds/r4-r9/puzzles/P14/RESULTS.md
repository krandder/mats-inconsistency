# P14 results: value-priced event publishing (LEG plus a LOCO listing market) against tuned sparse backprop

Status: **No gain.** Written 2026-09-29 ~23:55Z. The pre-registration is `PREREG.md` (frozen 22:03:46Z, sha256 `001137d1…4b13791`).

## Answer in three lines

- V (LEG training plus the LOCO listing market) ends with a sufficient single bit in **3/20** seeds. The strongest tuned sparse-backprop baseline, BP* = BP-L0-cm (straight-through plus L0 hard-concrete slot gates), does so in **17/20** at the same counted FLOPs. S1 and S2 fail, and F1, F2, F3 and F5 fire.
- The only significant attribution is the **estimator: LEG is worse than straight-through** (van Elteren z = +3.94, Holm p = 0.0002). The listing and penalty-type effects are not significant.
- The LOCO regulator values **planted** slots correctly (S4: P-LOCO passes 10/10). On **trained** slots it churns about 16 slots per run, with harmful delistings in 9/20 runs. Money does not replace it (S5 fails), so this report says "LEG plus LOCO pruning", not "economy".

P14 cannot bear on the THEORY.md programme falsifier (beating Bansal-style recall plus progressive loss at large extrapolation). P14 has no depth or extrapolation axis; its 21x21 test only checks translation equivariance.

## Setup (as run)

- **Task.** T-reach on 9x9 mazes: output map Y[c] = [dist(start, c) <= 4] on open cells.
  - k = 4 publisher slots each emit one bit per cell, and a reader R maps bits plus walls to Y.
  - The sufficient single-bit codes are known and checked in gate g6: g2 = [dist <= 2], d02, and near-sufficient ring and d12, plus complements.
- **Arms.**
  - Primary arms (20 seeds each): V, V-noD, BP, BP-D, V-L1, BP-L1-cm, and BP* extended to 20.
  - Secondary, floor and planted arms (10 seeds each).
  - T-dist: 4 arms x 10 seeds.
  - Seeds: tuning 100-102 with validation set 777; eval seeds 0-19; probe-test set 4102.
- **Matched budget.** B = 1.9273e12 counted FLOPs (V's 6000 steps). Every other arm trains until its counter reaches B; the multipliers are in PREREG.md (for example, BP-L0-cm runs 14707 steps).
- **Frozen values.**
  - G = 500, V fee c* = 1, BP-L0-cm lambda = 3; the other values are in the PREREG.md table.
  - BP* = BP-L0-cm.
  - The endpoint switch fired, so S1 is a one-sided log-rank test on FLOPs to first SUFF1 (12 checkpoints, censored at B).
- **Metrics.**
  - SUFF1: best live bit's single-bit probe gap_1 <= 0.02 nats/cell and probe accuracy >= 0.995.
  - A run with no live bit scores the walls-only floor, 0.5614.
- **Compute.**
  - CPU only (c2d-highcpu-32 spot, max-run 6 h, DELETE on termination): 3 VMs, 6.40 VM-hours.
  - **$7.69** at the $1.20/h on-demand ceiling (spot actual is lower) of the $15 cap.
  - All VMs and disks are deleted; `gcloud compute instances list` and `disks list` filtered on `trader-b3-p14` return 0 items (23:48Z).

## Main table (final checkpoint = budget B)

gap_1: median [95% bootstrap CI], nats/cell. SUFF1: count [Wilson 95% CI]. 8-sample: SUFF1 recomputed from 8 sampled bits. ll9: greedy-bit 9x9 output log loss. H: mean binary entropy of sigmoid(l).

| arm | n | gap_1 | SUFF1 | 8-sample | SUFFALL | live slots | ll9 | H |
|---|---|---|---|---|---|---|---|---|
| **V** (LEG + listing) | 20 | 0.0418 [0.0224, 0.0514] | 3/20 [0.05, 0.36] | 3/20 | 9/20 | 2.90 | 0.0404 | 0.167 |
| V-noD (LEG) | 20 | 0.0223 [0.0166, 0.0288] | 4/20 | 5/20 | 18/20 | 2.95 | 0.0031 | 0.002 |
| BP-D (ST + listing) | 20 | 0.0170 [0.0088, 0.0235] | 8/20 | 1/20 | 19/20 | 3.20 | 0.0072 | 0.231 |
| BP (ST) | 20 | 0.0161 [0.0097, 0.0212] | 10/20 | 5/20 | 20/20 | 4.00 | 0.0014 | 0.087 |
| V-L1 | 20 | 0.0227 [0.0140, 0.0313] | 9/20 | 9/20 | 18/20 | 3.00 | 0.0035 | 0.001 |
| BP-L1-cm | 20 | 0.0254 [0.0182, 0.0283] | 3/20 | 3/20 | 20/20 | 3.55 | 0.0004 | 0.028 |
| **BP-L0-cm = BP\*** | 20 | **0.0005 [0.0002, 0.0078]** | **17/20 [0.64, 0.95]** | 16/20 | 20/20 | 1.60 | 0.0010 | 0.009 |
| BP-L1-D-cm | 10 | 0.0207 [0.0123, 0.0319] | 4/10 | 1/10 | 10/10 | 2.00 | 0.0018 | 0.019 |
| DisARM-L1-cm | 10 | 0.0294 [0.0249, 0.0385] | 0/10 | 0/10 | 6/10 | 3.50 | 0.0029 | 0.001 |
| STG-L1-cm | 10 | 0.0070 [0.0026, 0.0137] | 7/10 | 7/10 | 10/10 | 3.00 | 0.0015 | 0.004 |
| V-W (wealth survival) | 10 | 0.0142 [0.0063, 0.0318] | 4/10 | 4/10 | 9/10 | 2.20 | 0.0048 | 0.021 |
| V-local (sanity) | 10 | 0.0676 [0.0636, 0.0763] | 0/10 | 0/10 | 0/10 | 2.60 | 0.0232 | 0.046 |
| N-sel (floor) | 10 | 0.549 | 0/10 | 0/10 | 0/10 | 3.70 | 0.561 | 0.691 |
| N0 (floor) | 10 | 0.559 | 0/10 | 0/10 | 0/10 | 3.60 | 0.557 | 0.690 |
| P-LOCO (planted) | 10 | 0.0000 | 10/10 | 10/10 | 10/10 | 1.00 | 0.0001 | 0.000 |
| P-W (planted) | 10 | 0.0000 | 10/10 | 10/10 | 10/10 | 3.20 | 0.0001 | 0.000 |
| ORACLE (g2 reader) | 20 | - | - | - | - | - | 0.0001 | - |

- **Equal steps.** At the 6000-step checkpoint the ordering is the same: V 3/20, BP-L0-cm 15/20 (gap_1 0.0051).
- **Kaplan-Meier.** Fraction reaching SUFF1 by B: V 0.25, BP-L0-cm 0.90. V reached SUFF1 at some checkpoint in 5 runs but held it at the end in only 3.
- **Collapse.** No arm collapsed (0 runs).
- **Cross-seed AMI of the live-bit partitions.** V 0.47, V-noD 0.72, BP-L0-cm 0.69, floors 0.02.
- **Random-init readout.** Median NMI(bit; g2) over 1000 random trader inits is 0.0006 (max 0.023), so the N-sel floor is as expected.

## Pre-registered tests

**Primary family (Holm over S1, S2), one-sided "V lower".**

- **S1 fails.** V vs BP*, log-rank on FLOPs to SUFF1: z = -4.95, i.e. BP* gets there sooner; p ≈ 1.0, Holm 1.0.
  - Paired Wilcoxon sensitivity: p ≈ 1.0.
  - Median gap_1 difference is +0.032 (V worse).
- **S2 fails.** V vs BP-D, Mann-Whitney on gap_1: p = 0.9994, Holm 1.0; median difference +0.024.

**Attribution family (van Elteren, Holm over 3), two-sided.**

| effect | contrast | z | p | Holm p |
|---|---|---|---|---|
| estimator (factorial A) | LEG minus ST | +3.94 (LEG worse) | 8.1e-5 | **0.0002** |
| listing (factorial A) | on minus off | +1.72 (listing worse) | 0.085 | 0.17 |
| penalty type (factorial B) | slot fee minus L1 | -0.10 | 0.92 | 0.92 |

Interactions are descriptive only (difference of median differences in gap_1):

- **Factorial A: +0.019.** Listing costs LEG 0.0195 but ST only 0.0009.
- **Factorial B: +0.027.**

**S3 to S5.**

- **S3 fails** (non-inferiority, paired t 95% upper bound <= 0.02).
  - 9x9: V minus BP* log loss = 0.039 [0.027, 0.052], upper bound 0.052 > 0.02. Fails.
  - 21x21: V minus ORACLE = 0.0095 [0.0065, 0.0124]. Passes.
- **S4 passes.**
  - P-LOCO meets planted criteria (a)-(c) in 10/10 seeds. It delists 5 of 6 slots and ends with one g2 copy.
  - Spearman of final VoI against Shapley joint-probe contribution: median 0.70 [0.48, 0.90].
  - V's median count of consequence-defined false delistings is 0.
- **S5 fails.**
  - V-W's SUFF1 on seeds 0-9 is 4 vs V's 2, which is within 2.
  - P-W passes (a)-(c) in **0/10**:
    - it fails (b) in every seed, keeping g2, the duplicate g2 and the ring listed (3-4 slots; the set is sufficient but not minimal);
    - (c) holds in 8/10 under the listed reading and 10/10 under the entropy-filtered reading.
  - Money kills the worthless slots first (random, then wall copy, then g1) but never the redundant ones. Wealth ranks slots about as well as VoI does (Spearman 0.74 [0.68, 0.77]).
  - Wealth ledger conservation error <= 5e-16. V-W: 36 bankruptcies per run.

**Falsifiers.**

- **F1 fires.** S1 fails, and V's SUFF1 (3/20) <= BP*'s (17/20). Newcombe difference -0.70 [-0.84, -0.40].
- **F2 fires.** V <= BP-D on both median gap_1 (0.042 vs 0.017) and SUFF1 (3/20 vs 8/20). Pay as a training signal adds nothing over straight-through under the same listing market here.
- **F3 fires** on the 9x9 half of S3.
- **F5 fires.** Consequence-defined false delistings occurred in 9/20 V seeds (the bar is 3), although the median is 0; P-LOCO passes.
- **F4 does not fire** (0/20 collapse).
- **F6 does not fire.** V has no advantage to reverse: median gap advantage -0.041 greedy and -0.042 sampled; SUFF1 advantage -0.70 and -0.65.
- **Sanity check passes.** V-local >= V is false (V-local 0.068, 0/10), so no bug hunt was triggered.

**Reading, by the pre-registered rules.**

- "LEG plus listing finds a sufficient single-bit event more reliably than the best tuned sparse backprop" is **falsified**.
- The listing main effect is not significant, so the pre-registered fallback "useful pruning tool, not a better learner" is not earned on the learning task either. What is earned is S4: the refit regulator values planted slots well.
- The significant estimator effect runs **against** LEG.
- S5 fails, so money is not a substitute for the refit regulator.
- Secondary SUFF1 differences, V minus arm (Newcombe 95%):

  | arm | difference |
  |---|---|
  | BP | -0.35 [-0.57, -0.06] |
  | V-L1 | -0.30 [-0.53, -0.02] |
  | STG-L1-cm | -0.55 [-0.77, -0.18] |
  | BP-L1-cm | 0.00 [-0.23, 0.23] |
  | DisARM-L1-cm | +0.15 [-0.14, 0.36] |

  These are unadjusted. DisARM-L1-cm is the only tuned baseline V is not behind.

## Mechanism notes (descriptive)

- **Churn with resets.**
  - V delists 16.2 slots per 6000-step run [14.6, 17.8], about one every 370 steps. Each respawn resets the slot's reader row and its Adam state.
  - V's output log loss is 0.040, against 0.003 for V-noD, and its bits stay young and unsaturated (entropy 0.167 vs V-noD's 0.002).
  - BP-D delists just as often (17.8) at almost no cost (gap_1 0.017 vs BP's 0.016), so the churn hurts the LEG-trained slots much more than ST-trained ones (interaction A above).
- **What the bits become.**
  - The best bit is closest to g2 in most runs: V 18/20, BP-L0-cm 20/20.
  - NMI(best bit, g2) >= 0.7: V 5/20, V-noD 12/20, BP-L0-cm 15/20 (median about 0.96; >= 0.9 in 11/20).
  - V's delisted slots carry little family information: for each delisted slot, take its best match over the g6 family by I/min(H); the median is 0.023.
- **Why BP-L0-cm wins.** Its slot gates prune to 1.6 live slots on average and push one bit onto g2. Its joint gap is 0.0003, and its bits are near-saturated (entropy 0.009), so greedy and sampled readouts agree (17/20 vs 16/20).
- **Wall clock per run** (one thread): V 20.6 min, V-noD 48 min (2.09x steps), BP-L0-cm 5.6 min.

## T-dist (secondary task, descriptive only)

Setup: k = 4, retuned on seeds 100-101 with the joint probe. Tuned values: V fee 3, BP-D fee 0.0333, BP-L1-cm lambda 0.01, BP-L0-cm lambda 0.1. The target is the distance class; the oracle joint-probe accuracy is 0.9998.

| arm | code type (NMI 0.7 rule) | at 0.5 | at 0.9 | joint-probe acc [95% t CI] | gap_all | live |
|---|---|---|---|---|---|---|
| V | insufficient 6, other-sufficient 3, dense 1 | dense 8, insufficient 2 | insufficient 6, other-suff. 4 | 0.9899 [0.9853, 0.9946] | 0.0253 | 2.8 |
| BP-D | other-sufficient 7, dense 3 | dense 8, other-suff. 2 | other-suff. 10 | 0.9977 [0.9971, 0.9984] | 0.0060 | 3.5 |
| BP-L1-cm | other-sufficient 10 | dense 9, thermometer 1 | other-suff. 10 | 0.9968 [0.9963, 0.9973] | 0.0080 | 3.5 |
| BP-L0-cm | other-sufficient 10 | dense 8, other-suff. 2 | other-suff. 10 | 0.9971 [0.9964, 0.9977] | 0.0080 | 4.0 |

- No arm found a thermometer code (at most one live bit matched a thermometer event; that event is g2).
- The backprop arms find sufficient joint codes of 7-16 states that are neither thermometer nor 4-state dense.
- V is the only arm with insufficient joint codes (6/10).

## Tuning notes

- **Tolerance never binds** for STG, V, BP-D and BP-L1-D-cm: the output-log-loss constraint excluded no value, so their choice is the plain score minimum.
- **Still at a grid edge after 2 bracketing extensions:**
  - STG-L1-cm (lambda 1.11e-4, low edge), which may be under-regularised;
  - T-dist BP-D (fee 0.0333, low edge).
  - T-dist BP-L0-cm needed 2 extensions but ended interior.
- **Gate g2 has two verdicts** (PREREG deviation 3).
  - As written it fails: a reader fed g1 alone reaches 9x9 accuracy 0.983, above the spec's 0.97 bar.
  - Rescored on the sufficiency criterion it passes: g1's log loss is 0.079 above the g2-fed reader's.
- **Preemption.** Tuning VM trader-b3-p14-1 was spot-preempted at 20:52Z and resumed on trader-b3-p14-1-r1 from the last verified sync (PREREG deviation 9).

## Departures after the freeze

1. **T-dist code-type rule.** PREREG says only "report thermometer vs dense".
   - The classification rule in `tdist_stats.py` was written at 22:08Z and refined at 22:09Z. That is after the freeze (22:03:46Z) but before any T-dist eval result existed.
   - The refinement added the "other-sufficient" class after a dry run on the seed-100/101 tuning runs.
   - The rule's sha256 is logged in `vm/LEDGER.md` (RULE 22:09Z; RULE-FIX 22:20Z corrected docstring times only).
   - T-dist is descriptive, and thresholds 0.5 and 0.9 are reported as a sensitivity check.
2. **BP-L0-cm T-dist eval.** The 10 eval lines were added to `vm/queue_tdist_eval.txt` by hand, because `make_queue.py` knew only the spec's default BP* (BP-L1-cm). This follows PREREG deviation 7 and uses the frozen T-dist value (lambda 0.1).
3. **Watchdog and sync bugs during eval.** Two bugs in `vm.sh` were found and fixed mid-run (ledger FIX lines 23:12Z and 23:19Z).
   - The watchdog loop let ssh consume its VM list on stdin, so only the first VM was synced and idle-checked. trader-b3-p14-2 relied on manual syncs and its 6 h max-run from about 21:00 to 23:19Z. The spend cap would still have fired, one VM per 5-minute cycle; spend never came near it.
   - The remote tar exited 1 when a checkpoint changed while being read, so syncs during busy periods failed silently (r1 had no good sync between 22:57 and 23:19; -2 none after 22:03).
   - No data was lost: the final syncs verified all 290 eval results by sha256 before deletion.
   - The same bug class was flagged to the other puzzle agents (fleet-now key `vm-watchdog-stdin-bug`).
4. **Eval scheduling.** The 250 reach and 40 T-dist eval runs were merged and cost-balanced into two 145-job queues, one per VM (`vm/queue_evalall_{0,1}.txt`), so T-dist ran alongside reach eval rather than after it. This affects wall time only.
5. **Cleanup.** After the results were summarised, the 224 `.partial` resume checkpoints (165 MB) were moved to trash, including those of two V tuning extension runs stopped before the freeze (PREREG deviation 8). Every finished run's JSON is kept.

## Files

- `PREREG.md`, `PREREG.sha256`
- Code:
  - `p14.py` (all arms, LEG, the listing and wealth markets, probes);
  - `stats14.py` (reach analysis), `tdist_stats.py` (T-dist analysis);
  - `gates.py`, `select_hp.py`, `make_queue.py`, `bracket_queue.py`, `planted_shapley.py`;
  - `vm.sh` (VM launcher, watchdog, sync).
- Results:
  - `results/summary.json`, `results/summary_tdist.json`;
  - `results/hp.json`, `results/hp_select_{reach,dist}.json`, `results/gates*.json`;
  - `results/runs/*.json` (per-run finals).
- Charts: `charts/gap1_by_arm.json`, `charts/suff1_by_arm.json`, `charts/km_suff1.json`, `charts/tdist_by_arm.json`.
- Ledgers: `vm/LEDGER.md`, `../../GPU-LEDGER.md` (P14 section).
