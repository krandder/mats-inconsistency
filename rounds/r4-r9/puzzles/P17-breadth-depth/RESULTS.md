# P17 results: a breadth-by-depth compute market on EqR Sudoku-Extreme

Final, 2026-09-30 05:20Z. Bank built 2026-09-29 23:48Z to 09-30 05:04Z on spot L4 VMs (trader-bxd-p17-2 to -12); the test simulation
was run once, at 05:05:51Z (`results/TEST_RUN_STARTED`). Pre-registration: `PREREG.md` (sha256 `09ef7e59...c575`
after amendment 1; the frozen original `1ff8759d...7799` is line 1 of `PREREG.sha256`).

## Verdict

**Checker gain, not a market gain.** The kill rule passes, but the pre-registered secondary test fails, so no mazes.

- **The market beats EqR's own selection rules at equal NFE.** At the primary budget b = 64 on 2,048 test puzzles:
  M-chk 98.54% vs smallest-residual (SR) 92.72% and majority vote (MV) 92.38%. The paired differences are
  +5.81 [+4.83, +6.84] and +6.15 [+5.13, +7.23] points.
- **A plain restart-until-valid rule does the same.** SEQ-chk (round robin: run the next restart to depth 16, stop at
  the first valid grid) scores 98.49%. The market's edge is +0.05 [0.00, +0.15] points: 1 puzzle won, 0 lost. The
  lower bound is not above 0, so the secondary test fails.
- **What the market learned is "restart, don't deepen".** At b = 64 it made 8,519 spawns and 626 extends. The extends
  used 16,645 of 128,723 NFE (13%) and produced 1 valid grid. A spawn settles a puzzle 24% of the time, an extend
  1.7% (val base rates). Once a checker exists, the breadth-by-depth price is lopsided enough that a fixed rule gets
  it right.
- **The gain splits into checker plus pooling, not pricing.**
  - Without pooling (a fixed budget of b per puzzle), the market's choices are identical, puzzle by puzzle, to
    restart-until-valid at every budget (M-chk-local = SEQ-chk-local). Both reach 93.41% at b = 64, +0.68
    [+0.34, +1.03] over SR.
  - Pooling the budget across the 256 puzzles of a block adds the other +5.1 points. Easy puzzles stop at their first
    valid step, and the hard ones get the leftover NFE.

Brackets are 95% paired bootstrap CIs (10,000 resamples over puzzles).

## Accuracy vs budget (test, 2,048 puzzles, %)

| method | 16 | 32 | **64** | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| M-chk (market, checker) | 96.29 | 97.71 | **98.54** | 98.68 | 98.68 | 98.68 | 98.68 |
| SEQ-chk (restart until valid) | 96.34 | 97.71 | **98.49** | 98.49 | 98.49 | 98.49 | 98.49 |
| SR+chk (static, valid first) | 85.21 | 89.94 | 93.31 | 95.56 | 96.92 | 97.36 | 98.00 |
| SR (EqR smallest residual) | 85.69 | 89.94 | **92.72** | 94.87 | 96.00 | 97.17 | 97.75 |
| MV (EqR majority vote) | 85.69 | 89.94 | **92.38** | 93.65 | 95.12 | 96.29 | 97.12 |
| FD (fixed depth, 1 restart) | 85.69 | 89.94 | 92.38 | 93.65 | 95.21 | 96.14 | 96.63 |
| FR (fixed restarts at depth 16) | 85.69 | 89.45 | 92.72 | 94.87 | 96.00 | 97.17 | 97.75 |
| RS (random split) | 83.50 | 88.48 | 91.80 | 94.24 | 95.90 | 96.83 | 97.22 |
| M-res (market, no checker) | 84.57 | 96.00 | 96.78 | 96.83 | 96.83 | 96.83 | 96.83 |
| RT (residual threshold, no checker) | 85.69 | 95.56 | 96.83 | 98.39 | 98.39 | 98.39 | 98.39 |
| M-chk-local (no pooling) | 85.69 | 90.43 | 93.41 | 95.56 | 96.44 | 97.41 | 98.00 |
| M-chk, random prices | 41.50 | 54.74 | 69.87 | 89.89 | 98.68 | 98.68 | 98.68 |

The CIs at b = 64 are M-chk [98.00, 99.02], SEQ-chk [97.95, 98.97], SR [91.60, 93.85] and MV [91.21, 93.51]. The
single-trader ablation equals M-chk to within one puzzle at every budget, so the wealth-weighted trader pool adds
nothing either. Val-chosen static splits at b = 64: SR 16 x 4, MV 64 x 1, SR+chk 8 x 8, SEQ-chk depth 16.

M-chk minus SEQ-chk at each budget: -0.05 [-0.15, 0.00], 0.00, **+0.05 [0.00, +0.15]**, then +0.20 [+0.05, +0.39]
at b = 128 and above. The b >= 128 edge is 4 puzzles that need depth beyond 16. The market reaches them by extending
after all 128 restarts are spent; SEQ-chk at its val-chosen depth 16 cannot. That is outside the primary budget, and
"restart until valid, then deepen" would get the same 4.

The verifier-free comparison is reported only and decides nothing. M-res beats SR and MV at b = 32-256, and loses to
SR at b = 16 and 1,024. Against its own simple rule RT it scores -0.05 [-0.63, +0.54] at b = 64 and -1.56
[-2.15, -0.98] at b >= 128, so it adds nothing there either.

## Where restarts matter (the coordinator's note)

- **At noise 0.5, depth alone reaches 99.0% at 1,024 steps** (val gate), so restarts have little headroom at high
  budgets. At noise 0.01 (this bank) depth alone reaches 97.36% at 1,024 on the val gate and 96.63% (FD) on test.
- **The comparison covers the budgets where restarts matter (16-128).** There, breadth wins clearly: the val-chosen
  SR split at b = 64 is 16 x 4. Restart-until-valid at a mean of 55 NFE (98.49%) beats depth alone at 1,024 NFE
  (96.63%).
- **Above b = 128 the bank is the limit.** 98.68% of test puzzles have some restart that becomes valid within its
  staircase cap, and 98.49% have one valid within 16 steps.
  - The adaptive policies hit that ceiling and stop spending: M-chk uses a mean of 139 NFE at b = 256-1,024.
  - The b >= 256 rows therefore compare bank-capped adaptive policies with full-budget statics.
  - The primary b = 64 is below the cap: M-chk used 62.9 of 64.
- **The brief's "93.0% to 99.8% with 128 restarts" is in the paper** (Table 4: D64B1 93.0, D64B128 99.8). It was
  probably measured at noise 0.5 (see amendment 1). At noise 0.01, our 128-restart staircase bank (8,192 NFE per
  puzzle) tops out at 98.68%.

## Gates and amendment 1

| gate (val blocks 0-1, restarts 0-1) | band | noise 0.5 (frozen setting) | noise 0.01 (amendment 1) |
|---|---|---|---|
| G1 D16B1 | 81.4-91.4 | **76.07% (fail)** | 85.74% [83.0, 88.5] |
| G1 D64B1 | 88.0-98.0 | 89.45% | 93.07% [90.9, 95.0] |
| G2 valid-but-wrong (row, step) | 0 | 0 | 0 (also 0 in all 12 blocks of the full bank) |
| G3 projected bank cost | < $20 | $8.93 | $9.29 |
| B1 accuracy at depth 1,024 (info) | | 99.02% | 97.36% |

- **What happened.** The frozen noise 0.5 (from the repo's eval configs) failed G1, and the run stopped automatically
  (status Problem, 23:18Z).
- **The approval.** The coordinator approved a rerun only if the noise matched the authors' single-restart protocol.
- **The amendment.** Amendment 1 (PREREG.md, 23:47Z) was written before any market run and before any test-block
  data. It cites:
  - paper Section 5.2 ("small path noise (β= 0.01) performs best ... At test time, one can increase β under breadth
    scaling");
  - Table 12 (β=0.01, Eval 86.4, the D16B1 headline);
  - Table 13 (restarts at fixed β=0.01);
  - the repo defaults (`models/eqr.py:57` and `config/arch/eqr.yaml:27`: `noise_scale: 0.01`).
- **Evidence the other way.** The amendment also records it: every repo eval config and the README use 0.5.
- **Scope.** Only the noise changed. The kill rule, secondary test, budgets, sample and seeds did not.
- **The rerun matched the paper.** At 0.01 the released checkpoint reproduces Table 4 (86.4 and 93.0). The depth-64
  pre-check and the full gate ran on different VMs and gave identical numbers.
- **Correct grids are absorbing.** At both noises, no trajectory is correct at one step and wrong later, so
  stopping at the first valid step loses nothing.

Before the one-time test run, a val-only dry run (val block 0 standing in for test, scratch in /tmp) checked the code
on real data. No design choice changed after it.

## Equal-NFE accounting

One NFE is one outer EqR step, and every policy is replayed from the bank with exact per-puzzle NFE.
- Static policies spend exactly b per puzzle.
- Pooled policies spend at most 256 x b per block of 256 puzzles. `sim.py` asserts this for every method and budget,
  and `analyze.py` rechecks it (`equal_nfe_ok` true for all).
- Checker policies are charged only up to the first valid step.
- Nothing uses labels at run time: settlement is the constraint check, and valid equals correct in this bank.

## Cost

- **Total: $10.76 of the $30 cap**, at the $0.71/h ceiling: 15.15 VM-h across 12 spot VMs, including the noise-0.5
  gate ($0.63). Spot actual is lower.
- **Preemptions:** 7 spot preemptions. They resumed from synced, sha256-verified units, and no unit was lost.
- **Cleanup:** all VMs and disks were deleted. `./gpu.sh check` (instances and disks filtered on trader-bxd-p17)
  was empty at 05:05Z and again at 05:20Z.
- **Ledgers:** rows are in GPU-LEDGER.md and BUDGET-24H-20260929.md.
- **gpu.sh fix:** the leftover-disk alert now counts only unattached disks. It had flagged disks of VMs that were
  still running.

## Independent check

A read-only verifier subagent (09-30, about 05:30Z) passed all 9 checks with its own code. Nothing it found changes
the status or the headline.
- **PREREG and timing.** The hash matches the amended PREREG.md. The first test-block file is from 01:27Z, after the
  23:47Z amendment. The only earlier simulation was the val-only dry run. `sim.log` shows a single test run.
  `bank_meta` has noise 0.01.
- **Data.** 84 units; all 97 bank files match every checksum file that lists them.
- **SR, MV and SEQ-chk recomputed from the raw bank.** 0 per-puzzle mismatches at every budget: SR 92.72%, MV 92.38%,
  SEQ-chk 98.49%. SEQ-chk NFE also matches. The residual score follows the PREREG and EqR's eval code.
- **Bootstrap.** Mchk-SR +5.81 [+4.83, +6.84], Mchk-MV +6.15 [+5.13, +7.23], Mchk-SEQchk +0.05 [0.00, +0.15]. The
  secondary lower bound is exactly 0 because it rests on 1 puzzle won and 0 lost, so the status follows.
- **Equal NFE.** No method exceeds its budget. The b = 64 market trace sums to the recorded NFE in every block and has
  0 cost, spawn-order or depth-cap errors. Replaying it reproduces M-chk's answers exactly. The market never reads the
  labels, and there are 0 valid-but-wrong grids on test.
- **Ceilings.** 98.68% (any restart valid within its cap) and 98.49% (valid by depth 16) recomputed.
- **Cloud and charts.** No trader-bxd-p17 instances or disks remain (the unfiltered lists are non-empty, so the query
  works). All 91 chart points match `summary.json`.

## Files

- `PREREG.md`, `PREREG.sha256`: the design, amendment 1 and both hashes.
- `bank.py`, `gate.py`, `run_vm.sh`, `startup.sh`, `gpu.sh`: the bank, the gates, the VM driver and the watchdog.
- `sim.py`: summarizes the bank and replays every policy.
- `analyze.py`: the bootstrap, the status and the charts. `gatecurve.py BANKDIR NOISE`: the gate curves in the chart JSON.
- `results/vm_out/bank/`: the raw bank, 84 units, with per-VM `SHA256SUMS.*`.
- `results/summ/`: per-block summaries.
- `results/per_puzzle.npz`: per-puzzle correctness and NFE for every method and budget.
- `results/summary.json`, `results/choices.json`, `results/val_table.json`, `results/wealth_test.json`.
- `results/trace_M*_b{64,256}.npy`: market action traces.
- `results/gate_noise05/`: the failed noise-0.5 gate.
- `charts/p17_charts.json`: accuracy, differences and NFE vs budget; both gate curves; paper points; bank ceiling.
- `gpu/LEDGER.md`, `gpu/watchdog.out`: VM lifecycle.
