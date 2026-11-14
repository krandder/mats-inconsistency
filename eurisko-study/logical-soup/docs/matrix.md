# Run matrix (Phase 3): scripts/matrix.sh and scripts/summarize.py

## Running a matrix

The matrix is a small JSON file; edit cells there. `matrices/phase3.json` is the
Phase 3 default: K in {2^12, 2^14, 2^16} x F in {2^8, 2^10, 2^12} x seeds
{1, 2, 3}, 5000 epochs, b=64, free income 0, chunk 100 (27 runs), plus three
variants on K=2^14 F=2^12, 3 seeds each (9 runs):

| variant | run.sh flags | meaning |
|---|---|---|
| `heldpi` | `--held-out-family pi --release 2000` | pi digits held (frozen, unscored, unseen) until epoch 2000: time-to-price |
| `r1` | `--visible-radius 1` | each cell sees prices of questions within distance 1 only |
| `boot` | bootstrap `{free-income 4096, epochs 50}` | 4096 Q16 ticks (1/16 share, ~29x a cell's mean fuel cost) per cell per epoch for epochs 0-49, then 0 |

run.sh takes a constant `--free-income`, so matrix.sh does the bootstrap as two
legs: `<run>/bootstrap/` runs the first 50 epochs with income 4096, its
snapshot, metrics and timing are copied into `<run>/`, and run.sh resumes there
with income 0 (the snapshot carries no income, so the handoff is exact).
`<run>/bootstrap.json` records it; `config.json` of `<run>/` says income 0.

    scripts/matrix.sh matrices/phase3.json            # the default matrix, -j 8
    scripts/matrix.sh --dry-run                       # print the 36 run.sh commands
    scripts/matrix.sh --only '^k1[24]-' -j 8          # a subset (regex on the run name)
    scripts/matrix.sh --epochs 100 --name phase3-bench --only 's1'   # short copy elsewhere
    infra/gpu-run.sh --up -- scripts/matrix.sh --only '^k16-' -j 1 \
        --engine "/root/soup/build/native/soup --threads 8"          # on the VM

Runs land in `runs/<name>/k<logK>-f<logF>-s<seed>[-<variant>]/` (run.sh's files
plus `run.log`); `runs/<name>/matrix.log` has one line per finished run (rc,
wall seconds, rows) and `runs/<name>/matrix.json` the spec used. Rerun the same
command to resume: runs whose metrics.csv has epochs+1 lines are skipped,
unfinished ones resume from their last chunk. Runs start largest K first so the
long runs do not trail at the end.

Engine: with `--engine CMD`, every run gets `SOUP_CMD=CMD`. Without it the JS
engine is built once and frozen to `runs/<name>/soup.js` (with
`soup.js.commit`: HEAD and any uncommitted src/ changes), so a rebuild of
build/soup.js during a long matrix does not mix engines; resuming reuses the
frozen copy. K >= 2^15 overflows the default JS stack (`bend: memory fault
(machine stack overflow?)`, also at HEAD 3f13626); matrix.sh raises the stack
rlimit to 4 GB and sets `BUN_JSC_maxPerThreadStackUsage=4000000000` (values of
2^32 and up are rejected by JSC), which runs K=2^16 at ~1.3 GB RSS. Plain
`scripts/run.sh` at K >= 2^15 needs the same two settings.

## Summarizing

    PYTHONPATH=.pylib python3 scripts/summarize.py runs/<name>/ --plot
    python3 scripts/summarize.py runs/a-k12-f12-s1 runs/a-k12-f12-s2 --out DIR

Writes `summary.md` and `summary.csv` (and with `--plot`, `brier_vs_K.png` and
`brier_vs_F.png`, one panel per value of the other axis, one line per family,
error bars = sd over seeds, dashed line = 0.25) into the matrix directory or
`--out`. A cell is the run name without its seed (`k14-f12-heldpi`); names
without the k/f pattern group by config.json. Per run, then mean ± sample sd
over seeds:

| column | definition (defaults as flags) |
|---|---|
| Brier per family | mean of `brier_<fam>` over the last `--window 50` epochs, 1.0 units |
| takeoff | first epoch with births >= `--takeoff 50`; (k/n) = seeds that took off, mean over those |
| ttp | held-out family only: epochs from release to the first epoch that starts `--hold 50` consecutive epochs with its Brier < `--price 0.22` (a nan epoch breaks the streak) |
| Δsh_<trader> | least-squares slope of the shadow trader's cumulative profit over the last `--slope 1000` epochs, shares per epoch (Q16 ticks / 65536). Inductor claim: goes to 0 |
| coh_and, coh_or | last-window mean of `coh_and_ppm`, `coh_or_ppm`, 1.0 units |
| genotypes, gini | last epoch |
| epochs, wall_s | metrics rows; sum of timing.csv wall seconds |

`make check` runs `tests/summarize_test.py` (synthetic runs with known answers,
plus matrix.sh --dry-run expanding phase3.json to the 36 expected commands).

## Verification (2026-09-27)

**Calibration runs as a one-seed matrix** (`runs/2026-09-27-calib-summary/`,
1000 epochs, JS lane; k12 has seeds 1 and 2):

| cell | n | periodic | lfsr | ca30 | ca110 | prime | pi | and | or | takeoff | Δsh_mom | Δsh_pat4 | Δsh_arb | coh_and | coh_or | genotypes | gini |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k12-f12 | 2 | 0.225±0.022 | 0.251±0.001 | 0.273±0.033 | 0.252±0.001 | 0.213±0.038 | 0.252±0.001 | 0.221±0.026 | 0.251±0.002 | 378 (1/2) | +493±64 | +510±61 | +72±81 | 0.054±0.051 | 0.007±0.005 | 4015±110 | 0.618±0.006 |
| k14-f12 | 1 | 0.187 | 0.259 | 0.323 | 0.256 | 0.144 | 0.258 | 0.162 | 0.254 | 180 | +359 | +384 | +204 | 0.096 | 0.035 | 16198 | 0.518 |

(Full table with all six shadow slopes in `runs/2026-09-27-calib-summary/summary.md`.)
It matches CALIBRATION.md's 1000-epoch reading (windowed means, so the
digits differ slightly from its last-row values): patterns and pi untouched,
shadow pattern traders still earning ~500 shares/epoch, arbitrage unexploited.

**Tiny matrix that executes** (`matrices/tiny.json`: K=2^8, F in {2^4, 2^6},
seeds 1-2, 20 epochs, all three variant kinds, 10 runs): all rc=0 in 7 s at
-j 8, 2-5 s per run; a second invocation skipped all 10; the boot runs carry
more wealth at epoch 5 than their base runs (income paid), then continue at
income 0; the heldpi runs show pi `nan` until the release epoch.

## Wall time and projection (JS lane, this box)

Measured: the 12 s1 cells of phase3.json for 100 epochs
(`scripts/matrix.sh --epochs 100 --name phase3-bench --only ...`), 9-12 runs
at once on a box whose load average was 30-58 on 16 cores from other jobs.

| run | wall s (100 ep) | s/epoch | instr/cell/epoch | M instr/s |
|---|---|---|---|---|
| k12-f8 / f10 / f12 | 40 / 43 / 44 | 0.40-0.44 | 105 | 1.0-1.1 |
| k14-f8 / f10 / f12 | 137 / 135 / 137 | 1.35-1.37 | 102 | 1.2 |
| k14-f12 heldpi / r1 / boot | 139 / 146 / 142 | 1.39-1.46 | 102-108 | 1.2 |
| k16-f8 / f10 / f12 | 410 / 412 / 432 | 4.1-4.3 | 101 | 1.5-1.6 |

F barely matters (< 10%); cost is instructions, ~linear in K. Instructions per
cell grow as the soup evolves: 105 -> 136-139 by epoch 900 at k12, 102 -> 110 at
k14 (calibration timing.csv); fuel caps them at 256. Projection for 5000
epochs, per run:

| K | flat (100-epoch rate) | central (140 instr/cell) | bound (256 instr/cell) |
|---|---|---|---|
| 2^12 | 0.6 h | 0.8 h | 1.5 h |
| 2^14 | 1.9 h | 2.6 h | 4.7 h |
| 2^16 | 5.8 h | 8.1 h | 14.8 h |

Whole default matrix (9 k12 + 18 k14 incl. variants + 9 k16 runs): about 127
CPU-hours central (231 at the bound). At -j 8 with largest K first the
makespan is ~16 h central (8 k16 runs, then the 9th alone), ~30 h at the
bound; -j 9 brings the central case to ~14 h (all nine k16 in one wave).
K=2^16 is ~57% of the compute and at the bound exceeds the 12-hour-per-run
stop rule (GOAL.md).

For the VM decision (runs/gpu-cube, soup-astra, 10-epoch trials): at K=2^16
the L4 VM's GPU lane did 3.81 M instr/s and its 8-thread native CPU lane 2.91 M,
against 1.5-1.6 M per JS run here. One VM running one k16 run at a time is
therefore worth ~2.4 JS runs on this box: k16 there is ~2.4 s/epoch central
(~3.4 h per 5000-epoch run, 6.1 h at the bound, over the 6 h VM limit unless
the run resumes on a second VM). Options for the lead:

- all on this box: -j 9, ~14 h central, ~30 h at the bound, $0;
- K <= 2^14 here (27 runs, ~55 CPU-h: ~7-8 h at -j 8) and the 9 k16 runs on 9
  parallel L4 VMs (quota 16) with the GPU lane: ~3.4 h wall, ~30 VM-h,
  ~$26 at the $0.87/h ceiling in runs/gpu-cube/cost-audit.json;
- one VM serially: ~31 h for the k16 runs, no faster than this box.
