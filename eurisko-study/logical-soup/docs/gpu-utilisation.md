# GPU utilisation study (pool-7tcu9)

Guarding genome hashes improved whole-program GPU-lane throughput by 7.7% at
K16 and 6.6% at K18 (three paired trials pooled). CUDA launch/sync time barely
changed: the gain came from work outside CUDA. The variant remains isolated;
the long Phase 3 runs use the unchanged baseline.

## Measured throughput

Q4096, fuel256, seed1, eight host threads, ten epochs per invocation, same L4.
K18 pooled rate is total virtual trader instructions divided by total wall time.

| Population | Paired trials | Baseline M instr/s | Guarded M instr/s | Speedup | Baseline / guarded total CUDA seconds |
|---|---:|---:|---:|---:|---:|
| K=2^16 | 1 | 4.115 | 4.433 | 1.0773x | 3.359 / 3.352 |
| K=2^18 | 3 | 3.038 | 3.239 | 1.0659x | 33.426 / 33.405 |

K18 paired speedups were 1.0670x, 1.0865x and 1.0452x, with alternating run
order. Baseline total wall time was 268.269s; guarded was 251.671s for the same
815,117,832 instructions each. The 16.598s saving is almost entirely outside
CUDA; the CUDA-window difference is only 0.021s. Three pairs on one machine
are evidence of a modest gain, not a confidence interval or a prediction for
an evolved 5000-epoch population. Main timings include startup, initialization,
host phases and output, but exclude builds and diagnostic profiling.

This is an eurisko brief under GOAL.md's existing GPU authorization. The three
Phase 3 seeds use frozen baseline source; this experiment does not modify them.

The preceding scaling experiment measured two separate limits. Whole-process
performance is dominated by host metrics and reproduction. Within the GPU
kernel, register allocation and launch size account for most of the low
occupancy; only 3.39 threads per warp execute on average, but that counter
includes Bend's scheduler and allocator as well as the trader interpreter.

## Occupancy calculation

The kernel uses 255 registers per thread, allocated as 256, with 128 threads per
block. Hardware counters report a register ceiling of two blocks per SM.

| Device | SMs | Max warps/SM | Register-limited warps/SM | Grid blocks | Predicted occupancy ceiling with this grid | Measured occupancy |
|---|---:|---:|---:|---:|---:|---:|
| L4 | 58 | 48 | 8 | 128 | 16.67% | 15.86% |
| A100 | 108 | 64 | 8 | 128 | 7.41% | 7.35% |

L4 has enough blocks to fill its register-limited residency. A100 has only
128/(108×2) = 0.593 of a full wave, reducing its 12.5% register ceiling to
7.41%. This predicts the measured occupancy closely. Increasing the grid
alone is therefore a more plausible A100 experiment than an L4 optimization.
Reducing register allocation could raise residency on either GPU, but forcing
a lower register limit may spill into memory; it requires a measured trial.

89.58% uniform branch targets does not mean 89.58% useful SIMD work. The
3.39 active-lane average can include early-halted traders, data-dependent
control flow, work queues and allocator code. The counter does not isolate the
opcode switch. A dynamic wavefront keyed by current opcode or PC/opcode would
be a direct interpreter experiment; sorting by a static genome histogram
cannot keep program counters aligned. A function-pointer opcode table still
branches to different work per lane.

## Concrete experiment: guard genome hashing

The baseline computes `code.hash` eagerly in both population-ID passes,
including the ancestry pass and inactive cells. The variant branches on
activity and requested ID kind before calling the hash. It preserves traversal
order, IDs, active-cell threshold, sort inputs and all numeric reductions.
It is generated into `build/utilisation/src` by
`scripts/make-utilisation-variant.py`; production Bend sources remain unchanged.

The local 256-cell, 100-epoch, Q4096 seed-1 JS comparison is byte-identical.
It includes 78 births and 87 deaths; final active fraction is 964843 ppm,
so both active and inactive cases occur. The variant typechecks.
The GPU protocol uses the same L4, eight host threads, seed 1, fuel 256 and
Q4096. K16 gets one pair; K18 gets three pairs with alternating run order.
Reported rates include initialization and host work. Separate K18 one-epoch
split profiles measure metrics and reproduction without contaminating the
throughput timings. All completed CSVs must agree byte-for-byte.

## Host work that could move to the device

Settlement is a tree map over independent traders once prices and outcomes
are fixed. Reproduction reads the previous population and computes local
neighbour decisions; a balanced device map could preserve those decisions,
original indices and PRNG keys, followed by exact integer reductions.
Flattened arrays would reduce repeated tree walks and managed graph migration.
Both need prototypes and exact snapshot/CSV comparisons before adoption.

Genome hashes, active flags and ancestry IDs can be computed per cell in a
balanced fork. Sorting and unique counts need a device-friendly representation
and reduction. Wealth Gini uses ordered compensated F32 sums: retaining the
same sorted order and final serial accumulation is necessary for byte equality.
Returning compact keys to the host is an intermediate option; moving the
entire metric calculation is not required to remove redundant hashing.

Evidence for the existing hardware and phase measurements:
`runs/gpu-scaling-l4/{counter-summary,profile-summary,summary}.json` and
`runs/gpu-scaling-a100-retry/{counter-summary,profile-summary,summary}.json`.
These are K18 one-epoch diagnostic counters, not evolved-population profiles.

## Matched host-phase diagnostic

Separate K18 one-epoch runs, milliseconds (these are not ten-epoch averages):

| Phase | Baseline | Guarded |
|---|---:|---:|
| Preparation | 8 | 8 |
| Adjacent IO boundary | <1 | <1 |
| Trader execution | 1064 | 1072 |
| Sparse aggregation | 34 | 34 |
| Prices and settlement | 242 | 238 |
| Reproduction | 1935 | 1865 |
| Metrics and CSV construction | 2833 | 2534 |

Metrics fell by 299ms (10.6%). The whole diagnostic process took 7.283s versus
6.935s. Reproduction was also 70ms faster despite having no source change;
that difference should not be attributed to the hash guard. The three paired
throughput trials provide the stronger evidence for whole-program speedup.
Hardware occupancy counters were not re-collected for this host-only change;
its nearly identical CUDA windows do not demonstrate better warp utilization.

## Validation and rental

All ten invocations passed independent exact ledger and CSV-prefix checks.
The local JS guard comparison was also byte-identical. Isolated `make check`
passed, including native-launch safety/checkpoint tests; production Bend source
and the three frozen Phase 3 runs remain unchanged.

One spot L4 was used: `soup-gpu-utilisation-hash-0927125423-24184` in
us-central1-a. It was deleted after 849.478 seconds (14.158 minutes), under the
30-minute limit. Its instance and disk lists were empty, checked at
2026-09-27T13:09:38Z. Rate ceiling $0.87/hour gives $0.20529, unverified against
billing. Raw timings, build hashes, CSVs, validation and deletion/cost receipts
are in `runs/gpu-utilisation/`; the GPU log records the lifecycle. Periodic
backups and the wrapper's successful final sync preserved all results. A later
redundant backup reached the already-deleted VM and timed out; no data was lost.

The measured change is retained as an isolated reproducible variant. Adopting
it for future production runs is a separate change; it was not injected into
an experiment already in progress.

Cross-model closeout review: eurisko (Fable), Agent Mail #2130, 2026-09-27,
SHIP after checking pooled rates, CSV hashes, ledger, occupancy arithmetic and
cleanup receipts. Bead pool-7tcu9 is closed.
