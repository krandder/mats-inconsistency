# Soup CUDA lane

The epoch sends one balanced population tree to `cube.run!`. Each leaf executes
its U32 trader VM and budgets only its emitted orders. Each join returns the
trader-result tree and merges the sparse buy/sell tries. Both trader execution
and market aggregation therefore run inside the same cube. Pricing (including
LMSR exp/log), settlement, reproduction, questions and metrics stay on the host.

`IO.now()` after question-vector preparation is a scheduling boundary, not a
source of randomness. Bend 2.0.29 only dispatches a bang from a sequential event
loop: a bang encountered inside an existing host fork can stay on the CPU.
The pure `epoch.cube` wrapper is called after that boundary. This follows
`guide/SHADERS.md` and `bend2/docs/BendRT/main.typ` in the pinned compiler.

Bend's GPU supports recursive Data trees and Nat. Population indices and trader
operations are U32; sparse aggregate totals stay Nat to preserve exact sums
beyond U32. The GPU runtime limits Nat to 48 bits. At the tested K=65536 and
current wealth/collateral limits, aggregate share totals are within that bound.

## Build and reproduce

On the CUDA VM with `/opt/bend2` at v2.0.29, CUDA 12.9 and clang19:

```sh
BEND_HOME=/opt/bend2 scripts/build-gpu.sh
build/gpu/soup --gpu on --threads 8 14 10 1 256 12 > gpu.csv
build/gpu/soup --gpu off --threads 8 14 10 1 256 12 > cpu.csv
cmp gpu.csv cpu.csv
```

`build-gpu.sh` fails on missing CUDA headers or build failures. `--gpu on` fails
if no device is available. The recipe inserts a host-side stderr trace directly
before each CUDA kernel launch: `SOUP_CUDA_PASS frontier=...`. A successful
process without these traces is not evidence of GPU execution. Traces are
included in measured wall time. `build-native.sh` remains a deliberate CPU-only
recipe even though the source now contains a bang.

`scripts/gpu-cube-bench.sh` runs the 256x10 smoke and K14/K16,Q12 comparisons,
with seed1, fuel256, 10 epochs and an individual 240-second timeout. All timings
include initialization, host phases, CSV output and runtime startup. It uses
`build/pool-on8tb-base/soup.js` when supplied (the pre-change JS reference),
otherwise compiles the current source to JS; `js-source.txt` records which.
The runner verifies the exact ledger and byte identity against JS and records
actual CUDA pass counts. `infra/gpu-run.sh` owns output sync and VM deletion.
Use a new output directory when repeating a recorded experiment.

## K16 JS stack limit

The original JS reference on the L4 VM failed before epoch 1 with
`bend: memory fault (machine stack overflow?)`. An unmasked local stack trace
identifies recursive `List.length` in the metrics path, not trader execution or
CUDA. Raising only `ulimit -s` is insufficient; JavaScriptCore has its own limit.
The original JS succeeds with both limits raised and produces the exact K16 GPU
CSV:

```sh
ulimit -s 131072
JSC_maxPerThreadStackUsage=67108864 bun reference.js 16 10 1 256 12
```

The benchmark recipe now sets both limits for JS. The first L4 session had
already entered automatic deletion when this was diagnosed; no second VM was
rented. Its K16 JS throughput is therefore unavailable. The separately recorded
General-host JS result verifies identity and supplies a clearly labelled
cross-host timing, not a same-machine GPU speedup. A source-level alternative is
a tail-recursive length implementation (or balanced metrics reduction); that
refactor is outside this CUDA leaf.
