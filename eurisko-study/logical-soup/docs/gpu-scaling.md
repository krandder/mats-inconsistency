# GPU scaling experiment (pool-dhl36)

L4 throughput falls as the population grows: GPU/native speedup is 1.274x at
K=2^18 and 1.221x at K=2^20. Only 12.2% and 9.5% of elapsed time is inside
CUDA launch/synchronization windows. Host metrics and reproduction are the
first optimization targets. The sampled kernel has poor lane utilization,
but these measurements do not isolate opcode divergence from runtime scheduling.

This is an eurisko agent brief, not a Kelvin quotation. Spending is covered by
GOAL.md. Production `src/*.bend` is unchanged from the working CUDA cube.
Calibration directories were excluded from uploads and left untouched.

## Throughput

Q=4096 questions, fuel=256, seed=1, ten epochs; rates are virtual trader
instructions/s over the whole process. CPU uses eight threads on the same VM.

| Device | K | Instructions | GPU seconds | GPU M instr/s | Native CPU seconds | CPU M instr/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | 2^18 | 271,705,944 | 91.773 | 2.961 | 116.901 | 2.324 | 1.274x |
| L4 | 2^20 | 1,076,450,270 | 437.426 | 2.461 | 534.196 | 2.015 | 1.221x |
| A100 | 2^20 | 1,076,450,270 | 436.250 | 2.468 | 534.455 | 2.014 | 1.225x |

The L4 CUDA windows total 11.161s and 41.719s respectively. Eliminating these
windows entirely, with all other work unchanged, would improve those GPU runs
by at most 1.138x and 1.105x. That bound rules out interpreter divergence as the
main end-to-end bottleneck in these runs.

A100 K20 GPU completed in 436.250s, or 2.468M instr/s, only 0.27% above
L4. Its CUDA windows total 46.509s (10.66% of wall), slightly longer than L4's
41.719s. The extra GPU hardware did not improve end-to-end throughput in
this single trial; the two VM types also have different host CPUs.
The A100 K20 GPU CSV is byte-identical to L4's ten epochs.

## Phase and transfer profile

Separate K18, one-epoch split profiles, milliseconds:

| Phase | L4 GPU | L4 CPU | First A100 GPU | First A100 CPU |
|---|---:|---:|---:|---:|
| Preparation | 8 | 8 | 8 | 8 |
| Adjacent IO boundary | <1 | <1 | <1 | <1 |
| Trader execution | 1047 | 4665 | 1204 | 4741 |
| Sparse aggregation | 36 | 23 | 40 | 25 |
| Host finishing | 5370 | 5164 | 6578 | 6624 |
| ↳ prices and settlement | unmeasured | unmeasured | 255 | 120 |
| ↳ replication | unmeasured | unmeasured | 1964 | 2145 |
| ↳ metrics and CSV construction | unmeasured | unmeasured | 4359 | 4359 |

The last three rows decompose host finishing; do not add them to it again.
Trader execution takes one runtime pass; aggregation takes 29 short passes.
A pass can launch four actual kernels. The adjacent IO probe measures only
boundary overhead, not graph migration. Metrics, including genome hashing,
sorting and distinct counts, are the largest measured host phase on A100.

A separate fused K18 L4 Nsight capture recorded 416.7 MB H→D in 9,397 transfers
(copy-engine sum 54.1ms), and 509.0 MB D→H in 10,569 transfers (52.6ms).
It also recorded 11,403 CPU unified-memory fault events and 2,381 GPU faults
in 293 events (GPU fault duration sum 94.8ms). CPU fault durations were not
available. Transfer, fault and kernel times overlap; these sums cannot be
added to estimate elapsed time or used to rule out host fault overhead.

The longest actual kernel lasted 1.061s in Nsight Systems. Nsight Compute
application replay of that kernel reported 3.39 active threads/warp, 15.86%
occupancy, 89.58% uniform branch targets, 22.74% long-scoreboard stalls and
0.57% peak DRAM throughput. It uses 255 registers/thread (256 allocated),
which limits L4 residency to two 128-thread blocks/SM, a 16.67% occupancy
ceiling. These are runtime-kernel counters; they include the Bend scheduler,
allocator, pointer chasing and early-halting traders. They establish low lane
utilization, but cannot attribute all of it to the interpreter opcode switch.

The replacement A100 K18 diagnostics also passed byte equality against L4:
GPU traders 1221ms, aggregation 39ms, prices/settlement 241ms, replication
1925ms, metrics 3013ms; CPU respectively 4953/23/110/2000/2870ms.
The difference from the first A100 metrics timing illustrates host variability;
neither one-epoch diagnostic is a ten-epoch timing average.

The replacement A100 Nsight capture reports 313.7 MB H→D (8,008 transfers,
53.9ms summed copy time), 511.3 MB D→H (10,474 transfers, 52.5ms), 12,452
CPU fault events and 2,322 GPU faults in 164 events (156.5ms summed).
Its longest kernel took 1.252s, with the same 128×128 grid and 255 registers
per thread. Counters again show 3.39 active threads/warp and 89.58% uniform
branch targets; occupancy is 7.35%, long-scoreboard stalls 26.24%, DRAM
throughput 0.41% of peak. The 128 blocks cover only 0.59 waves at the
register-limited two-block residency across 108 SMs. A100 therefore has a
smaller fraction of its potential warps in use, as well as the same low
active-lane count. Increasing the launch grid and reducing register pressure
are runtime experiments worth separating from opcode-dispatch changes.

## Measurement method

- K=2^18 and 2^20 on L4; K=2^20 on A100. Q=4096 questions (`--F` in the shell
  harness), fuel=256, seed=1, ten epochs. GPU and native CPU use the same binary
  and eight host threads; CPU selects `--gpu off`.
- Reported instructions are the sum of CSV `ops`. Wall time includes population
  initialization, all epoch phases, metrics, output, runtime startup and tracing.
  Compilation is excluded. These are single trials, not confidence intervals.
- Every completed run checks exact wealth conservation. Successful GPU/CPU CSVs
  and diagnostic prefixes are compared byte-for-byte.
- Every CUDA pass is measured around kernel launch plus `cuCtxSynchronize`.
  `cuda_wall_s` therefore includes launch/synchronization overhead and GPU-side
  managed-memory stalls. It is not pure arithmetic time. The remaining wall
  time includes host work, host-side faults/transfers, setup and output.
- A separate generated variant runs trader execution and sparse aggregation as
  two marked cubes, with `IO.now()` boundaries. It reports preparation, a pair
  of adjacent IO boundaries, trader stage, aggregation stage and host finishing
  work in milliseconds. It changes scheduling and creates an extra crossing;
  its stage times are diagnostic, not an additive decomposition of the fused
  baseline. The IO boundary reading of zero means below 1ms resolution.
- The refined A100 variant further separates pricing/settlement, reproduction,
  and metrics/CSV construction. Metrics include question and shadow metrics,
  population collection, genome hashing, sorting, Gini and distinct counts.
- Nsight Systems captures a separate one-epoch K18 invocation, including CPU
  and GPU unified-memory faults. Nsight Compute samples the longest actual CUDA kernel
  from that Nsight Systems capture (a runtime pass launches up to four kernels). Counter data describes that runtime kernel, including
  Bend scheduling/allocation, not just the trader opcode switch.

## Interpretation rules

Low end-to-end speedup does not establish warp divergence. First separate the
fraction spent outside CUDA, then inspect the device metrics. A kernel-only
optimization cannot eliminate host settlement, replication or metrics.

The VM dispatcher matches the current opcode, and traders can halt early, jump,
or emit different orders. Divergence is plausible inside that stage. A table
of function pointers still dispatches different work per lane; it is not by
itself branch-free SIMD. Executing every opcode and selecting a result can
replace divergence with much more work, and must mask invalid reads/writes.
Sorting by a static opcode histogram only groups similar programs: their PCs
and branch outcomes can still diverge immediately.

If interpreter divergence proves limiting after host work is reduced, the
specific candidate is wavefront execution: compact active traders and bucket
by current opcode (or PC/opcode) each step, execute uniform batches on flat U32
code/state buffers, then scatter results back to original cell identities.
Original indices, PRNG inputs and neighbour relations must stay unchanged.
This costs compaction and scheduling; it needs an isolated benchmark before
being called a speedup. Flattening the current refcounted trees can also reduce
pointer chasing and allocator/atomic traffic, independently of divergence.

For host metrics, preserve exact semantics: integer key sorting and genome
hashing can be batched; the Gini path uses ordered F32 compensated sums, so
changing its reduction order can break the CSV contract. Approximate statistics
or less frequent sampling would change the experiment and need a separate
choice, not a hidden performance shortcut.

A concrete host-side inefficiency visible in `metrics.bend:pop.ids` is eager
argument evaluation: `T.pick(k, a, code.hash(code, 0))` computes a genome hash
even for the ancestry pass, and `keep.if` receives an already-computed hash even
for an inactive trader. Moving those guards before hashing could avoid redundant
work without changing outputs. This is a source-based candidate, not a measured
fix in this experiment.

Replication is another host candidate: `rep` shares the complete population
while each leaf repeatedly looks up itself and four torus neighbours through
`T.Pop.get`, including nested target/acceptance checks. The measured A100 host profile ranks metrics ahead of replication. A device-resident settlement and
reproduction pass over flat wealth/state buffers, with only compact aggregate
and metric keys crossing to the host, addresses both tree walks and ownership
traffic. Exact event order and original cell identities remain constraints.

The stage profiles are one-epoch K18 diagnostics. The ten-epoch CUDA totals come
from the actual K18/K20 throughput runs. Later epochs can have different trading
and reproduction work; do not read the diagnostic phase row as a ten-epoch
average or as a profile of an evolved population after hundreds of epochs.

The pinned runtime also caps `CUBE_G` at 128 groups of 128 threads (16,384 lanes),
regardless of population size; each runtime pass includes growth/work kernels
and a one-block packing kernel. More traders therefore do not automatically
increase the launch grid. On a larger GPU, fixed grid size, registers/shared
memory, and queue imbalance can limit occupancy independently of opcode
branches. Tuning the runtime cube size is a separate candidate; a low active-
thread counter alone cannot distinguish these mechanisms from VM divergence.

## Reproduction and checks

The measured source and compiler hashes are in each run's `*.build-info.txt`.
`scripts/gpu-scaling-bench.sh` builds the production CUDA executable and the
separate diagnostic executable. `scripts/gpu-scaling-bench.py` selects eight
host threads, records process wall time and CUDA windows, checks the ledger,
and compares successful CSV prefixes. `scripts/gpu-profile-counter.py` selects
the longest actual kernel from the Nsight SQLite capture before replay.

Commands used for the sequential rentals:

```sh
GPU_RUN_TIMEOUT=45m infra/gpu-run.sh --up -- \
  bash scripts/gpu-scaling-bench.sh l4
GPU_MAX_RUN=28m GPU_RUN_TIMEOUT=24m timeout --kill-after=60 27m \
  infra/gpu-run.sh --up --a100 -- \
  env GPU_BENCH_OUT=/root/runs/gpu-scaling-a100-retry \
  bash scripts/gpu-scaling-bench.sh a100
```

Use a fresh output directory for any future authorized run. The wrapper owns
result copying and VM deletion; GCE also enforces the lifetime cap. During the
replacement, read-only rsync backups preserved line-buffered epoch output.
The first A100 was preempted before K20 completion, so its empty K20 output is
recorded as unavailable, never as zero throughput. Agent Mail #2100 clarified
one replacement, at most 28 minutes, with no further retry after preemption.

An isolated `make check` passed: typechecks, market/questions/VM/soup/shadow
suites, Python/matrix checks, 256×10 smoke, and exact snapshot resume.
See `runs/gpu-scaling-l4/make-check.log` for the captured output. Production
Bend sources have no changes since the completed cube implementation.

All three sequential rentals were deleted: L4 25.987min, preempted A100 8.199min, replacement A100 21.483min; total 55.669min. Rate ceilings are unverified $0.87/h (L4) and $3.69/h (A100); combined compute estimate $2.2022. These are rate-ceiling estimates, not a billing invoice. Instance and disk lists were empty, checked 2026-09-27T12:21:21Z; operation timestamps and empty-list receipts are committed with each run.
