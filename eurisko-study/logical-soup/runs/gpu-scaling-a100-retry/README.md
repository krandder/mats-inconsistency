# A100 scaling replacement

Completed on 2026-09-27 after the first spot A100 was preempted. Same seed 1,
Q=4096, fuel=256, eight host threads, ten epochs, same production executable
for GPU and native CPU (`--gpu on/off`). Main rates include initialization,
host phases and output; they exclude compilation and profiler replay.

| K | Instructions | GPU seconds | GPU instr/s | CPU seconds | CPU instr/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|---:|
| 2^20 | 1,076,450,270 | 436.249808 | 2,467,508.87 | 534.455203 | 2,014,107.57 | 1.2251x |

`summary.json` contains timings and exact ledger checks. `validation.json`
records the independent CSV/ledger recheck: all ten A100 GPU/CPU epochs match
L4 byte-for-byte. `profile-*.csv` are K18 one-epoch split diagnostics, also
identical to L4's first epoch. `profile-summary.json` describes a separate
fused Nsight capture; `counter-summary.json` describes application replay of
its longest actual kernel. Neither profiler invocation contributes to the
reported main throughput. Raw captures and source/compiler hashes are kept.

3.39 active threads/warp and 7.35% occupancy do not identify opcode branching
as the sole cause: the sampled kernel includes Bend scheduling/allocation.
CUDA windows occupy just 10.66% of whole-program time. Host metrics and
replication dominate the split profile. See ../../docs/gpu-scaling.md.

Line-buffered output and periodic read-only backups preserved completed epochs
throughout this replacement. `counter-tmp/` is an uncommitted profiler lock
directory, not measurement data.

All three sequential rentals were deleted: L4 25.987min, preempted A100 8.199min, replacement A100 21.483min; total 55.669min. Rate ceilings are unverified $0.87/h (L4) and $3.69/h (A100); combined compute estimate $2.2022. These are rate-ceiling estimates, not a billing invoice. Instance and disk lists were empty, checked 2026-09-27T12:21:21Z; operation timestamps and empty-list receipts are committed with each run.
