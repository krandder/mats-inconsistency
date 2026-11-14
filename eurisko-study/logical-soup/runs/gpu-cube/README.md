# CUDA experiment, pool-on8tb

One spot g2-standard-8, L4, Bend2.0.29, clang19.1.1, 8 CPU threads.
Seed1, fuel256; K14/K16 use Q=4096. Ten epochs per invocation; single trial,
wall time includes startup, all host phases, CSV and CUDA tracing. No warmup
subtraction. GPU cubin compilation is outside timings.

| K | lane | seconds | instructions/s | CUDA passes |
|---|---|---:|---:|---:|
| 2^14 | js | 6.337 | 2,749,188 | — |
| 2^14 | cpu | 5.488 | 3,174,678 | — |
| 2^14 | gpu | 4.783 | 3,642,400 | 246 |
| 2^16 | js | 2.877 | 0 (failed before epoch 1) | 0 |
| 2^16 | cpu | 23.840 | 2,912,110 | — |
| 2^16 | gpu | 18.221 | 3,810,187 | 280 |

K14 all lanes have byte-identical CSVs. K16 GPU and CPU match exactly; original
JS on the VM failed with a machine-stack error, so the K16 summary's initial
comparison reference is CPU. The original JS with enlarged OS/JSC stacks was
then run on General: its complete CSV matches K16 GPU byte-for-byte. Every
successful CSV satisfies the exact wealth ledger on every epoch.

General-only JS K16 timing: 40.999s, 1,693,316 instr/s. This is NOT a same-host comparison. The VM had already entered automatic deletion; no second VM was rented.

`source-commit.txt` is the pre-edit base commit, not a claim that the run used an
unchanged checkout. `soup.build-info.txt` records hashes of the exact source and
emitted C/binary used. The committed src/soup.bend matches that hash. Original
JS was compiled from that base commit before edits. `js-stack-trace.txt` unmasks
the JS RangeError in recursive List.length. docs/gpu.md documents the stack
workaround, now included in the repeatable benchmark script.

`make-check.log` is the final isolated full make check (including the concurrently
added matrix tests). Smoke output was redirected under build, so live calibration
outputs were untouched. `cost-audit.json`, operations.json and deletion JSONs
record 277.980 seconds from insert to completed deletion, at the unverified
$0.87/h ceiling ($0.0671785). Instance and experiment disk lists were empty.
