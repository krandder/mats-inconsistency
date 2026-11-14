# L4 scaling receipts

Spot g2-standard-8, one L4, eight runtime host threads. Seed1, Q4096, fuel256.
Production source unchanged; `soup.build-info.txt` identifies exact source/binary.

| K | epochs | GPU instr/s | CPU instr/s | GPU seconds | CPU seconds | GPU/CPU |
|---|---:|---:|---:|---:|---:|---:|
| 2^18 | 10 | 2,960,628 | 2,324,238 | 91.773 | 116.901 | 1.274 |
| 2^20 | 10 | 2,460,876 | 2,015,083 | 437.426 | 534.196 | 1.221 |

`summary.json` contains exact ledger checks and byte-identical CSV comparisons.
GPU pass wall totals: 11.161s (K18), 41.719s (K20). These include synchronization
and device-side managed-memory stalls. Remaining wall time is outside those
windows; it includes initialization, host computation and host-side faults.

The one-epoch split K18 profile measured preparation/boundary/traders/aggregation/
host finish in milliseconds: GPU 8/0/1047/36/5370, CPU 8/0/4665/23/5164. Splitting
adds a crossing; these are diagnostics, not an exact decomposition of fused runs.

Nsight Systems: `nsys.nsys-rep`, `nsys.sqlite`, `profile-summary.json`.
Counters: `counter.csv`, `counter-selection.json`, `counter-command.json`,
`counter-summary.json`. The longest actual kernel (filtered index2, grid128 x
block128) was sampled with application replay; counter collection succeeded.
An earlier pass-index/kernel-replay attempt (`ncu.*`, retained locally) failed
with code9 and is not used in the analysis. Hardware counter timing is excluded
from the throughput table. Nsight runtime/device metrics include Bend's scheduler
and allocator; they do not isolate the VM opcode switch.

Copy-engine totals and GPU fault durations overlap kernel execution. CPU fault
counts have no durations in this capture. Do not add these quantities into a
wall-time decomposition. These are single trials and one-epoch diagnostic
samples, not a profile of a mature evolved population.
