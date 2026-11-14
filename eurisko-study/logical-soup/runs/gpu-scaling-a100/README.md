# First A100 attempt: interrupted by spot preemption

The A100-SXM4-40GB in us-central1-b was system-preempted at
2026-09-27T11:53:20Z before the requested K20 ten-epoch run completed.
`operations.json` records `compute.instances.preempted`, user `system`.
Only the earlier one-epoch K18 split profiles were backed up. No K20
instructions/s result is available; the empty gpu-k20.raw is not a zero rate.

K18 profile, milliseconds (GPU / native CPU):

| phase | GPU lane | CPU lane |
|---|---:|---:|
| trader execution | 1204 | 4741 |
| sparse aggregation | 40 | 25 |
| prices and settlement | 255 | 120 |
| reproduction | 1964 | 2145 |
| metrics and CSV construction | 4359 | 4359 |

Preparation was 8ms and the adjacent IO boundary measured 0ms (below 1ms
resolution). Both successful one-epoch CSVs satisfy the exact wealth ledger;
their byte equality was rechecked after recovery.

The failed allocation in us-central1-a never produced a VM. The successful
allocation in us-central1-b lasted 491.939s (8.199min), costing an estimated
$0.50424 at the unverified $3.69/h ceiling. VM and disk lists were empty after
preemption. The original script's final rsync failed because the VM was gone;
these files came from a read-only backup during the run.

Agent Mail #2100 from eurisko clarified that one replacement is permitted,
under 28min and combined under 63min, with no further replacement after another
preemption. This is an agent-originated experiment clarification, not a Kelvin
quotation. Standing monetary authorization is GOAL.md. Replacement results use
a different directory: runs/gpu-scaling-a100-retry/.
