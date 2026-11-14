# Phase 3: three K16 GPU seeds

Completed and checked 2026-09-27T16:51:55Z for agent-originated bead `pool-sddpv`.
All three seeds reached 5000 epochs on spot L4 VMs. The final group instance
and disk lists were empty at 2026-09-27T16:51:46Z.

| Seed | Epochs | Virtual instructions | Successful chunk wall seconds | M instructions/s |
|---|---:|---:|---:|---:|
| 1 | 5000 | 39,172,227,238 | 11,421.76 | 3.429614 |
| 2 | 5000 | 41,843,398,687 | 12,494.83 | 3.348857 |
| 3 | 5000 | 39,224,723,208 | 11,563.03 | 3.392253 |

The configuration matches the K16/F12 baseline cells of `matrices/phase3.json`:
K=65536 traders, F/Q=4096 questions, fuel=256, b=64, zero free income,
100-epoch chunks, all prices visible, no held-out family. Each run has 50
contiguous timing rows and 5000 metric rows. Rates divide virtual trader
instructions by successful chunk wall time, including process startup and
snapshot load/save. They exclude allocation, builds, sync and discarded work
from interrupted chunks. These long-run rates are not paired CPU comparisons.

## Verification and source

For every CSV row, wealth equals prior wealth plus trader P&L minus instruction
charges and burned victim wealth. Final snapshots match CSV wealth and cumulative
instructions exactly; timing instruction totals also equal the sum of CSV ops.
Each final `verification.json` records these checks and SHA256 hashes.

Every rental used immutable control/source commit
`103c7fcae6913fffd78ec99e804f2aca25329238`, Bend 2.0.29,
clang 19.1.1, CUDA dispatch enabled and eight host threads. `origin.json`
records the commit because the VM's rsynced `config.json` reports commit unknown.
All eight source hashes match the build manifests. The guarded-hash utilisation
experiment was isolated and was not applied to these runs. Native snapshot
resume smoke checks (5+5 versus straight 10) and CUDA traces are preserved.

Recheck the final datasets from the repository root:

```sh
for seed in 1 2 3; do
  python3 scripts/verify-checkpoint.py "runs/phase3/k16-f12-s$seed" 5000
done
gcloud compute instances list --project=GCP_PROJECT --filter=labels.soup-group=pool-sddpv
gcloud compute disks list --project=GCP_PROJECT --filter='name~^soup-gpu-phase3-k16-f12-s'
```

## Recovery and cost

Three GCE-confirmed spot preemptions occurred: seed 3 at saved epoch 4000,
seed 2 at 4000, and seed 2 again at 4800. Seed 1 also lost SSH at epoch 4700;
its operations show our cleanup deletion, so this is not classified as a
confirmed preemption. Every interruption resumed from the last verified local
snapshot, with exact accounting at the next checkpoint.

After the last preemption, L4 allocation failed in all three supported
us-central1 zones. The coordinator stopped with no VM or disk left. One manual
restart of the same frozen coordinator after a short capacity pause obtained
an L4 and completed the final 200 epochs. No source/configuration change,
A100 substitution or on-demand rental was used. Six earlier allocated VMs
failed before any epochs (timeout format and remote privilege ordering);
these are included in the cost total.

Total: **13 allocated rentals**, one allocation request
rejected in all three zones, **10.780962 aggregate VM hours**,
and **$9.379437 estimated ceiling** against the **$40** leaf cap.
The unverified rate ceiling is $0.87/hour including disk, not a billing invoice.
All allocated time, failed starts and discarded partial chunks count in costs.
The rejected allocation has zero VM time, supported by three failed insert
operations plus empty instance/disk checks. Watchdog rules and six-hour DELETE
TTLs were unchanged. The longest rental was
3.045 hours.

Evidence: [completion summary](../runs/phase3/infra/completion-summary.json),
[final deletion](../runs/phase3/infra/final-deletion.json), per-VM operation/cost
receipts and supervisor logs under `runs/phase3/infra/`, and
[GPU-LOG](../GPU-LOG.md). Scientific interpretation belongs to the matrix report;
this report establishes completion, reproducibility, recovery and cost.

## Cross-model closeout review

Eurisko accepted commit `1c0e4f6`: **SHIP**, Agent Mail #2179,
2026-09-27T16:54:46Z. The reviewer checked all three CSVs (5000 data rows plus
one header each), agreement between this report and the completion summary,
empty live VM/disk lists at 16:54Z, the $9.38 ceiling under $40, and recovery
from three confirmed preemptions plus one SSH loss.
