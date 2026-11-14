# P19 deviations from PREREG.md (dated; none changes the verdict inputs or a kill rule; Deviation 1 lowers comparator replicates)

## Deviation 1 (2026-09-30 01:52Z): fewer comparator replicates, to stay under the $5 cap

When this was decided, no test-seed job had finished and no test-seed result had been read.

**Why.** Measured CPU per job was far above the plan's estimate. The estimate was about 5,900 CPU-s per seed, which
was wrong.

| job | measured CPU (pilot, seed 99) | count per seed as frozen |
|---|---|---|
| truthrep | 6,564 and 8,123 CPU-s | 2 |
| r7brep | 2,070 CPU-s | 12 |
| pricerep | more than 5,600 CPU-s and still running when stopped | 8 |

The frozen queue would take about 6 VM-hours, about $7.40 at the $1.20/h ceiling, over the $5 cap.

**Change.** The comparator rules get fewer replicates.

| rule | replicates, frozen | replicates, now | seeds |
|---|---|---|---|
| R7b | 12 | 4 (reps 0 to 3) | every seed |
| R1 and R4 (pricerep) | 8 | 2 (reps 0 and 1) | every seed |

- The 60 dropped jobs are listed in `queues/skipped_deviation1.txt`. The runner skips them through placeholder
  markers containing "SKIPPED-deviation-1".
- Seed 0 already had R7b reps 0 to 11 running, and seed 1 reps 0 to 5. They finish, but the primary comparator
  taus use reps 0 to 3 on every seed.
- The analysis runs with `P19_KREP_W=4 P19_KREP_P=2`.
- The pilot's timing-only pricerep (seed 99) was stopped.

**What does not change.**

- Truth: 2 replicates per seed.
- FR and FO, which come from `frfo`.
- K1 and K2, the gates and the primary FR compute ratio. That ratio uses the per-replicate R7b work, which does not
  depend on how many replicates run.

**Effect.** R1, R4 and R7b ranks are noisier than in P8. RESULTS reports them with their replicate counts, and
their per-replicate-count curves are in the chart JSON.

## Deviation 2 (2026-09-30 01:56Z): serialization bug fix in analyze19.py (no statistic changes)

A dry run of `analyze19.py` on synthetic files built from the seed-99 pilot outputs (before any test-seed result
existed) crashed when writing `p19_summary.json`: the gate flags were numpy booleans, which `json` cannot write.
The fix wraps the three gate flags in `bool()` and adds `default=float` to that one `json.dump`. No computation
changed. The dry run then reproduced the pilot's numbers (FR tau 0.611, FR compute ratio 0.0407).

- frozen sha256 of `code/analyze19.py`: `bf388a4d…d79880`
- after the fix: `c2d46e5cdc01b933b09404272ac98c9bd0f7981758f7c7e8de5fc8c1593e7183`

## Deviation 3 (2026-09-30 02:10Z): spot preemption, restart on an on-demand VM with a smaller queue

When this was decided, no test-seed job had finished; nothing from seeds 0 to 4 had been read.

- **What happened.** GCE preempted the spot VM `trader-p19-1` at 01:59:13Z (operation `compute.instances.preempted`;
  termination action DELETE, so the boot disk went with it). The main queue had run 28 minutes and no test-seed job
  had finished. The files synced before that point hold only the pilot outputs. Booked spend: $0.85 at the ceiling.
- **Restart.** VM `trader-p19-2` (us-east1-b), c2d-highcpu-32, **on-demand** (`P19_PROV=STANDARD` in `vm.sh`), max-run 3h,
  termination DELETE. It books the same ceiling rate ($1.20/h is the on-demand list price), so the $5 accounting does
  not change. A second preemption would have left too little budget for a full rerun.
- **Queue `queues/main2.txt` (45 jobs).** Only the jobs the analysis reads after Deviation 1: 10 truthrep, pricerep
  reps 0 and 1, r7brep reps 0 to 3, 5 frfo. The command lines are the frozen `main.txt` lines unchanged, except
  `P8_THREADS=2` (was 1), run with 16 slots (was 28). The seed-0 R7b reps 4 to 11 and seed-1 reps 4 and 5, which
  Deviation 1 let finish but never used, are not run.
- **Effect.** None on any statistic. The thread count changes only floating-point reduction order in torch. The
  compute accounting counts MACs, not CPU time, so it is unaffected.
