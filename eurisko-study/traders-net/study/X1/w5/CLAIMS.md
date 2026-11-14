# X1 worker 5: V1 and G1 claim checks

The worker checked each V1 and G1 claim against its files, then reproduced it with fresh seeds 4 and 5 on one G1 GPU job (`x1-w5-wn45`, audited cost $0.4518).

- Pre-registered plan: `w5/PREREG.md`, written 12:04Z before the first run.
- Paths below are relative to `study/X1/w5/` unless marked `G1/` or `V1/`.
- Status is one of verified, refuted or unverified.
- The rerun commands assume `W5=/home/agent/workspaces/eurisko-study/traders-net/study/X1/w5`; run them from `/tmp`.

## V1 (Grafana view of wealth-net)

| # | Claim (V1.md / V1-w*.md) | Status | checked_at | Evidence | Rerun |
|---|---|---|---|---|---|
| V1.1 | The dashboard check passes (hermetic, exporter, Grafana) | verified | 2026-09-28T12:08:43Z | `v1/check-20260928T120843Z.json` (rc 0) | `cp -r V1/{check.py,grafana,SCHEMA.md} /tmp/x && cd /tmp/x && mkdir -p check && python3 check.py` |
| V1.2 | Panels 12 (ec_*) and 13 (price_*) are PENDING | refuted, now superseded | 2026-09-28T12:08:43Z | same JSON: both PASS. Panel 12 has 15 M5 runs; panel 13 has the K1 runs. Panel 2 is SKIP by design. | as V1.1 |
| V1.3 | summary-at-common-step.json (14 quantities × 3 seeds at step 23300) matches the V1 CSVs, the live exporter and the raw snapshots | verified | 2026-09-28T12:12:56Z | `v1/recompute_summary.json`: max relative difference 0 (CSV), 0 (exporter), 2.8e-7 (independent raw reader; float32 sum of wealth) | `python3 $W5/v1/recompute_summary.py` |
| V1.4 | V1.md rounded table (R 1962±181, head −1241±69, neuron 1610±235, pi +621/−3479/−8465/−25326, Gini 0.64/0.40, neg 34%, churn 0.003, turnover −0.27) matches the summary JSON | verified | 2026-09-28 | `V1/w1/summary-at-common-step.json` vs `V1/V1.md` | read both |
| V1.5 | The summary JSON has a script behind it | unverified | 2026-09-28 | No generating script exists in V1/. `v1/recompute_summary.py` now fills that gap. | as V1.3 |
| V1.6 | Fresh seeds reproduce the snapshot at step 23300 (\|z\| ≤ 2 vs 3-seed mean ± sd) | 10 verified, 4 unverified, 0 refuted | 2026-09-28T12:38Z | `v1/fresh_seed_z.json`. serve.wn_csv and the independent reader agree within 4.5e-7. | `python3 $W5/v1/fresh_seed_z.py /tmp/traders-net/X1w5/wn45` |
| V1.6a | R_total, wealth_out, wealth_neuron, all four pi-family wealths, neg_neuron, churn_neuron, turnover_neuron | verified | 2026-09-28T12:38Z | z-scores in `v1/fresh_seed_z.json`; worst is −1.71 (wf_out_pi, seed 4) | as V1.6 |
| V1.6b | head wealth −1241±69 | unverified (sd understated) | 2026-09-28T12:38Z | seed 4 = −358 (z +12.8), seed 5 = −1112 (z +1.9). Pooled over 5 seeds: −1039 ± 388, 5.6× the 3-seed sd. The sign is negative in 5 of 5 seeds. | as V1.6 |
| V1.6c | comp wealth 4147±267 | unverified (sd understated) | 2026-09-28T12:38Z | seed 4 = 6083 (z +7.3), seed 5 = 4403 (z +1.0). Pooled: 4585 ± 865, 3.2× the 3-seed sd. | as V1.6 |
| V1.6d | gini_out 0.40±0.03; gini_neuron 0.64±0.02 | unverified (one seed each at z 2.45 / 2.55) | 2026-09-28T12:38Z | pooled 0.380±0.040 and 0.649±0.034. The sign and the ordering (neurons more unequal than readouts) hold in 5 of 5 seeds. | as V1.6 |
| V1.7 | Wealth-net runs stopped at the common step 23300 | refuted (stale) | 2026-09-28 | The snapshots now run to step 31100–31200 in all 3 seeds (mtime 09-27 22:28Z). The 23300 table is a mid-run cut. | `ls wealth-net/results/train-s*/snapshots \| sort -V \| tail` |

The fresh-seed arm is valid for these reasons:
- It runs the same code: a copy of `wealth-net/gpu/*.py`, in `gpu-job/`.
- LR is constant. The stream (T=24000) and the batch generator depend only on the seed, so `--steps 23400` does not change the trajectory up to step 23300.
- The speed was 66.1 ms/step, with 2 runs in parallel on 1 L4.

## G1 (GPU queue and ledger)

| # | Claim (G1 README / COST.md / GPU-LOG) | Status | checked_at | Evidence | Rerun |
|---|---|---|---|---|---|
| G1.1 | The infra tests pass (13) | verified | 2026-09-28T12:06:01Z | `g1/test_infra.out` (13 OK, rc 0, from a clean /tmp copy) | `cd /tmp/traders-net/X1w5/g1copy/study/G1 && python3 -m unittest discover infra/tests` |
| G1.2 | Every COST-AUDIT equals the GCE receipts (insert → delete/preempted) × $0.87/h | verified | 2026-09-28T12:38:46Z | `g1/ledger_check.json`: 23 of 23 VMs, max abs diff $0.000000. The code is independent of cost-audit.py. | `python3 $W5/g1/ledger_check.py` |
| G1.3 | COST.md total and per-workstream sums | verified | 2026-09-28T12:38:46Z | Total re-summed $6.0673 against "$6.07" (was $5.62 before X1). G1 0.4284, M1 1.2709, M6 2.2074, M7 1.4087, M9 0.3001, X1 0.4518. | as G1.2 |
| G1.4 | Smoke spend $0.43 = 4 VMs | verified | 2026-09-28T12:38:46Z | 0.2194+0.0582+0.0492+0.1016 = 0.4284 | as G1.2 |
| G1.5 | Caps M1 3, M6 7, M7 22, M8 5, M9 4, X1 4, G1 1; the $45 total holds regardless | verified | 2026-09-28 | Caps sum to $46 > $45, but `gpu-up.sh:32` and `queue-runner.sh:45` refuse when ledger + reserve > $45 independently of caps.tsv. | `cat G1/infra/caps.tsv; grep -n BUDGET_USD G1/infra/*.sh` |
| G1.6 | At most 2 VMs at once | verified from receipts | 2026-09-28T12:38:46Z | Receipt intervals: max concurrent 2, 5 overlapping pairs, 0 triples | as G1.2 |
| G1.7 | Cron runs queue-runner every 2 min and dispatches jobs | refuted (partly) | 2026-09-28T12:06:01Z | `g1/runner_jam.json`. job-run.sh inherits the runner's flock fd 7 (`queue-runner.sh:11` and the setsid line). While any job runs, cron dispatches nothing and skips orphan detection (stop layer 4). Live stopgaps: runner.log 06:54Z, 07:10Z, 07:31Z, 09:56Z, 10:06Z. The hermetic test shows adding `7>&-` to the setsid line fixes it; this has not been applied. | `python3 $W5/g1/test_runner_jam.py` |
| G1.8 | Preemption mid-run is requeued (≤ 2×) | verified (live) | 2026-09-28 | GPU-LOG: m7-w2-famwealth preempted (receipt compute.instances.preempted), rerun 08:18 rc 0; m6-s2-noise2 preempted at 11:07, rerun rc 0. There are 5 preempted receipts in total. | as G1.2 (`preempted_vms`) |
| G1.9 | Preempted spot VMs leave no instance or disk | verified | 2026-09-28 | `G1/infra/state/audit/<vm>/{instances,disks}.json` = [] for all 5 | `cat` those files |
| G1.10 | Zone shuffle on create | verified | 2026-09-28 | CREATE-FAILED zone orders differ ("b c a", "a b c"). ZONE_RESOURCE_POOL_EXHAUSTED failures cost $0 (no receipt). | as G1.2 |
| G1.11 | Zero traders-net VMs or disks after jobs | verified | 2026-09-28T12:38:34Z | `g1/zero_vm_readonly.json`: 0 instances, 0 disks, 0 running jobs. This was a read-only gcloud run, with no write to G1/state. | see the `cmd` in that JSON |
| G1.12 | The L4 smoke benchmark (fp32 matmul) runs at 11.74 and 11.51 TFLOPS | verified | 2026-09-28T12:36:20Z | Fresh result 11.41 TFLOPS against 11.625 ± 0.163: z = −1.32. Same torch 2.14.0+cu130 and CUDA 13.0. `g1/smoke-x1.json` | G1 job with `gpu-job/run.sh` |
| G1.13 | create→READY takes a few minutes | verified | 2026-09-28T12:38:46Z | 17 READY VMs: median 3.63 min, min 3.07, max 12.55. X1 took 3.15. | as G1.2 |
| G1.14 | Admission refuses a job over its cap | unverified (live) | 2026-09-28 | `queue/refused/` is empty, so this was never exercised live. It is covered only by the unit tests. | as G1.1 |
| G1.15 | The audit handles the `compute.instances.maxRunDurationTermination` receipt | unverified | 2026-09-28 | That op type was never seen: op types in receipts are insert 34+, delete, compute.instances.preempted. GCE docs confirm only `compute.instances.preempted` ([Spot VMs](https://docs.cloud.google.com/compute/docs/instances/create-use-spot), [preemption history sample](https://docs.cloud.google.com/compute/docs/samples/compute-preemptible-history)). If the name is wrong, a max-run-killed VM would fall back to the DELETE estimate or the reserve, which is conservative. | — |
| G1.16 | $0.87/h is a ceiling | plausible | 2026-09-28 | Third-party trackers list the us-central1 g2-standard-8 price at about $0.85 on-demand and about $0.71 spot ([Economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/g2-standard-8/), [Holori](https://calculator.holori.com/gcp/vm/g2-standard-8)). Not checked against the Billing API. | — |
