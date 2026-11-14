# X1 worker 5 PREREG: skeptic + reproduction of V1 and G1

Written 2026-09-28T12:04Z, before any rerun. Scope: V1 (live wealth view) and G1 (GPU broker),
claimed in `../STATUS.md`. Outputs under `study/X1/w5/`; bulky files under `/tmp/traders-net/X1w5/`.
Workstream files are read, never edited; reruns run on copies.

## V1
- **V-a check rerun.** Copy `study/V1/{check.py,grafana,SCHEMA.md,PREREG*.md}` to /tmp, run the check
  against the live exporter/Grafana, JSON result into `w5/v1/`. Pass = no FAIL. Panels 12/13 status
  recorded (M5 claims ec_* columns exist now).
- **V-b recompute.** The 15 quantities of `V1/w1/summary-at-common-step.json` recomputed (i) from V1's
  committed CSV copies, (ii) from the live exporter, (iii) by an independent reader of the raw
  wealth-net snapshot JSONs that does not import serve.py. Pass = |diff| <= 1e-6 relative to the
  rounding of the source (7 significant digits).
- **V-c fresh seeds.** Wealth-net's exact GPU code (`wealth-net/gpu/`, the copy that produced
  train-s1..s3), seeds 4 and 5 (never used), 23400 steps, same config, one G1 job (`x1-w5-wn45`,
  HOURS<=1.0). Quantities at step 23300 via serve.py's own derivation. Metric per quantity:
  z = (fresh - mean3) / sd3. Verdict: verified if |z| <= 2 for both seeds; refuted if |z| > 2 for
  both; unverified (mixed) otherwise. Seed 4 is the pre-declared primary; seed 5 is the second draw.
  Caveat pre-stated: sd from n=3 is noisy, so a few misses among ~10 quantities are expected by chance.

## G1
- **G-a tests.** `infra/tests/test_infra.py` in a /tmp copy: pass = 13/13.
- **G-b ledger.** Each COST-AUDIT cost recomputed from its receipts (`state/audit/<vm>/operations.json`,
  insert -> delete/preempt x $0.87/h); COST.md per-ws and total re-summed; smoke $0.43 = 4 VMs.
  Pass = every VM within $0.001, totals exact to the cent.
- **G-c caps.** caps.tsv sums to $46 (claim).
- **G-d zero VMs.** Same gcloud filter as zero-vm-check.sh, run read-only by me (no write to G1/state).
- **G-e fresh live job.** The V-c job doubles as a fresh smoke: its run.sh first runs G1's smoke matmul.
  Pass = job lands in queue/done with rc 0, CUDA true on an NVIDIA L4, torch 2.14.0+cu130, TFLOPS within
  2 sd of the two prior smoke values (11.74, 11.51), GPU-LOG shows CREATE/READY/RUN/RUN-END/DELETE/
  COST-AUDIT, zero VMs afterwards. Create->READY time compared with the claim "about 3.5-4 min" over
  every READY line in GPU-LOG.md.
- **G-f dispatcher.** Hermetic test in /tmp of the fd-7 lock inheritance reported in runner.log at
  06:54Z (queue-runner jams while any job runs). Pass/refute of the claim "cron dispatches jobs, at most
  2 VMs at once".
- **G-g unverified items.** From logs/receipts only: mid-run preemption requeue exercised?, 2 VMs at
  once (and never 3)?, zone shuffle exercised?, cap refusal exercised?, `maxRunDurationTermination`
  op name (GCE docs, cited URL).

## Stop rule and cost
- GPU: one job, reserve <= $1.16 (1.0 h + 20 min at $0.87), expected ~$0.65, from X1's $4 cap. If it is
  preempted to `failed/` (3 preemptions) or refused, report V-c/G-e as unverified; no second job.
- CPU: at most 2 concurrent jobs, only if `uptime` load < 40.
- Seeds: V-c uses 2 fresh seeds (4, 5) against the 3-seed V1 values (mean +- sd). Everything else is
  deterministic arithmetic or a pass/fail check.
