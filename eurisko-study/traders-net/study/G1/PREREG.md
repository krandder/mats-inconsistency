# G1 PREREG: GPU broker smoke (written before the first run, 2026-09-27)

G1 is infrastructure, not an experiment; this pre-registers the one smoke run the brief allows.

- **Arm:** one queue job `g1-smoke` (source `/tmp/traders-net/G1/smoke-src/run.sh`, copied into
  `queue/`), HOURS=0.1, submitted with `infra/submit.sh` and dispatched by the cron queue runner,
  not by hand. One spot L4 (g2-standard-8), GCE max-run-duration 26 min.
- **Metric (pass/fail):** all of (1) `out/smoke.json` synced back with `torch.cuda.is_available()`
  true on an NVIDIA L4; (2) job lands in `queue/done/`; (3) GPU-LOG.md has CREATE, READY, RUN,
  RUN-END rc=0, DELETE and COST-AUDIT lines for the VM; (4) `infra/zero-vm-check.sh --log` exits 0
  afterwards; (5) COST.md total equals the COST-AUDIT cost.
- **Seeds:** 1 (a plumbing check has no seed variance; the 3-seed rule applies to experiments).
- **Stop rule:** at most 2 smoke attempts. If the first fails, fix and retry once; if the second
  fails, stop, delete everything, confirm zero VMs, and report the queue as not proven.
- **Cost cap:** G1 cap $0.50 (infra/caps.tsv); expected ~$0.15-0.25 at the $0.87/h ceiling.

## Amendment 2026-09-27T22:43:20Z (before attempt 3)

Attempt 2 (g1-smoke2) did not fail in the broker: GCE preempted the spot VM 3.6 min after insert
(`infra/state/audit/tn-gpu-g1-smoke2-09272232-25494/operations.json`, compute.instances.preempted),
before READY. The broker then waited blind; now gpu-up exits 7 on a vanished VM and job-run requeues a
preempted job at most twice. Deviation: one more attempt (g1-smoke3) is allowed because a spot
preemption is not a broker failure; stop rule otherwise unchanged; G1 spend stays under its $1.00 cap
(ledger $0.28 + reserve $0.38).
