# G1: GPU queue for traders-net (every GPU minute goes through here)

Spot L4 VMs (g2-standard-8, 1x L4 24 GB, 8 vCPU, 32 GB) in `GCP_PROJECT`, one VM per job, at most
**2 VMs at once**, a **$45 total cap** plus per-workstream caps, priced at an **unverified $0.87/h
ceiling**. Adapted from `../../../logical-soup/infra/` (6 h auto-delete, watchdog, cost audit), with
its own label (`traders-net`), name prefix (`tn-gpu-`), SSH key and ledger, so it never touches the
soup's or wealth-net's VMs and their watchdog never touches ours.

**Proven 2026-09-27** by the smoke job `g1-smoke3` (`queue/done/g1-smoke3/status.json`, `out/smoke.json`:
NVIDIA L4, torch 2.14.0+cu130, 11.5 TFLOPS fp32 matmul; ~4 min from create to READY; auto-requeued once
after a spot preemption). Smoke total $0.43 ceiling over 4 VMs (COST.md).

All paths below are relative to `study/G1/`. `G1=/home/agent/workspaces/eurisko-study/traders-net/study/G1`.

## Submit a job (the only command most workstreams need)

1. Make a source dir (in your own `study/<WS>/`) with a `run.sh`. On the VM it runs as root with
   cwd = a copy of that dir, `$OUT=/root/out`, and `python` = `/opt/tn/bin/python` (CUDA torch,
   numpy, scipy, pandas, matplotlib; `pip install` more inside run.sh if needed). Write results to
   `$OUT`. Keep the dir small (≤300 MB): code, not data; generate or download data in run.sh.
2. Submit:

       $G1/infra/submit.sh --ws M7 --name stage2-s1 --src $STUDY/M7/gpu-job --hours 2 \
           --results-to $STUDY/M7/results/stage2-s1

   - `--hours H` (≤5.5): run.sh is killed after H hours; the VM auto-deletes at H h + 20 min
     (≤6 h). The admission reserve is $0.87 × (H h + 20 min); ask for what you need, not more.
   - `--results-to` must be under your own `study/<WS>/` or `/tmp/traders-net/`; default is
     `queue/<state>/<job>/out/`. Synced every ~5 min and at the end: `*.csv *.json *.png *.md *.txt
     *.log` only, each ≤100 MB. Add `--sync-include '*.pt *.npz'` for other small files.
   - Job id = `<ws>-<name>` lowercase, e.g. `m7-stage2-s1`; must match `^[a-z][a-z0-9-]{0,30}$`.
   - One job = one VM = one setup (~3-5 min boot + torch install). Put all seeds of an arm in
     one run.sh (the L4 fits several small runs in parallel) rather than one job per seed.
3. Dispatch is automatic: cron runs `infra/queue-runner.sh --once` every 2 min. To start it now:
   `$G1/infra/queue-runner.sh --once`.

## Watch your job

    ls $G1/queue/{pending,running,done,failed,refused}/      # where it is
    tail -f $G1/queue/running/<job>/job.log                   # broker log (VM, sync, exit)
    cat $G1/queue/done/<job>/status.json                      # final state, run.sh rc, VM, times
    cat <results-to>/run.log                                  # your run.sh stdout+stderr
    cat $G1/queue/refused/<job>/REFUSED.txt                   # why a job was refused (budget)

States: `pending` → `running` → `done` (run.sh rc 0) or `failed` (rc≠0, 124 = timeout, VM
preempted, upload failure). A spot preemption (before READY or mid-run) sends the job back to
`pending/` automatically, at most twice per job (count in `<job>/preemptions`); a third one lands in
`failed/`. A requeued job reruns run.sh from the start on a fresh VM, so make run.sh safe to rerun
(write each seed's final files atomically, skip seeds whose final file already exists in $OUT only if
you also ship them in the job dir; the simplest is to accept a full rerun). A job that would push the ledger over $45 or its workstream over its cap goes
to `refused/` without creating anything.

## Budget and ledger

- `COST.md` (generated, do not edit): total, remaining, per workstream, per VM, live VM count
  with `checked_at`. Refresh: `python3 $G1/infra/cost-report.py`.
- `GPU-LOG.md`: the machine ledger (PENDING-CREATE, CREATE, READY, RUN, RUN-END, DELETE,
  COST-AUDIT, ZERO-VM-CHECK lines). Per VM the ledger counts the COST-AUDIT cost (GCE insert→delete
  operation receipts × rate) if present, else the DELETE estimate (age × rate), else the full reserve.
- Per-workstream caps in `infra/caps.tsv` (from STUDY-PLAN §3): M1 $3, M6 $7, M7 $22 (stage 1 $5 +
  stage 2 $17), M8 $5, M9 $4, X1 $4, G1 $1.00. Workstreams not listed have $0 (CPU only).
  Changing a cap is the orchestrator's call; the $45 total holds regardless.
- Quick numbers: `bash -c 'source $G1/infra/lib.sh; spend_usd; spend_usd M7'`.

## Stop guarantee (four layers)

1. GCE: every VM is spot with `--max-run-duration` = job hours + 20 min (≤6 h) and
   `--instance-termination-action=DELETE`; the boot disk auto-deletes.
2. `infra/job-run.sh` deletes its VM on every exit path (trap), then audits the cost.
3. `infra/gpu-watchdog.sh` (cron, every 15 min): deletes any traders-net VM older than 6.25 h,
   without the label, or not owned by a job in `queue/running/` after 30 min; posts one line per VM
   per hour to Fleet - Status; audits pending costs; regenerates COST.md. Log: `infra/state/watchdog.log`.
4. `infra/queue-runner.sh` marks jobs whose runner process died as failed and deletes their VM.

Check cron: `crontab -l | grep traders-net-gpu`. Reinstall: `infra/install-cron.sh`
(`--remove` to uninstall). Pause dispatch: `touch infra/state/PAUSED` (remove to resume).

## Zero-VM check (run at the end of the study, and any time)

    $G1/infra/zero-vm-check.sh --log     # exit 0 = no traders-net instance, disk or running job

It is read-only apart from `infra/state/zero-vm-check.json` and the optional GPU-LOG line.
Emergency: `$G1/infra/gpu-down.sh --all` deletes every traders-net VM (and nothing else).

## Files

`infra/lib.sh` settings and helpers · `startup.sh` VM boot (venv + torch, writes /var/lib/tn/READY)
· `gpu-up.sh` admission + create · `gpu-down.sh` delete + audit · `job-run.sh` one job end to end ·
`queue-runner.sh` dispatcher · `submit.sh` · `gpu-watchdog.sh` · `cost-audit.py` · `cost-report.py`
· `zero-vm-check.sh` · `install-cron.sh` · `caps.tsv` · `tests/test_infra.py` (fake gcloud, no
rentals: `python3 infra/tests/test_infra.py`). `infra/state/` holds the SSH key, locks, logs, audit
evidence; not for sharing.

Debug a live VM: `ssh -i $G1/infra/state/tn_ed25519 tn@<IP>` (IP from `job.log`), then `sudo -i`;
job copy in `/root/job`, outputs in `/root/out`, boot log `/var/log/tn-startup.log`.
