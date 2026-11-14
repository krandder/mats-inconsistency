# GPU infra for the logical-induction soup

Rented GCP GPU VMs for Phase 2+ runs (GOAL.md, "GPU" and "Stop guarantee"). Kelvin authorized
up to $200 total on 2026-09-27 and said "Make sure to stop it later". Project `GCP_PROJECT`,
region `us-central1`, account `[removed]` (this box's gcloud).

## Commands

    infra/gpu-up.sh                 # one spot L4 VM (g2-standard-8), waits until READY (~5-10 min)
    infra/gpu-up.sh --a100          # one spot A100 40GB VM (a2-highgpu-1g) instead
    infra/gpu-run.sh -- make check  # rsync this dir to /root/soup on the VM, run as root under
                                    # timeout 19800s (5h30m) (GPU_RUN_TIMEOUT), rsync /root/runs back to runs/,
                                    # then delete the VM, also on failure or Ctrl+C
    infra/gpu-run.sh --up -- CMD    # same, creating the VM first (--up --a100 for A100)
    infra/gpu-run.sh --keep -- CMD  # leave the VM up afterwards (6 h limit and watchdog still apply)
    infra/gpu-down.sh               # legacy unnamed VM only; refuses if named runs exist
    infra/gpu-down.sh --only NAME   # exact VM; named runs never sync unverified remote data
    infra/gpu-down.sh --all         # explicitly delete every soup VM (named runs still never sync)
    infra/gpu-watchdog.sh --dry-run # what the watchdog would do now
    infra/install-watchdog-cron.sh  # (re)install the 15-min cron entry; --remove to uninstall

Interactive shell on the VM: `ssh -i infra/state/soup_ed25519 soup@<IP>` (IP from
`gcloud compute instances list --filter=labels.logical-soup:*`), then `sudo -i`.

## What a VM is

- Unique `soup-gpu-*` name, label `logical-soup=1`; named runs also carry `soup-run` and optional `soup-group` labels. Multiple named runs may coexist.
- Spot, `--max-run-duration=6h --instance-termination-action=DELETE`: GCE deletes it at 6 h or on
  preemption. Longer sessions need a new VM.
- Image family `common-cu129-ubuntu-2404-nvidia-580` (project `deeplearning-platform-release`,
  CUDA 12.9, NVIDIA driver 580), 100 GB pd-balanced boot disk, default network (external IP,
  `default-allow-ssh`), no service account (the VM cannot call GCP APIs).
- Startup script `infra/startup.sh`: clang, build-essential, bun 1.3.14 (same as General), Bend2
  `v2.0.29` from github.com/bendlang/bend at `/opt/bend2`, symlinked at the General path
  `/home/agent/workspaces/eurisko-study/research/bend2-scratch/bend-2.0.29` so
  `../selfmod/bend2-lab/run.sh`-style scripts work unchanged. Writes `/var/lib/soup/READY`
  (GPU, nvcc, clang, bun, bend commit), which gpu-up copies into GPU-LOG.md. Log:
  `/var/log/soup-startup.log`.
- SSH: a dedicated key `infra/state/soup_ed25519` (user `soup`) in instance metadata, with
  `block-project-ssh-keys=TRUE`, so the shared project ssh-keys are never touched.

## Stop guarantee (three layers)

1. GCE: 6 h max run duration, then DELETE.
2. `infra/gpu-watchdog.sh` from user cron every 15 min: lists VMs labelled `logical-soup` or named
   `soup-gpu-*`, posts one line per VM per hour to Fleet - Status, deletes any older than 8 h or
   named `soup-gpu-*` without the label, appends to GPU-LOG.md. If gcloud cannot list, it posts
   once per 6 h. Its log: `infra/state/watchdog.log`. Check it is installed: `crontab -l | grep logical-soup`.
3. `gpu-run.sh` calls `gpu-down.sh` on exit, whatever the exit status.

The watchdog deliberately does not touch other GPU VMs in the project (other teams use it).

## Cost math

Rates are ceilings, not verified prices: the Cloud Billing API is disabled in GCP_PROJECT, so
the scripts use public on-demand list prices for us-central1 (machine incl. GPU) plus the disk.

| VM | vCPU / RAM | GPU | ceiling $/h | worst case per VM (6 h) |
|---|---|---|---|---|
| g2-standard-8 | 8 / 32 GB | 1x L4 24 GB | 0.87 | 5.22 |
| a2-highgpu-1g | 12 / 85 GB | 1x A100 40 GB | 3.69 | 22.14 |

Boot disk 100 GB pd-balanced is about $10/month, $0.014/h, included in the ceilings. Spot usually
costs 60-70% less than on-demand, so real spend should be well under the log's estimate.

Budget: $200 = 38 full L4 sessions or 9 full A100 sessions at the ceiling. gpu-up reserves the
full 6 h when it creates and replaces that with rate x age at delete.
Check the running total: `bash -c 'source infra/lib.sh; spend_usd'`.

## Files

`lib.sh` (settings, rates, helpers), `startup.sh` (VM boot), `gpu-up.sh`, `gpu-down.sh`,
`gpu-run.sh`, `gpu-watchdog.sh`, `install-watchdog-cron.sh`, `state/` (SSH key, watchdog state and
log; not for git).

## Concurrent named rentals

Use `gpu-run.sh --up --run-id phase3-k16-f12-s1 -- CMD` for a named rental.
Names and labels isolate admission and cleanup; exits delete only that VM.
Unnamed launches still refuse when another soup VM exists. Admission holds a
local lock through reservation, and `GPU_BUDGET_GROUP=pool-sddpv
GPU_GROUP_BUDGET_USD=40` adds a group cap within the $200 project cap.
GCE's six-hour DELETE TTL and watchdog rules are unchanged.

`--driver scripts/gpu-phase3-drive.sh -- 1` runs a local driver under the same
bounded timeout and deletion trap. It gets GPU_VM_IP/NAME/ZONE and owns result
sync. Generic final sync is disabled for drivers so interrupted remote data
cannot overwrite a verified checkpoint. `scripts/gpu-phase3.sh 1` supervises
one seed, rotates rentals after 5h25m, and retries from the local snapshot.
Use `scripts/launch-gpu-phase3.sh` to launch all three seeds, or `--dry-run`
to print their plans and reservation checks without contacting GCE. It archives
the committed tree to `build/phase3-runtime-<sha>`, then execs the coordinator
from that directory before creating any VM. Coordinator, supervisors, drivers,
infra and remote code all use the frozen tree. `GPU_CONTROL_ROOT` pins shared
state, SSH keys, output and GPU-LOG.md to this live workspace. Do not edit a
frozen runtime. Replacements reuse it; optimization experiments cannot change it.
Every 100 epochs the remote process exits, the snapshot/CSV pair is copied
home, and its epoch count and exact ledger are checked before the next chunk.
Only one checkpoint per seed is retained. Failed chunks restart from the last
verified checkpoint. Per-seed locks prevent duplicate supervisors.

Each seed permits at most seven attempts, stops after two consecutive rentals
without checkpoint progress, and treats a chunk's 30-minute timeout as a
non-retryable failure distinct from session rotation. Cleanup knows the exact
VM name before creation; a per-name zone intent supports cleanup if listing
fails. Audits retry deletion once, use the recorded machine type's rate, and
clear per-attempt state. The coordinator records an empty instance/disk check
after all supervisors exit. The watchdog schedule and rules remain unchanged.

Launch the coordinator independently of the interactive harness session:

```sh
setsid nohup scripts/launch-gpu-phase3.sh > infra/state/phase3-logs/coordinator.log 2>&1 < /dev/null &
```

Admission logs PENDING-CREATE with the full reservation before invoking GCE;
CREATE updates the same VM entry without double counting. A targeted deletion
continues with sync disabled if reading the owner label fails.
