#!/usr/bin/env bash
# Run an exploratory job on one spot L4 VM (8 vCPU + L4 GPU, ~$0.87/h list ceiling), then sync /root/runs back
# and delete the VM. For stage-1 lanes whose training would overload General.
#
# Usage: run_on_l4.sh <run-id> <repo-commit> '<command run from the repo root on the VM>'
#   The repo (inconsistency-search) is checked out at <repo-commit> into /tmp/l4runs/<run-id>/, synced to the VM
#   as /root/soup, and the command runs with /root/venv/bin/python (torch + numpy + pytest installed).
#   Write outputs under /root/runs/<run-id>/; they come back to /tmp/l4runs/<run-id>/runs/<run-id>/.
#   Log: /tmp/l4runs/<run-id>/gpu-run.out (last line "GPU-RUN EXIT <rc>"). GPU-LOG lines are appended to the
#   central eurisko GPU-LOG when the run ends.
set -euo pipefail
RUN_ID=$1; COMMIT=$2; CMD=$3
REPO=/home/agent/workspaces/inconsistency-search
DIR=/tmp/l4runs/$RUN_ID
mkdir -p /tmp/l4runs
git -C "$REPO" worktree remove --force "$DIR" 2>/dev/null || true
git -C "$REPO" worktree prune
git -C "$REPO" worktree add -q --detach "$DIR" "$COMMIT"
mkdir -p "$DIR/infra/state" "$DIR/runs"
SETUP='export DEBIAN_FRONTEND=noninteractive; apt-get update -q >/dev/null && apt-get install -y -q python3-venv python3-pip >/dev/null && python3 -m venv /root/venv && /root/venv/bin/pip install -q torch numpy pytest && cd /root/soup && mkdir -p /root/runs/'"$RUN_ID"' && export PATH=/root/venv/bin:$PATH && '
(
  rc=0
  GPU_CONTROL_ROOT="$DIR" GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} GPU_BUDGET_GROUP=explore-stage1 \
    bash "$DIR/scripts/infra/gpu-run.sh" --up --run-id "$RUN_ID" -- bash -c "$SETUP $CMD" || rc=$?
  echo "GPU-RUN EXIT $rc"
  cat "$DIR/GPU-LOG.md" >> /home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md 2>/dev/null || true
) > "$DIR/gpu-run.out" 2>&1
tail -1 "$DIR/gpu-run.out"
# keep runs/, gpu-run.out and GPU-LOG.md; drop the repo checkout (disk: ~400M per run)
/home/agent/workspaces/mats/work/l4_slim.sh "$RUN_ID" >/dev/null 2>&1 || true
