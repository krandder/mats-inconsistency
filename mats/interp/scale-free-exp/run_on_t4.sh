#!/usr/bin/env bash
# Fallback 2 (coordinator, 2026-10-29): on-demand T4. Same as work/run_on_l4.sh, but the temporary worktree's
# gpu-up.sh / lib.sh are patched (worktree copy only; shared infra untouched) to create n1-standard-8 + 1x T4
# with a $0.75/h budgeting ceiling (n1-standard-8 ~$0.38 + T4 ~$0.35 + disk, us list prices; unverified).
# Usage: run_on_t4.sh <run-id> <repo-commit> '<command>'   (GPU_SOURCE_DIR, GPU_ZONES as for run_on_l4.sh)
set -euo pipefail
RUN_ID=$1; COMMIT=$2; CMD=$3
REPO=/home/agent/workspaces/inconsistency-search
DIR=/tmp/l4runs/$RUN_ID
mkdir -p /tmp/l4runs
git -C "$REPO" worktree remove --force "$DIR" 2>/dev/null || true
git -C "$REPO" worktree prune
git -C "$REPO" worktree add -q --detach "$DIR" "$COMMIT"
sed -i 's/^MACHINE="g2-standard-8"; ACCEL="nvidia-l4";/MACHINE="n1-standard-8"; ACCEL="nvidia-tesla-t4";/' "$DIR/scripts/infra/gpu-up.sh"
sed -i 's/^    g2-standard-8)  echo "0.87" ;;   # 1x L4, 8 vCPU, 32 GB/    g2-standard-8)  echo "0.87" ;;   # 1x L4, 8 vCPU, 32 GB\n    n1-standard-8)  echo "0.75" ;;   # 1x T4 + 8 vCPU (sfx fallback; unverified ceiling)/' "$DIR/scripts/infra/lib.sh"
grep -q 'nvidia-tesla-t4' "$DIR/scripts/infra/gpu-up.sh" && grep -q 'n1-standard-8)' "$DIR/scripts/infra/lib.sh" || { echo "T4 patch failed"; exit 2; }
mkdir -p "$DIR/infra/state" "$DIR/runs"
SETUP='export DEBIAN_FRONTEND=noninteractive; apt-get update -q >/dev/null && apt-get install -y -q python3-venv python3-pip >/dev/null && python3 -m venv /root/venv && /root/venv/bin/pip install -q torch numpy scipy pytest && cd /root/soup && mkdir -p /root/runs/'"$RUN_ID"' && export PATH=/root/venv/bin:$PATH && '
(
  rc=0
  GPU_CONTROL_ROOT="$DIR" GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} GPU_BUDGET_GROUP=explore-stage1 \
    bash "$DIR/scripts/infra/gpu-run.sh" --up --run-id "$RUN_ID" -- bash -c "$SETUP $CMD" || rc=$?
  echo "GPU-RUN EXIT $rc"
  cat "$DIR/GPU-LOG.md" >> /home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md 2>/dev/null || true
) > "$DIR/gpu-run.out" 2>&1
tail -1 "$DIR/gpu-run.out"
/home/agent/workspaces/mats/work/l4_slim.sh "$RUN_ID" >/dev/null 2>&1 || true
