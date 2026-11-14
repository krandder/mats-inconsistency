#!/usr/bin/env bash
# Usage: ARMS='Q4:4:0:0 ...' HOURS=4.5 launch_w2b.sh <run-id>
# Like run_on_l4.sh, but ships the W2b code and the pilot evalsets through the synced worktree (too large for the
# command line). One on-demand L4 VM; gpu-run.sh deletes it when the command ends.
set -euo pipefail
RID=$1
REPO=/home/agent/workspaces/inconsistency-search; DIR=/tmp/l4runs/$RID; SRC=/home/agent/workspaces/mats/work/worlds
mkdir -p /tmp/l4runs
git -C "$REPO" worktree remove --force "$DIR" 2>/dev/null || true
git -C "$REPO" worktree prune
git -C "$REPO" worktree add -q --detach "$DIR" 0a36abb
mkdir -p "$DIR/infra/state" "$DIR/runs" "$DIR/w2b/es"
cp "$SRC"/{harness.py,world2_market.py,w2fast.py,w2_train.py,w2_summarize.py,w2b_curves.py,test_world2_market.py,test_w2fast.py,run_w2b.sh,PREREG-W2.md} "$DIR/w2b/"
for es in "$SRC"/runs_w2_1028a/es_*/; do cp -r "$es" "$DIR/w2b/es/"; done
SETUP='export DEBIAN_FRONTEND=noninteractive; apt-get update -q >/dev/null && apt-get install -y -q python3-venv python3-pip >/dev/null && python3 -m venv /root/venv && /root/venv/bin/pip install -q torch numpy pytest scipy && mkdir -p /root/runs/'"$RID"' && '
CMD="cd /root/soup/w2b && nvidia-smi -L && OUT=/root/runs/$RID/out PY=/root/venv/bin/python HOURS=${HOURS:-4.5} STEPS=${STEPS:-150000} ARMS='${ARMS}' bash run_w2b.sh"
(
  rc=0
  GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_CONTROL_ROOT="$DIR" GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-21600s} GPU_BUDGET_GROUP=explore-stage1 \
    bash "$DIR/scripts/infra/gpu-run.sh" --up --run-id "$RID" -- bash -c "$SETUP $CMD" || rc=$?
  echo "GPU-RUN EXIT $rc"
  cat "$DIR/GPU-LOG.md" >> /home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md 2>/dev/null || true
) > "$DIR/gpu-run.out" 2>&1
tail -1 "$DIR/gpu-run.out"
