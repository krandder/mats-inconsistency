#!/usr/bin/env bash
# tractable-gates PFN arm (PREREG grid T): adapted from cm-interp-B/gpu/launch_x.sh. Stages train_act.py, train_sw.py, pfn2.py,
# world.py, e30_fmt.py, testset.npz, gpu_jobs.sh and jobs.txt; runs gpu_jobs.sh on ONE on-demand L4 via work/run_on_l4.sh (the VM
# is deleted at the end, also on failure). Outputs come back to /tmp/l4runs/<run-id>/runs/<run-id>/. Logged in ../GPU-LOG.md.
# usage: launch.sh <run-id>
set -euo pipefail
RID=$1
HERE=/home/agent/workspaces/mats/work/tractable-gates
STAGE=/tmp/tg_stage/$RID
mkdir -p "$STAGE"
cp "$HERE"/gpu/{train_act.py,train_sw.py,pfn2.py,world.py,e30_fmt.py,testset.npz,gpu_jobs.sh,jobs.txt} "$STAGE"/
(cd "$STAGE" && sha256sum *.py *.sh jobs.txt > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"bash gpu_jobs.sh $RID jobs.txt train_act.py 7200 9600\" (PFN activation arm, 18 trainings, cap \$3)" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "bash gpu_jobs.sh $RID jobs.txt train_act.py 7200 9600" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; instance list checked separately)" >> "$HERE/GPU-LOG.md"
