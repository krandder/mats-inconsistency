#!/usr/bin/env bash
# Usage: launch_t2.sh <run-id> <jobfile> <note>. Stages pfn2.py, world.py (pfn2 imports it), t2_world.py, train_t2.py,
# gpu_jobs_t2.sh and the job file; runs gpu_jobs_t2.sh on one ON-DEMAND L4 VM via work/run_on_l4.sh (the VM is deleted at the
# end, also on failure). Outputs come back to /tmp/l4runs/<run-id>/runs/<run-id>/. Every VM is logged in ../GPU-LOG.md.
set -euo pipefail
RID=$1; JOBFILE=$2; NOTE=$3
HERE=/home/agent/workspaces/mats/work/cm-interp-B
STAGE=/tmp/cmB_stage/$RID
mkdir -p "$STAGE"
cp /home/agent/workspaces/mats/work/tiny-pfn-interp/heads/pfn2.py /home/agent/workspaces/mats/work/tiny-pfn/world.py \
   "$HERE/t2_world.py" "$HERE/train_t2.py" "$HERE/gpu/gpu_jobs_t2.sh" "$STAGE"/
cp "$JOBFILE" "$STAGE/jobs.txt"
(cd "$STAGE" && sha256sum *.py *.sh jobs.txt > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"bash gpu_jobs_t2.sh $RID jobs.txt\" ($NOTE)" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-18600s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "bash gpu_jobs_t2.sh $RID jobs.txt" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; instance list checked separately)" >> "$HERE/GPU-LOG.md"
