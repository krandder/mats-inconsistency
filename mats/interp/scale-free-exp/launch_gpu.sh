#!/usr/bin/env bash
# Usage: launch_gpu.sh <run-id> [PAR]. Stages the code, data and job lists, runs gpu_job.sh on one ON-DEMAND L4 VM via
# work/run_on_l4.sh (the VM is deleted at the end, also on failure). Outputs come back to
# /tmp/l4runs/<run-id>/runs/<run-id>/. Every VM is logged in GPU-LOG.md (lane cap $10).
set -euo pipefail
RID=$1; PAR=${2:-10}
HERE=/home/agent/workspaces/mats/work/scale-free-exp
STAGE=/tmp/sfx_stage/$RID
mkdir -p "$STAGE"
cp "$HERE"/sfworld.py "$HERE"/sfmodels.py "$HERE"/sftrain.py "$HERE"/sfgpu.py "$HERE"/e4fa.py "$HERE"/fa_core_copy.py "$HERE"/gpu_job.sh "$STAGE"/
cp "$HERE"/${JOBS1:-jobs1.txt} "$STAGE"/jobs1.txt; cp "$HERE"/${JOBS2:-jobs2.txt} "$STAGE"/jobs2.txt
cp -r "$HERE"/data "$STAGE"/
(cd "$STAGE" && sha256sum *.py *.sh *.txt > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"bash gpu_job.sh $RID $PAR\" (scale-free exp; lane cap \$10)" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-18000s} \
  ${RUNNER:-/home/agent/workspaces/mats/work/run_on_l4.sh} "$RID" 0a36abb "bash gpu_job.sh $RID $PAR" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; instance list checked separately)" >> "$HERE/GPU-LOG.md"
