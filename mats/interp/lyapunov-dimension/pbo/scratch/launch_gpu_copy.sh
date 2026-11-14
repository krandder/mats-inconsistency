#!/usr/bin/env bash
# Usage: scratch/launch_gpu.sh <run-id> '<command run in the staged dir on the VM>'
# Stages lib/ and scratch/ into a fresh /tmp/ngfam_stage/<run-id>-<timestamp>/ and runs the command on one ON-DEMAND
# L4 VM through work/run_on_l4.sh (VM deleted at the end, also on failure). Outputs written under /root/runs/<run-id>/
# come back to /tmp/l4runs/<run-id>/runs/<run-id>/.
set -euo pipefail
RID=$1; CMD=$2
HERE=/home/agent/workspaces/mats/work/nongauss-family
STAGE=/tmp/ngfam_stage/$RID-$(date -u +%H%M%S)
mkdir -p "$STAGE/lib" "$STAGE/scratch"
cp "$HERE"/lib/*.py "$HERE"/lib/rtable_ng.json "$STAGE/lib/"
cp "$HERE"/scratch/*.py "$HERE"/scratch/gpu_job*.sh "$STAGE/scratch/"
(cd "$STAGE" && sha256sum lib/*.py lib/rtable_ng.json scratch/* > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) stage=$STAGE cmd=\"${CMD:0:200}\"" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; verify with gcloud list)" >> "$HERE/GPU-LOG.md"
