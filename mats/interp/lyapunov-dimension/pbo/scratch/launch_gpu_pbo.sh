#!/usr/bin/env bash
# Usage: scratch/launch_gpu_pbo.sh <run-id> '<command run in the staged dir on the VM>'
# Copy of work/nongauss-family/scratch/launch_gpu.sh for this addendum: stages lib/, scratch/, truth/ and the PREREG
# files into /tmp/lyapdim_stage/<run-id>-<time>/ and runs the command on one on-demand L4 through work/run_on_l4.sh
# (VM deleted at the end, also on failure). Outputs under /root/runs/<run-id>/ come back to /tmp/l4runs/<run-id>/.
set -euo pipefail
RID=$1; CMD=$2
HERE=/home/agent/workspaces/mats/work/lyapunov-dimension/pbo
STAGE=/tmp/lyapdim_stage/$RID-$(date -u +%H%M%S)
mkdir -p "$STAGE/lib" "$STAGE/scratch" "$STAGE/truth"
cp "$HERE"/lib/*.py "$HERE"/lib/rtable_ng.json "$STAGE/lib/"
cp "$HERE"/scratch/run3.py "$HERE"/scratch/gpu_job_pbo.sh "$STAGE/scratch/"
cp "$HERE"/truth/*.npz "$STAGE/truth/"
cp "$HERE"/PREREG-PBO.md "$HERE"/PREREG-PBO-predictions.txt "$STAGE/"
(cd "$STAGE" && sha256sum lib/*.py lib/rtable_ng.json scratch/* PREREG-PBO.md > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) stage=$STAGE cmd=\"${CMD:0:200}\"" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-7200s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; verify with gcloud list)" >> "$HERE/GPU-LOG.md"
