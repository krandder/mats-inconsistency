#!/usr/bin/env bash
# Usage: launch_gpu.sh <run-id> '<command run in the staged dir on the VM>'
# Stages lib/, ext/ and gpu/ of work/sparse-codes into /tmp/sparsecodes_stage/<run-id>/, runs the command on one
# ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD) via work/run_on_l4.sh (VM deleted at the end, also on failure).
# Outputs written under /root/runs/<run-id>/ come back to /tmp/l4runs/<run-id>/runs/<run-id>/.
# Lane cap: $10 (GPU-LOG.md). Copied from work/arc-noisy/launch_gpu.sh (paths changed).
set -euo pipefail
RID=$1; CMD=$2
HERE=/home/agent/workspaces/mats/work/sparse-codes
STAGE=/tmp/sparsecodes_stage/$RID
mkdir -p "$STAGE"
cp -r "$HERE"/lib "$HERE"/ext "$HERE"/gpu "$STAGE"/
find "$STAGE" -name __pycache__ -prune -exec rm -r {} + 2>/dev/null || true
(cd "$STAGE" && find . -name '*.py' -o -name '*.sh' -o -name '*.json' | sort | xargs sha256sum > code_sha256.txt 2>/dev/null || true)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"${CMD:0:200}\"" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-14400s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; verify with gcloud list)" >> "$HERE/GPU-LOG.md"
