#!/usr/bin/env bash
# Usage: launch_e21.sh <run-id>. Stages pfn2.py (tiny-pfn-interp/heads), world.py (tiny-pfn) and gpu_e21.sh, runs gpu_e21.sh on one
# ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD) via work/run_on_l4.sh (the VM is deleted at the end, also on failure); outputs come
# back to /tmp/l4runs/<run-id>/runs/<run-id>/. Every VM is logged in ../GPU-LOG.md.
set -euo pipefail
RID=$1
HERE=/home/agent/workspaces/mats/work/cm-interp-B
STAGE=/tmp/cmB_stage/$RID
mkdir -p "$STAGE"
cp /home/agent/workspaces/mats/work/tiny-pfn-interp/heads/pfn2.py /home/agent/workspaces/mats/work/tiny-pfn/world.py "$HERE/gpu/gpu_e21.sh" "$STAGE"/
(cd "$STAGE" && sha256sum *.py *.sh > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"bash gpu_e21.sh $RID\" (E21, amendment 28; cap \$10 for this lane)" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "bash gpu_e21.sh $RID" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; instance list checked separately)" >> "$HERE/GPU-LOG.md"
