#!/usr/bin/env bash
# Usage: launch_e30.sh <run-id>. Stages pfn2.py (tiny-pfn-interp/heads), world.py (tiny-pfn), train_fmt.py and gpu_e30.sh, runs
# gpu_e30.sh on one ON-DEMAND L4 VM via work/run_on_l4.sh (the VM is deleted at the end, also on failure); outputs come back to
# /tmp/l4runs/<run-id>/runs/<run-id>/. Every VM is logged in ../GPU-LOG.md. Cap for E30: $3 (coordinator, 2026-10-29).
set -euo pipefail
RID=$1
HERE=/home/agent/workspaces/mats/work/cm-interp-B
STAGE=/tmp/cmB_stage/$RID
mkdir -p "$STAGE"
cp /home/agent/workspaces/mats/work/tiny-pfn-interp/heads/pfn2.py /home/agent/workspaces/mats/work/tiny-pfn/world.py \
   "$HERE/train_fmt.py" "$HERE/e30_fmt.py" "$HERE/gpu/gpu_e30.sh" "$STAGE"/
(cd "$STAGE" && sha256sum *.py *.sh > code_sha256.txt)
echo "$(date -u +%FT%TZ) LAUNCH rid=$RID provisioning=STANDARD machine=g2-standard-8(L4) cmd=\"bash gpu_e30.sh $RID\" (E30, amendment 38; cap \$3 for E30)" >> "$HERE/GPU-LOG.md"
rc=0
GPU_SOURCE_DIR="$STAGE" GPU_PROVISIONING=STANDARD GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-11400s} \
  /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "bash gpu_e30.sh $RID" || rc=$?
grep -h "$RID" /tmp/l4runs/$RID/GPU-LOG.md 2>/dev/null | sed 's/^/    /' >> "$HERE/GPU-LOG.md" || true
echo "$(date -u +%FT%TZ) END rid=$RID rc=$rc (VM deleted by gpu-run.sh cleanup; instance list checked separately)" >> "$HERE/GPU-LOG.md"
