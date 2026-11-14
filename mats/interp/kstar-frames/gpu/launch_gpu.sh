#!/usr/bin/env bash
# Usage: gpu/launch_gpu.sh <run-id>. Adapted from work/narrow-deep/gpu/launch_gpu.sh (not edited there).
# Packs lib/ and the run scripts, runs gpu/run_gpu.sh on one ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD) via
# work/run_on_l4.sh (VM auto-deletes); results come back to /tmp/l4runs/<run-id>/runs/<run-id>/out/.
set -euo pipefail
RID=$1
STAGE=$(mktemp -d /tmp/kstar_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/kstar-frames
mkdir -p "$STAGE/lib" "$STAGE/scratch"
cp lib/kf.py lib/cover.py lib/nd.py "$STAGE/lib/"
cp scratch/r1_layers.py scratch/r2_big.py scratch/r2b_analyze.py scratch/r3_pairs.py scratch/r4_mlp.py scratch/r5_states.py "$STAGE/scratch/"
cp gpu/run_gpu.sh "$STAGE/"
cd "$STAGE" && sha256sum lib/*.py scratch/*.py run_gpu.sh > code_sha256.txt
B=$(tar cz lib scratch run_gpu.sh code_sha256.txt | base64 -w0)
echo "stage $STAGE, payload $(echo -n $B | wc -c) bytes"
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && PY=/root/venv/bin/python bash run_gpu.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-9000s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
