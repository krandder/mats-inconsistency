#!/usr/bin/env bash
# Usage: gpu/launch_gpu.sh <run-id>. Adapted from a copy of work/algzoo-noisy/gpu/launch_gpu.sh (not edited there).
# Packs the trainers and gpu/run_gpu.sh, runs them on one ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD) via
# work/run_on_l4.sh (VM auto-deletes), results come back to /tmp/l4runs/<run-id>/runs/<run-id>/out/.
set -euo pipefail
RID=$1
STAGE=$(mktemp -d /tmp/ndgpu_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/narrow-deep
cp gpu/rnn_train_gpu.py train/mlp_train.py gpu/run_gpu.sh "$STAGE/"
cd "$STAGE" && sha256sum *.py *.sh > code_sha256.txt
B=$(tar cz *.py *.sh code_sha256.txt | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && PY=/root/venv/bin/python bash run_gpu.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-9000s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
