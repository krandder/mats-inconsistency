#!/usr/bin/env bash
# Usage: gpu/launch_gpu.sh <run-id>. Packs az.py, train_batched.py, gpu/run_gpu.sh and the two reference weight files,
# runs on one ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD) via work/run_on_l4.sh (VM auto-deletes), results come back
# to /tmp/l4runs/<run-id>/runs/<run-id>/out/.
set -euo pipefail
RID=$1
STAGE=$(mktemp -d /tmp/azgpu_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/algzoo-noisy
mkdir -p "$STAGE/ext/weights"
cp az.py train_batched.py gpu/run_gpu.sh "$STAGE/"
cp ext/weights/one_layer_16_hidden_94_acc_2nd_argmax.pth ext/weights/2nd_argmax_16_10_1073741824_2.pth "$STAGE/ext/weights/"
cd "$STAGE" && sha256sum *.py *.sh ext/weights/* > code_sha256.txt
B=$(tar cz *.py *.sh ext code_sha256.txt | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && NTRAIN=${NTRAIN:-1073741824} PY=/root/venv/bin/python bash run_gpu.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
