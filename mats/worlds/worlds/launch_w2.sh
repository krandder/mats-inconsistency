#!/usr/bin/env bash
# Usage: launch_w2.sh <run-id>   (env passes through: STEPS, P, SEEDS, ARMS). Packs the W2 code, runs run_w2.sh on
# one on-demand L4 VM; the VM deletes itself when the command ends.
set -euo pipefail
RID=$1
cd /home/agent/workspaces/mats/work/worlds
B=$(tar cz harness.py world2_market.py test_world2_market.py w2_summarize.py run_w2.sh PREREG-W2.md PREREG-W2.sha256 | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy && OUT=/root/runs/$RID/runs STEPS=${STEPS:-30000} P=${P:-6} SEEDS='${SEEDS:-0 1}' ARMS='${ARMS:-2:128 4:128 4:64}' PY=/root/venv/bin/python bash run_w2.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-18000s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
