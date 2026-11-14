#!/usr/bin/env bash
# Usage: launch_w1b.sh <run-id> <seed> [STEPS]   (one on-demand L4 VM; VM auto-deletes)
set -euo pipefail
RID=$1; SEED=$2; STEPS=${3:-100000}
cd /home/agent/workspaces/mats/work/worlds
B=$(tar cz junta.py harness.py diag_junta.py test_junta.py test_harness.py run_w1b.sh PREREG-W1.md code_sha256_w1b.txt | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy && OUT=/root/runs/$RID/runs STEPS=$STEPS SEED=$SEED PY=/root/venv/bin/python bash run_w1b.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-19800s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
