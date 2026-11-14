#!/usr/bin/env bash
# Usage: launch.sh <run-id> [STEPS] [P]   (packs the W1 code, runs run_pilot.sh on one on-demand L4 VM; VM auto-deletes)
set -euo pipefail
RID=$1; STEPS=${2:-40000}; P=${3:-8}
cd /home/agent/workspaces/mats/work/worlds
B=$(tar cz junta.py harness.py diag_junta.py test_junta.py run_pilot.sh PREREG.md code_sha256.txt | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy && OUT=/root/runs/$RID/runs STEPS=$STEPS P=$P PY=/root/venv/bin/python bash run_pilot.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-18000s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
