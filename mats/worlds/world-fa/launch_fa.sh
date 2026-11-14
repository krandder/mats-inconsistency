#!/usr/bin/env bash
# Usage: launch_fa.sh <run-id> [STEPS] [P] [SCRIPT]  — packs the World FA code + harness, runs SCRIPT
# (default run_pilot_fa.sh; run_binding_fa.sh reads HOURS from the environment) on one
# ON-DEMAND L4 VM (GPU_PROVISIONING=STANDARD), syncs runs back to /tmp/l4runs/<run-id>/runs/<run-id>/, VM auto-deletes.
set -euo pipefail
RID=$1; STEPS=${2:-20000}; P=${3:-8}; SCRIPT=${4:-run_pilot_fa.sh}; HOURS=${HOURS:-3.2}
STAGE=$(mktemp -d /tmp/fa_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/world-fa
cp fa_core.py world_fa.py fa_brute.py test_fa.py diag_fa.py train_fa.py fa_tests.py run_pilot_fa.sh run_binding_fa.sh run_decomp_fa.sh run_a7_fa.sh PREREG-FA.md \
   ../worlds/harness.py "$STAGE/"
cd "$STAGE" && sha256sum *.py *.sh PREREG-FA.md > code_sha256.txt
B=$(tar cz *.py *.sh PREREG-FA.md code_sha256.txt | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy && OUT=/root/runs/$RID/runs STEPS=$STEPS P=$P HOURS=$HOURS ONLY=${ONLY:-} PY=/root/venv/bin/python bash $SCRIPT"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-10800s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
