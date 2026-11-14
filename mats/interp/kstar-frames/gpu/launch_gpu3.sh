#!/usr/bin/env bash
# Usage: gpu/launch_gpu3.sh <run-id>: like launch_gpu.sh, but runs gpu/run_gpu3.sh over gpu/jobs_gpu3.txt.
set -euo pipefail
RID=$1
STAGE=$(mktemp -d /tmp/kstar_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/kstar-frames
mkdir -p "$STAGE/lib" "$STAGE/scratch"
cp lib/kf.py lib/cover.py lib/nd.py "$STAGE/lib/"
cp scratch/r1_layers.py scratch/r2_big.py scratch/r2b_analyze.py scratch/r3_pairs.py scratch/r4_mlp.py scratch/r5_states.py "$STAGE/scratch/"
cp gpu/run_gpu3.sh gpu/jobs_gpu3.txt "$STAGE/"
mkdir -p "$STAGE/out/runs"
for t in n128_s400_L32 n128_s401_L32 n128_s402_L32 n128_s400_L16; do   # k-tables only (keeps the payload small)
  python3 -c "import json,sys; d=json.load(open('runs/A_$t.json')); json.dump({'n': d['n'], 'seed': d['seed'], 'L': d['L'], 'ktable_only': True, 'k': d['k']}, open(sys.argv[1], 'w'))" "$STAGE/out/runs/A_$t.json"
done
cd "$STAGE" && sha256sum lib/*.py scratch/*.py run_gpu3.sh jobs_gpu3.txt > code_sha256.txt
B=$(tar cz lib scratch run_gpu3.sh jobs_gpu3.txt code_sha256.txt out | base64 -w0)
echo "stage $STAGE, payload $(echo -n $B | wc -c) bytes"
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && PY=/root/venv/bin/python bash run_gpu3.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-5400s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
