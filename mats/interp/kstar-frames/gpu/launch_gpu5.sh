#!/usr/bin/env bash
# Usage: gpu/launch_gpu5.sh <run-id>: like launch_gpu3.sh, over gpu/jobs_gpu5.txt (float64 covering for n = 64, 128).
set -euo pipefail
RID=$1
STAGE=$(mktemp -d /tmp/kstar_stage.XXXXXX)
cd /home/agent/workspaces/mats/work/kstar-frames
mkdir -p "$STAGE/lib" "$STAGE/scratch"
cp lib/kf.py lib/cover.py lib/nd.py "$STAGE/lib/"
cp scratch/r2_big.py scratch/r2c_cover.py "$STAGE/scratch/"
cp gpu/run_gpu3.sh "$STAGE/" && cp gpu/jobs_gpu5.txt "$STAGE/jobs_gpu3.txt"
mkdir -p "$STAGE/out/runs"
for t in n128_s400_L32 n128_s401_L32 n128_s402_L32 n128_s400_L16 n128_s401_L16 n128_s402_L16 n64_s400_L32 n64_s401_L32 n64_s402_L32 n64_s400_L16 n64_s401_L16 n64_s402_L16; do   # k-tables only
  python3 -c "import json,sys; d=json.load(open('runs/A_$t.json')); json.dump({'n': d['n'], 'seed': d['seed'], 'L': d['L'], 'ktable_only': True, 'k': d['k']}, open(sys.argv[1], 'w'))" "$STAGE/out/runs/A_$t.json"
done
cd "$STAGE" && sha256sum lib/*.py scratch/*.py run_gpu3.sh jobs_gpu3.txt > code_sha256.txt
B=$(tar cz lib scratch run_gpu3.sh jobs_gpu3.txt code_sha256.txt out | base64 -w0)
echo "stage $STAGE, payload $(echo -n $B | wc -c) bytes"
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && PY=/root/venv/bin/python bash run_gpu3.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-5400s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
