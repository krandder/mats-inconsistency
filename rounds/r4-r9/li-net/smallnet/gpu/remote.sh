#!/bin/bash
# Runs ON the GPU VM as root. /root/soup = frozen copy: harness/env (unmodified), smallnet/*.py, data/, jobs.txt.
# Usage: remote.sh JOBFILE [PAR]. One smallnet.py argument list per line; PAR lines run at a time on the one GPU.
# Writes /root/runs/smallnet/status/<host>.jsonl (one line per finished job: rc, seconds, args, log tail).
set -u
cd /root/soup
JOBS=$1; PAR=${2:-6}; MIN=${3:-0}
H=$(hostname)
mkdir -p /root/runs/smallnet/status /root/logs
ST=/root/runs/smallnet/status/$H.jsonl
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib LIR_HARNESS=/root/soup/harness SMALLNET_DATA=/root/soup/data PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
(( MIN > 0 )) && export SMALLNET_DEADLINE=$(( t0 + MIN * 60 - 120 ))
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "{\"host\": \"$H\", \"event\": \"setup\", \"sec\": $(( $(date +%s) - t0 )), \"torch\": \"$(python3 -c 'import torch; print(torch.__version__)')\"}" >> $ST
k=0
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  k=$((k+1))
  while (( $(jobs -rp | wc -l) >= PAR )); do sleep 2; done
  (
    s0=$(date +%s)
    python3 smallnet/smallnet.py $line --threads 1 > /root/logs/job$k.log 2>&1
    rc=$?
    tailj=$(tail -3 /root/logs/job$k.log | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-600:]))')
    echo "{\"host\": \"$H\", \"job\": $k, \"rc\": $rc, \"sec\": $(( $(date +%s) - s0 )), \"t\": $(( $(date +%s) - t0 )), \"args\": \"$line\", \"tail\": $tailj}" >> $ST
  ) &
  sleep 1
done < "$JOBS"
wait
echo "{\"host\": \"$H\", \"event\": \"all-done\", \"t\": $(( $(date +%s) - t0 ))}" >> $ST
