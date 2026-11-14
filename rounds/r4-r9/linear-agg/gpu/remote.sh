#!/bin/bash
# Runs ON the GPU VM as root in /root/soup (frozen copy of the linear-agg source). Usage: remote.sh JOBFILE PAR
# One shell command per job line (run from /root/soup); PAR jobs at a time. Status lines: /root/runs/la/status/<host>.jsonl
set -u
cd /root/soup
JOBS=$1; PAR=${2:-2}
H=$(hostname)
mkdir -p /root/runs/la/status /root/logs
ST=/root/runs/la/status/$H.jsonl
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# 2026-10-16 13:20Z: torch 2.14's Triton launcher needs Python.h (python3-dev is not in the image)
[ -f /usr/include/python3.12/Python.h ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-dev > /root/apt.log 2>&1
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "{\"host\": \"$H\", \"event\": \"setup\", \"sec\": $(( $(date +%s) - t0 )), \"torch\": \"$(python3 -c 'import torch; print(torch.__version__)')\"}" >> $ST
k=0
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  k=$((k+1))
  while (( $(jobs -rp | wc -l) >= PAR )); do sleep 2; done
  (
    s0=$(date +%s)
    bash -c "$line" > /root/logs/job$k.log 2>&1
    rc=$?
    tailj=$(tail -3 /root/logs/job$k.log | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-600:]))')
    cmdj=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$line")
    echo "{\"host\": \"$H\", \"job\": $k, \"rc\": $rc, \"sec\": $(( $(date +%s) - s0 )), \"t\": $(( $(date +%s) - t0 )), \"cmd\": $cmdj, \"tail\": $tailj}" >> $ST
  ) &
  sleep 1
done < "$JOBS"
wait
echo "{\"host\": \"$H\", \"event\": \"all-done\", \"t\": $(( $(date +%s) - t0 ))}" >> $ST
