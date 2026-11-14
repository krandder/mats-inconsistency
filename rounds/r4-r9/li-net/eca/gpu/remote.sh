#!/bin/bash
# Runs ON the GPU VM as root. /root/soup = frozen copy: eca/*.py, smallnet/netdef.py, data/, jobs.txt.
# Usage: remote.sh JOBFILE [PAR] [MINUTES]. One ecarun.py argument list per line; PAR lines run at a time on the GPU.
# Writes /root/runs/eca/status/<host>.jsonl (one line per finished job: rc, seconds, args, log tail).
set -u
cd /root/soup
JOBS=$1; PAR=${2:-2}; MIN=${3:-0}
H=$(hostname)
mkdir -p /root/runs/eca/status /root/logs
ST=/root/runs/eca/status/$H.jsonl
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib ECA_DATA=/root/soup/data SMALLNET_DIR=/root/soup/smallnet PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "{\"host\": \"$H\", \"event\": \"setup\", \"sec\": $(( $(date +%s) - t0 )), \"torch\": \"$(python3 -c 'import torch; print(torch.__version__)')\"}" >> $ST
k=0
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  k=$((k+1))
  while (( $(jobs -rp | wc -l) >= PAR )); do sleep 2; done
  (
    s0=$(date +%s)
    python3 eca/ecarun.py $line --threads 2 > /root/logs/job$k.log 2>&1
    rc=$?
    tailj=$(tail -4 /root/logs/job$k.log | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-800:]))')
    echo "{\"host\": \"$H\", \"job\": $k, \"rc\": $rc, \"sec\": $(( $(date +%s) - s0 )), \"t\": $(( $(date +%s) - t0 )), \"args\": \"$line\", \"tail\": $tailj}" >> $ST
  ) &
  sleep 1
done < "$JOBS"
# progress mirror: every 60 s copy the last log line of every job into a JSON file (synced home by the driver)
(
  while true; do
    python3 - <<'PY' > /root/runs/eca/status/progress-$H.json.tmp 2>/dev/null && mv /root/runs/eca/status/progress-$H.json.tmp /root/runs/eca/status/progress-$H.json
import glob, json, os
out = {}
for f in sorted(glob.glob('/root/logs/job*.log')):
    try:
        lines = [l.rstrip() for l in open(f, errors='replace') if l.strip()]
    except Exception:
        lines = []
    out[os.path.basename(f)] = lines[-3:]
print(json.dumps(out))
PY
    sleep 60
  done
) &
mon=$!
while (( $(jobs -rp | wc -l) > 1 )); do sleep 5; done
kill $mon 2>/dev/null
echo "{\"host\": \"$H\", \"event\": \"all-done\", \"t\": $(( $(date +%s) - t0 ))}" >> $ST
