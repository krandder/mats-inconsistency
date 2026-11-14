#!/bin/bash
# run_vm.sh : runs ON the VM from /root/soup. Three seeds in parallel on the one L4 (r8/run_r8e3.py).
export PYTHONPATH=/root/pylib PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 R5E2_DATA=/root/soup/data
O=/root/runs/r8e3
mkdir -p $O
cd /root/soup
for S in 101 102 103; do
  (
    mkdir -p $O/s$S
    python3 r8/run_r8e3.py --seed $S --ckpt /root/soup/ckpt/s$S/primary --out $O/s$S ${EXTRA:-} > $O/s$S/run.log 2>&1
    echo "{\"exit\": $?, \"seed\": $S, \"utc\": \"$(date -u +%FT%TZ)\"}" > $O/s$S/exit.json
  ) &
done
wait
echo "{\"utc\": \"$(date -u +%FT%TZ)\"}" > $O/all_done.json
