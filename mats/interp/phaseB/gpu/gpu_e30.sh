#!/usr/bin/env bash
# E30 job on one L4 VM (PREREG-B amendment 38): train the tiny-pfn recipe (2 layers, width 32, 2 heads, mixture head K = 4)
# with train_fmt.py in four input-format arms (ctrl, mag, sgnmag, fix), seeds 0-11, interleaved by seed so that every arm has
# the same seeds done if the deadline cuts the run. Up to 8 trainings at once. Own deadline: no new training after 8,400 s,
# everything stopped at 9,900 s, so the outputs are synced before the runner's timeout. Outputs: /root/runs/$RID/<arm>_s<seed>/
set -x
RID=$1; PY=/root/venv/bin/python; OUT=/root/runs/$RID
nvidia-smi -L
mkdir -p $OUT
export OMP_NUM_THREADS=1
T0=$(date +%s)
jobs_running() { jobs -rp | wc -l; }
for S in 0 1 2 3 4 5 6 7 8 9 10 11; do
  for ARM in ctrl mag sgnmag fix; do
    while [ "$(jobs_running)" -ge 8 ]; do sleep 20; done
    if [ $(( $(date +%s) - T0 )) -ge 8400 ]; then echo "E30 deadline: not starting ${ARM}_s${S}"; continue; fi
    $PY train_fmt.py $ARM train --width 32 --heads 2 --depth 2 --mixk 4 --seed $S --out $OUT/${ARM}_s${S} > $OUT/${ARM}_s${S}.log 2>&1 &
    sleep 3
  done
done
while [ "$(jobs_running)" -gt 0 ]; do
  if [ $(( $(date +%s) - T0 )) -ge 9900 ]; then echo "E30 hard stop: killing $(jobs_running) trainings"; kill $(jobs -rp); break; fi
  sleep 30
done
wait
ls $OUT
echo E30-DONE
