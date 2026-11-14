#!/bin/bash
# R4-E3b driver: per seed, warm-up then the three arms in parallel. Logs in r4-e3b/runs/*.log.
cd "$(dirname "$0")"
export TINKER_API_KEY="[removed]"
P=/home/agent/.venvs/tinker/bin/python
mkdir -p runs
for s in 101 102 103; do
  (
    $P lir_rl.py warmup --seed $s --out runs > runs/warm-s$s.log 2>&1 || exit 1
    for arm in fused tokmask lir; do
      $P lir_rl.py rl --arm $arm --seed $s --out runs > runs/$arm-s$s.log 2>&1 &
    done
    wait
  ) &
done
wait
echo ALL-DONE > runs/ALL-DONE
