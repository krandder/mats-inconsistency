#!/bin/bash
# run_vm.sh SEED [smoke] : runs ON the VM from /root/soup (the synced source dir). Both architectures in parallel on the
# one L4: train6.py (all training stages) -> eval6.py (all parts) -> audit6_vm.py (independent D0 recomputation).
# smoke: 300 training steps and smoke-sized stages/parts, for timing and GPU-only bugs (outputs are discarded).
SEED=$1
MODE=${2:-full}
export PYTHONPATH=/root/pylib PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 R5E2_DATA=/root/soup/data
O=/root/runs/r5e2/s$SEED
mkdir -p $O
cd /root/soup
TA=""; EA=""
if [[ $MODE == smoke ]]; then
  TA="--steps 300 --n-agg-roots 128 --agg-steps 100 --distill-roots 256 --distill-steps 200 --n-trust-roots 512 --trust-steps 200"
  EA="--smoke"
  [[ -f data/oracle_q_test.npz && -f data/oracle_q_cal.npz && -f data/d1_pos.npz ]] || EA="--smoke --parts d0,d0cal,d3b"
fi
# full mode: eval waits for data/READY_ALL (pushed by drive.sh once the late oracle files are staged and the CPU smoke
# of the remaining eval parts has passed); training needs only train/cal npz.
wait_ready() {
  [[ $MODE == smoke ]] && return 0
  for i in $(seq 1 540); do [[ -f /root/soup/data/READY_ALL ]] && return 0; sleep 10; done
  echo "READY_ALL never arrived" >&2; return 1
}
for ARCH in primary fallback; do
  (
    mkdir -p $O/$ARCH
    python3 train6.py --seed $SEED --arch $ARCH --out $O/$ARCH $TA > $O/$ARCH/train.log 2>&1 \
      && wait_ready >> $O/$ARCH/train.log 2>&1 \
      && python3 eval6.py --seed $SEED --arch $ARCH --out $O/$ARCH $EA > $O/$ARCH/eval.log 2>&1 \
      && python3 audit6_vm.py --seed $SEED --arch $ARCH --out $O/$ARCH > $O/$ARCH/audit.log 2>&1
    echo "{\"exit\": $?, \"arch\": \"$ARCH\", \"mode\": \"$MODE\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $O/$ARCH/done.json
  ) &
done
wait
echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"mode\": \"$MODE\"}" > $O/all_done.json
