#!/bin/bash
# Side jobs moved off lane a (its GPU time was short after the 14:00 re-plan) onto VM b, which had GPU slack.
# Runs as its own systemd unit next to run_lane.py; writes logs/markers in the same format.
#   sudo systemd-run --unit=p0-side-$(date +%s) --working-directory=/root/job /bin/bash /root/job/vm/side_b.sh [JOB...]
cd /root/job || exit 1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 P0_MR_DIR=/root/job
PY=/opt/venv/bin/python
MT=$($PY -c "import sys; sys.path.insert(0, 'vm'); import plan; print(plan.MAZE_TEST)")
SH=$($PY -c "import sys; sys.path.insert(0, 'vm'); import plan; print(plan.SHIFT)")
CK=/root/out/ckpt; EV=/root/out/eval
run() {
  local id=$1; shift
  [[ -f /root/out/markers/$id.done ]] && return 0
  local t0 t1 rc s; t0=$(date +%s)
  echo "### $(date -u +%FT%TZ) side unit: $*" >>"/root/out/logs/$id.log"
  $PY "$@" >>"/root/out/logs/$id.log" 2>&1; rc=$?; t1=$(date +%s); s=done; [[ $rc == 0 ]] || s=failed
  echo "{\"rc\": $rc, \"sec\": $((t1 - t0)), \"start\": $t0, \"end\": $t1, \"try\": 1, \"where\": \"side unit on b\"}" >"/root/out/markers/$id.$s"
}
for j in "${@:-ev_Mref ev_Mref_shift}"; do
  for id in $j; do
    case $id in
      ev_Mref) run ev_Mref evalrun.py --ckpt $CK --model M-ref --sets "$MT" --out $EV/Mref --probe v9 ;;
      ev_Mref_shift) run ev_Mref_shift evalrun.py --ckpt $CK --model M-ref --sets "$SH" --out $EV/Mref_shift ;;
      ev_M1_s0_shift) run ev_M1_s0_shift evalrun.py --ckpt $CK --model M1:0:best --sets "$SH" --out $EV/M1_s0_shift ;;
      ev_Mref_stretch) run ev_Mref_stretch evalrun.py --ckpt $CK --model M-ref --sets m201 --out $EV/Mref_stretch ;;
      ev_M1_s0_stab) run ev_M1_s0_stab evalrun.py --ckpt $CK --model M1:0:best --sets "m33[0:20]@30000=m33stab20" --out $EV/M1_s0_stab ;;
      *) echo "unknown side job $id" >&2 ;;
    esac
  done
done
