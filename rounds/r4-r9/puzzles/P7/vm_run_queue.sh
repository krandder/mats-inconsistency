#!/bin/bash
# Runs on the VM: every line of queue.txt is "script arm depth seed [--tag T other args...]"; P parallel
# single-thread jobs. Main grid -> /root/out/<maze|prefix>/, tagged runs -> /root/out/ablations/.
# Markers JOB_RUNNING / JOB_DONE in /root/out.
cd /root/job || exit 1
P=${1:-30}
touch /root/out/JOB_RUNNING
mkdir -p /root/out/maze /root/out/prefix /root/out/ablations
run_one() {
  local script=$1 arm=$2 dep=$3 seed=$4; shift 4
  local tag="" dir="/root/out/$script" prev=""
  for a in "$@"; do [[ $prev == --tag ]] && tag=$a; prev=$a; done
  [[ -n $tag ]] && dir=/root/out/ablations
  local args name
  if [[ $script == maze ]]; then
    args=(maze.py --arm "$arm" --D "$dep" --seed "$seed" --steps 2000 --out "$dir")
    name="maze_${arm}${tag}_D${dep}_s${seed}"
  else
    args=(prefix.py --arm "$arm" --d "$dep" --seed "$seed" --out "$dir")
    name="prefix_${arm}${tag}_d${dep}_s${seed}"
  fi
  OMP_NUM_THREADS=1 /opt/venv/bin/python "${args[@]}" "$@" >"$dir/$name.log" 2>&1
}
export -f run_one
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ < /root/job/queue.txt
rm -f /root/out/JOB_RUNNING
date -u +%FT%TZ > /root/out/JOB_DONE
