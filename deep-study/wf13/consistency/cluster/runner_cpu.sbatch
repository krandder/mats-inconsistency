#!/bin/bash
#SBATCH --job-name=w13-consistency
#SBATCH --partition=compute
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/home/user/wave13/consistency/runner-%j.log
# CPU-ONLY job (no GPU requested): the L40 queue estimate for this account was 2 h 40 min at 16:50Z, so the work fell
# back to CPU. Same pattern as wf10's runner: runs queue/*.task in name order; done/<name>.done marks a finished task.
cd /cluster/home/user/wave13/consistency || exit 1
mkdir -p queue done out data logs
export CUDA_VISIBLE_DEVICES=""
nproc
idle=0
while true; do
  if [ -f queue/STOP ]; then echo "$(date -u +%H:%M:%S) STOP"; break; fi
  next=""
  for t in $(ls queue/*.task 2>/dev/null | sort); do
    b=$(basename "$t" .task)
    if [ ! -f "done/$b.done" ]; then next="$b"; break; fi
  done
  if [ -n "$next" ]; then
    idle=0
    echo "$(date -u +%H:%M:%S) start $next"
    bash "queue/$next.task" > "logs/task_$next.log" 2>&1
    echo "$(date -u +%H:%M:%S) end $next rc=$?"
    date -u > "done/$next.done"
  else
    sleep 5; idle=$((idle+5))
    if [ $idle -ge 900 ]; then echo "$(date -u +%H:%M:%S) idle-exit"; break; fi
  fi
done
