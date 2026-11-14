#!/bin/bash
#SBATCH --job-name=w13-consistency
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/cluster/home/user/wave13/consistency/runner-%j.log
# One Slurm job, many tasks (pattern of wf10/aggregator/cluster/runner.sbatch): runs queue/*.task in name order.
# A task is finished when done/<name>.done exists. Stops on queue/STOP or after 15 idle minutes.
cd /cluster/home/user/wave13/consistency || exit 1
mkdir -p queue done out data logs
nvidia-smi --query-gpu=name,memory.total --format=csv
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
