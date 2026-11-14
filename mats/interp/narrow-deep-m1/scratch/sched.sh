#!/bin/bash
# Launch queued jobs (one shell command per line of scratch/sched_jobs.txt) whenever fewer than 4 heavy python
# processes of this lane are running. Each job runs in the background; a line is removed when launched.
cd /home/agent/workspaces/mats/work/narrow-deep-m1
JOBS=scratch/sched_jobs.txt
while true; do
  [ -s $JOBS ] || break
  n=$(pgrep -f '/home/agent/.venvs/arc-noisy/bin/python scratch/' | wc -l)
  if [ "$n" -lt 4 ]; then sleep 45; n=$(pgrep -f '/home/agent/.venvs/arc-noisy/bin/python scratch/' | wc -l); fi
  if [ "$n" -lt 4 ]; then
    cmd=$(head -n 1 $JOBS)
    tail -n +2 $JOBS > $JOBS.tmp && mv $JOBS.tmp $JOBS
    echo "$(date -u +%H:%M:%SZ) launch: $cmd" >> logs/sched.log
    nohup bash -c "$cmd" > /dev/null 2>&1 &
    sleep 60
  else
    sleep 30
  fi
done
echo "$(date -u +%H:%M:%SZ) sched done" >> logs/sched.log
