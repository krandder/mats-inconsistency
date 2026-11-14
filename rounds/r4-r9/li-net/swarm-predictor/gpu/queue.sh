#!/bin/bash
# Run a queue of swarm rentals in order, at most 2 at a time. Usage: queue.sh QUEUEFILE
# QUEUEFILE lines: RUN_ID REMOTE_MINUTES JOBFILE   (# comments allowed). A line is launched only after the
# pre-check passes: spend_usd(swarm-predictor) + reserve <= $38 and fewer than 2 swarm VMs live. A budget refusal
# stops the queue for good; a "2 VMs live" refusal waits and retries. Each rental ends with its VM deleted by
# gpu-run.sh. Log: /tmp/swarm-predictor/logs/queue.log
set -u
Q=$1
S=/home/agent/workspaces/eurisko-study/li-net/swarm-predictor
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/swarm-predictor
LOG=$T/logs/queue.log
mkdir -p $T/logs
source $INF/lib.sh
echo "$(date -u +%FT%TZ) queue start $Q (pid $$)" >> $LOG
while read -r run min jobs; do
  [[ -z ${run:-} || $run == \#* ]] && continue
  while (( $(jobs -rp | wc -l) >= 2 )); do sleep 30; done
  while true; do
    TO=$(( min + 15 ))
    reserve=$(awk -v m=$TO 'BEGIN{printf "%.2f", 0.87*(m/60+0.6)}')
    spent=$(spend_usd swarm-predictor)
    if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>38)}'; then
      echo "$(date -u +%FT%TZ) STOP: $run would exceed \$38 (spent $spent + reserve $reserve); queue ends" >> $LOG
      wait; echo "$(date -u +%FT%TZ) queue end (budget)" >> $LOG; exit 0
    fi
    live=$("${GC[@]}" compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-swarm' --format='value(name)' 2>/dev/null | grep -c .)
    (( live < 2 )) && break
    sleep 60
  done
  echo "$(date -u +%FT%TZ) launching $run ($min min, $jobs), spent $spent" >> $LOG
  $S/gpu/launch.sh "$run" "$min" "$S/gpu/$jobs" >> $LOG 2>&1 &
  sleep 120
done < "$Q"
wait
echo "$(date -u +%FT%TZ) queue end (all launched rentals finished)" >> $LOG
