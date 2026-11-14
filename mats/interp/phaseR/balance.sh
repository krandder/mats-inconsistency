#!/usr/bin/env bash
# usage: balance.sh JOBFILE MAXPROCS
# Starts the next line of JOBFILE (popped under flock) whenever fewer than MAXPROCS phase-R job processes are running
# (counted over every phase-R script, whichever queue started them). Start/end lines go to JOBFILE.done.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
f="$1"; max="${2:-4}"
pat='^/home/agent/.venvs/aggext/bin/python (truth_fz|chain_fz|eps_lb|score_cb|gc_tail|surr|score|rare_run|truth_hi)\.py'
while true; do
  n=$(pgrep -fc "$pat")
  if [ "$n" -lt "$max" ]; then
    line=$(flock "$f.lock" bash -c 'l=$(head -n1 "$0"); if [ -n "$l" ]; then sed -i 1d "$0"; fi; printf "%s" "$l"' "$f")
    [ -z "$line" ] && break
    echo "$(date -u +%H:%M:%S) start $line" >> "$f.done"
    ( bash -c "$line"; echo "$(date -u +%H:%M:%S) end rc=$? $line" >> "$f.done" ) &
    sleep 4
  else
    sleep 10
  fi
done
wait
