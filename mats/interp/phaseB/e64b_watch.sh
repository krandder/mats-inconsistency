#!/usr/bin/env bash
# E64b (amendment 72, addendum 1): wait until the GPU retry script (PID) has exited, find the run id whose VM ran, copy its
# e64_split* runs into runs/e64b/, then every 5 minutes queue their evaluations (e64b_queue.py) until both code-share outputs exist.
# usage: e64b_watch.sh RETRY_PID
cd /home/agent/workspaces/mats/work/cm-interp-B
PID=$1; LOG=results/e64b_collect.log
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
RID=$(grep -o "attempt [0-9]* (cmb-e64b-[0-9]*): VM was created" GPU-LOG.md | tail -1 | grep -o "cmb-e64b-[0-9]*")
echo "$(date -u +%TZ) retry script exited; rid=$RID" >> $LOG
[ -n "$RID" ] || exit 1
SRC=/tmp/l4runs/$RID/runs/$RID
mkdir -p runs/e64b
for d in "$SRC"/e64_split*; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); [ -e runs/e64b/$n ] && continue
  cp -r "$d" runs/e64b/$n; [ -f "$SRC/$n.log" ] && cp "$SRC/$n.log" runs/e64b/$n.log
done
cp "$SRC/jobs.txt" runs/jobs_$RID.txt 2>/dev/null
echo "$(date -u +%TZ) collected $(ls -d runs/e64b/*/ | wc -l) runs, $(ls runs/e64b/*/meta.json 2>/dev/null | wc -l) finished" >> $LOG
while true; do
  nice -n 19 /home/agent/.venvs/aggext/bin/python e64b_queue.py >> $LOG 2>&1
  n=$(ls results/e64b_codes/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 2 ]; then echo "$(date -u +%TZ) both code-share outputs exist" >> $LOG; touch /tmp/s7q/e64b_codes.done; break; fi
  sleep 300
done
