#!/usr/bin/env bash
# After a GPU episode (sixth agent; generalises e58_watch.sh): wait until the retry script (PID) has exited, find the run id whose
# VM ran (PREFIX + attempt), collect its runs (gpu/collect_b5.sh, e50-e59 names) or copy e6x runs into runs/EP/, then queue one
# evaluation job per net and the summary (evq_loop.sh). usage: gpu_watch.sh RETRY_PID PREFIX EP EVAL.py PRIO N "SUMM" SPRIO
cd /home/agent/workspaces/mats/work/cm-interp-B
PID=$1; PFX=$2; EP=$3; EVAL=$4; PRIO=$5; N=$6; SUMM=$7; SPRIO=$8
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
RID=$(grep -o "attempt [0-9]* (${PFX}[0-9]*): VM was created" GPU-LOG.md | tail -1 | grep -o "${PFX}[0-9]*")
echo "$(date -u +%TZ) retry script exited; rid=$RID" >> /tmp/s6/${EP}_collect.log
[ -n "$RID" ] || exit 1
SRC=/tmp/l4runs/$RID/runs/$RID
mkdir -p runs/$EP
for d in "$SRC"/${EP}_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); [ -e runs/$EP/$n ] && continue
  cp -r "$d" runs/$EP/$n; [ -f "$SRC/$n.log" ] && cp "$SRC/$n.log" runs/$EP/$n.log
done
cp "$SRC/jobs.txt" runs/jobs_$RID.txt 2>/dev/null
echo "$(date -u +%TZ) collected $(ls -d runs/$EP/*/ | wc -l) runs" >> /tmp/s6/${EP}_collect.log
./evq_loop.sh "$EP" "$EVAL" "$PRIO" "$N" "$SUMM" "$SPRIO"
