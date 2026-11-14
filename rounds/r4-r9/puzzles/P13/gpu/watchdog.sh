#!/bin/bash
# P13 spend watchdog, every 5 min, independent of ctl.sh (pattern: market-resnet watchdog3.sh, copied):
#  - spend = sum over every trader-b3-p13-* VM ever seen (deleted ones included) of (last seen - created + 5 min) x $0.35/h
#  - spend >= $40: touch gpu/STOP and delete every trader-b3-p13-* VM at once
#  - any VM at age >= 5.8 h, or whose synced runner heartbeat says idle (or is stale) for >= 20 min: delete
#  - rerenders the P13 section of GPU-LEDGER.md when it changed
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/gpu/common.sh
ST=$G/spend-state.tsv; touch $ST
while true; do
  now=$(date -u +%s)
  gcloud compute instances list $PROJ --filter="name~^trader-b3-p13-" --format="value(name,zone.basename(),creationTimestamp,status)" > $TMPD/wd_vms.txt 2>/dev/null || { sleep 60; continue; }
  while read -r n z c s; do
    [ -z "$n" ] && continue
    t=$(date -u -d "$c" +%s)
    grep -q "^$n	$t	" $ST || echo -e "$n\t$t\t$now\t$z" >> $ST
    awk -v n=$n -v s=$t -v t=$now 'BEGIN{OFS="\t"} $1==n && $2==s {$3=t} {print}' $ST > $ST.tmp && mv $ST.tmp $ST
  done < $TMPD/wd_vms.txt
  spend=$(awk -v r=$RATE '{h+=($3-$2+300)/3600} END{printf "%.2f", h*r}' $ST)
  echo "$(date -u +%FT%TZ) spend_est=\$$spend vms=$(grep -c . $TMPD/wd_vms.txt) $(awk '{printf "%s:%s ", $1, $4}' $TMPD/wd_vms.txt)" >> $G/watchdog.log
  echo "$spend" > $G/SPEND
  if awk -v s=$spend -v c=$CAP 'BEGIN{exit !(s>=c)}'; then
    touch $G/STOP
    while read -r n z c s; do [ -n "$n" ] && gcloud compute instances delete $n --zone $z $PROJ --quiet >/dev/null 2>&1 &
    done < $TMPD/wd_vms.txt; wait
    ev "WATCHDOG_CAP deleted all trader-b3-p13 VMs at est spend \$$spend (cap \$$CAP)"
    /home/agent/.local/bin/fleet-now change calibrated-ai "P13 watchdog deleted all trader-b3-p13 VMs at est spend \$$spend (cap \$$CAP)" --key p13-wd >/dev/null 2>&1
  fi
  while read -r n z c s; do
    [ -z "$n" ] && continue
    age=$(( now - $(date -u -d "$c" +%s) ))
    slot=$(awk -v n=$n '$2==n{print $1}' $G/vms.tsv | tail -1)
    hbf=$D/results/vm/$slot/out/HEARTBEAT
    why=""
    [ $age -ge 20880 ] && why="age $((age/60)) min >= 5.8 h"
    if [ -z "$why" ] && [ $age -ge 2400 ] && [ -n "$slot" ]; then
      hb=$(head -1 $hbf 2>/dev/null); hbt=$(date -u -d "$(echo $hb | awk '{print $1}')" +%s 2>/dev/null || echo 0)
      if [ $(( now - hbt )) -ge 1200 ]; then why="heartbeat stale $(( (now-hbt)/60 )) min"
      elif echo "$hb" | grep -q "running=0 pending=0"; then
        [ -f $TMPD/wdidle_$n ] || echo $now > $TMPD/wdidle_$n
        [ $(( now - $(cat $TMPD/wdidle_$n) )) -ge 1200 ] && why="idle >= 20 min"
      else rm -f $TMPD/wdidle_$n; fi
    fi
    if [ -n "$why" ]; then
      gcloud compute instances delete $n --zone $z $PROJ --quiet >/dev/null 2>&1
      echo -e "$n\t$(date -u +%FT%TZ)\twatchdog: $why" >> $G/deleted.tsv
      ev "WATCHDOG_DELETE $n ($why)"
    fi
  done < $TMPD/wd_vms.txt
  /home/agent/.venvs/trader-cat/bin/python $G/ledger.py >/dev/null 2>&1
  sleep 300
done
