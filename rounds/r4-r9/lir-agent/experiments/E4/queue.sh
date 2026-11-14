#!/bin/bash
# Background queue: launch E4 rentals in priority order as group admission allows, max 3 E4 VMs at once.
# Stops launching when lir-agent spend (actual + reservations) leaves no room (gpu-up refuses) or E4 actual >= $15.
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E4
Q=$E/runs/queue.state; touch $Q
JOBS="F:syco16:0 R2:syco16:0 F:syco16:1 F:syco16:2 R2:syco16:2 MM:syco16:02 Y:syco:0 Y:syco:1 Y:syco:2"
e4spent() { grep -E "DELETE vm=soup-gpu-lir-e4-" /home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md | sed -n 's/.*cost_usd=\([0-9.]*\).*/\1/p' | awk '{s+=$1} END{printf "%.2f", s+0}'; }
while true; do
  pending=0; running=$(grep -c ' running$' $Q)
  for j in $JOBS; do
    st=$(grep "^$j " $Q | tail -n 1 | awk '{print $2}')
    [ "$st" = done ] && continue
    if [ "$st" = running ]; then
      IFS=: read K P S <<< "$j"
      if grep -q "launch $K $P $S rc=" $E/runs/launch.log 2>/dev/null; then
        rc=$(grep "launch $K $P $S rc=" $E/runs/launch.log | tail -n 1 | sed 's/.*rc=//')
        if [ "$rc" = 3 ] || [ "$rc" = 1 ] && grep -q REFUSE $E/runs/$P/launch-$K-s$S.log; then
          echo "$j refused $(date -u +%H:%MZ)" >> $Q; sed -i "/launch $K $P $S rc=/d" $E/runs/launch.log
        else
          echo "$j done rc=$rc $(date -u +%H:%MZ)" >> $Q
        fi
      fi
      pending=1; continue
    fi
    pending=1
    running=$(for x in $JOBS; do grep "^$x " $Q | tail -n 1; done | grep -c ' running$')
    if [ "$running" -lt 3 ]; then
      sp=$(e4spent); if awk -v s=$sp 'BEGIN{exit !(s>=15)}'; then echo "STOP e4 spent $sp" >> $Q; exit 0; fi
      IFS=: read K P S <<< "$j"
      echo "$j running $(date -u +%H:%MZ)" >> $Q
      sed -i 's/ running \([0-9:Z]*\)$/ running/' $Q
      nohup $E/launch.sh $K $P $S > /dev/null 2>&1 &
      sleep 45
    fi
  done
  [ $pending = 0 ] && { echo "ALL DONE $(date -u +%H:%MZ)" >> $Q; exit 0; }
  sleep 120
done
