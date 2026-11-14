#!/bin/bash
# Phase-3 spend watchdog + result sync for trader-mresnet-* VMs (replaces watchdog.sh, same state file). Every 5 min:
#  - rsyncs ~/p3 of each RUNNING VM into results/vm/<name>/p3/ (incremental; checkpoints included)
#  - estimates spend = sum(hours since creation * RATE) incl. deleted and stopped VMs (conservative)
#  - deletes all trader-mresnet-* VMs when spend >= CAP
RATE=0.32; CAP=21.2   # 22:58 UTC: was 19.2; spot-preemption churn (about 20 relaunches since 21:30) re-ran partial blocks; +$2 over the $15 in-window allocation, as the coordinator allowed ("a few extra dollars are fine"), logged in BUDGET-24H-20260929.md. 21:47 UTC: was 18; 19.2 = the $20 experiment cap with margin, and $4.26 (phases 1-2) + $14.94 <= the $15 in-window allocation of BUDGET-24H-20260929.md
D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet
ST=$D/results/spend-state.tsv; touch $ST
SSH="ssh -i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
while true; do
  now=$(date -u +%s)
  gcloud compute instances list --project GCP_PROJECT --filter="name~^trader-mresnet-" --format="value(name,zone.basename(),creationTimestamp,status,networkInterfaces[0].accessConfigs[0].natIP)" > /tmp/mr_vms3.txt 2>/dev/null
  while read -r n z c s ip; do
    [ -z "$n" ] && continue
    t=$(date -u -d "$c" +%s)
    grep -q "^$n	$t	" $ST || echo -e "$n\t$t\t$now" >> $ST
    awk -v n=$n -v s=$t -v t=$now 'BEGIN{OFS="\t"} $1==n && $2==s {$3=t} {print}' $ST > $ST.tmp && mv $ST.tmp $ST
    if [ "$s" = RUNNING ] && [ -n "$ip" ]; then
      mkdir -p $D/results/vm/$n/p3
      timeout 600 rsync -a --exclude '*.tmp' -e "$SSH" user@$ip:p3/ $D/results/vm/$n/p3/ >/dev/null 2>&1
    fi
  done < /tmp/mr_vms3.txt
  spend=$(awk -v r=$RATE '{h+=($3-$2)/3600} END{printf "%.2f", h*r}' $ST)
  echo "$(date -u +%FT%TZ) spend_est=\$$spend vms=$(wc -l < /tmp/mr_vms3.txt) $(awk '{printf "%s:%s ", $1, $4}' /tmp/mr_vms3.txt)" >> $D/results/watchdog.log
  if awk -v s=$spend -v c=$CAP 'BEGIN{exit !(s>=c)}'; then
    while read -r n z c s ip; do gcloud compute instances delete $n --zone $z --project GCP_PROJECT --quiet; done < /tmp/mr_vms3.txt
    /home/agent/.local/bin/fleet-now change trader-mresnet "watchdog deleted trader-mresnet VMs at est spend \$$spend (cap \$$CAP)" --key mresnet-wd
  fi
  sleep 300
done
