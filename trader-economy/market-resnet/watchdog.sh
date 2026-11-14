#!/bin/bash
# Spend watchdog + result sync for trader-mresnet-* VMs. Every 5 min:
#  - pulls ~/results and ~/p2 from each VM into results/vm/<name>/ (rsync over gcloud ssh)
#  - estimates spend = sum(hours alive * RATE) incl. deleted VMs (state file)
#  - deletes all trader-mresnet-* VMs when spend >= CAP
RATE=0.32; CAP=18
D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet
ST=$D/results/spend-state.tsv; touch $ST
while true; do
  now=$(date -u +%s)
  gcloud compute instances list --filter="name~^trader-mresnet-" --format="value(name,zone.basename(),creationTimestamp)" > /tmp/mr_vms.txt 2>/dev/null
  while read -r n z c; do
    [ -z "$n" ] && continue
    s=$(date -u -d "$c" +%s)
    grep -q "^$n	$s	" $ST || echo -e "$n\t$s\t$now" >> $ST
    awk -v n=$n -v s=$s -v t=$now 'BEGIN{OFS="\t"} $1==n && $2==s {$3=t} {print}' $ST > $ST.tmp && mv $ST.tmp $ST
    mkdir -p $D/results/vm/$n
    timeout 240 gcloud compute scp --recurse --zone $z "$n:~/results/*" $D/results/vm/$n/ >/dev/null 2>&1
    mkdir -p $D/results/vm/$n/p2
    timeout 240 gcloud compute scp --recurse --zone $z "$n:~/p2/*" $D/results/vm/$n/p2/ >/dev/null 2>&1
  done < /tmp/mr_vms.txt
  spend=$(awk -v r=$RATE '{h+=($3-$2)/3600} END{printf "%.2f", h*r}' $ST)
  echo "$(date -u +%FT%TZ) spend_est=\$$spend vms=$(wc -l < /tmp/mr_vms.txt)" >> $D/results/watchdog.log
  if awk -v s=$spend -v c=$CAP 'BEGIN{exit !(s>=c)}'; then
    while read -r n z c; do gcloud compute instances delete $n --zone $z --quiet; done < /tmp/mr_vms.txt
    /home/agent/.local/bin/fleet-now change trader-mresnet "watchdog deleted trader-mresnet VMs at est spend \$$spend (cap \$$CAP)" --key mresnet-wd
  fi
  sleep 300
done
