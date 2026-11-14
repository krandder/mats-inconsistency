#!/bin/bash
# Phase-3b VM control loop, every 2 min, for the VMs in results/p4vms.txt ("name zone seed kind"); the VMs have
# termination action DELETE, so a spot preemption or the 4 h max-run deletes the VM and its disk.
#  - a listed VM that is gone before we deleted it, and whose synced p3.status has no P3_DONE, is recreated under
#    the same name by launch4.sh ... relaunch, which pushes the synced state back (resume); at most 6 relaunches (3 before 23:06 UTC).
#  - a VM whose p3.status says P3_DONE: final rsync, sha256 of every remote p3 file against the local copy, and
#    only with zero mismatches the VM is deleted with its disk and the deletion confirmed by gcloud lists.
#  - hard deadline (DEADLINE, default 2026-09-30T04:00Z): best-effort final rsync, then delete, done or not.
# Exits when every listed VM is deleted (or given up).
D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet; P="--project GCP_PROJECT"
SSH="ssh -i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
LOG=$D/results/p4ctl.log; touch $LOG
ev() { echo "$(date -u +%H:%M) $*"; echo "$(date -u +%FT%TZ) $*" >> $LOG; }
DEADLINE=${DEADLINE:-2026-09-30T04:00:00Z}; DL=$(date -u -d "$DEADLINE" +%s)
while true; do
  left=0
  cp $D/results/p4vms.txt /tmp/mrtest/p4vms.snap
  while read -r n z s k <&3; do   # 23:47 UTC fix: read the list on fd 3; ssh and launch4.sh used to eat stdin, so after one
                                  # verified delete or relaunch the rest of the list was skipped until the next round
    [ -z "$n" ] && continue
    grep -q "DELETED $n\|GAVEUP $n\|GONE_DONE $n" $LOG && continue
    left=$((left+1))
    L=$D/results/vm/$n/p3
    st=$(gcloud compute instances describe $n --zone $z $P --format="value(status)" 2>/dev/null)
    if [ -z "$st" ]; then
      if grep -qs P3_DONE $L/p3.status; then ev "GONE_DONE $n (deleted by GCE after P3_DONE; local copy from the last 5-min sync)"; continue; fi
      if [ "$(date -u +%s)" -ge "$DL" ]; then ev "GAVEUP $n (gone, past the deadline)"; continue; fi
      nr=$(grep -c "RELAUNCH $n" $LOG)   # (21:40 UTC fix: this count used to overwrite $k, the VM's kind)
      if [ "$nr" -ge 6 ]; then ev "GAVEUP $n after $nr relaunches"; continue; fi   # 23:06 UTC: was 3; heavy spot churn, and reeval.py now resumes inside a block
      ev "RELAUNCH $n seed $s kind ${k:-p3b} (gone: spot preemption or max-run; try $((nr+1)))"
      ( $D/launch4.sh $n $s relaunch ${k:-p3b} 2>&1 | tail -3 | sed "s/^/  /" ) >> $LOG
      continue
    fi
    [ "$st" != RUNNING ] && continue      # STOPPING / TERMINATED: GCE is deleting it; check again next round
    ip=$(gcloud compute instances describe $n --zone $z $P --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
    if [ "$(date -u +%s)" -ge "$DL" ]; then
      [ -n "$ip" ] && timeout 600 rsync -a --exclude '*.tmp' -e "$SSH" user@$ip:p3/ $L/
      gcloud compute instances delete $n --zone $z $P --quiet >/dev/null 2>&1
      still=$(gcloud compute instances list $P --filter="name=$n" --format="value(name)"; gcloud compute disks list $P --filter="name=$n" --format="value(name)")
      [ -z "$still" ] && ev "DELETED $n at the deadline $DEADLINE (DEADLINE_DELETED; instances and disks lists empty)" || ev "DELETEFAIL $n: $still"
      continue
    fi
    if grep -qs P3_DONE $L/p3.status; then
      timeout 900 rsync -a --exclude '*.tmp' -e "$SSH" user@$ip:p3/ $L/ || { ev "SYNCFAIL $n"; continue; }
      timeout 300 $SSH user@$ip 'cd p3 && find . -maxdepth 1 -type f ! -name "*.tmp" -exec sha256sum {} +' | sort -k2 > /tmp/mrtest/remote4_$n.sha
      nf=$(wc -l < /tmp/mrtest/remote4_$n.sha); bad=$( (cd $L && sha256sum -c --quiet /tmp/mrtest/remote4_$n.sha 2>&1) | grep -c FAILED)
      if [ "$nf" -gt 0 ] && [ "$bad" = 0 ]; then
        gcloud compute instances delete $n --zone $z $P --quiet >/dev/null 2>&1
        still=$(gcloud compute instances list $P --filter="name=$n" --format="value(name)"; gcloud compute disks list $P --filter="name=$n" --format="value(name)")
        [ -z "$still" ] && ev "DELETED $n after sha256-verified sync of $nf files (instances and disks lists empty)" || ev "DELETEFAIL $n: $still"
      else ev "VERIFYFAIL $n files=$nf mismatches=$bad"; fi
    fi
  done 3< /tmp/mrtest/p4vms.snap
  [ $left = 0 ] && { ev "ALL_P4_VMS_DELETED"; exit 0; }
  sleep 120
done
