#!/bin/bash
# Phase-3 VM control loop, every 2 min, for the VMs listed in results/p3vms.txt ("name zone"):
#  - a VM stopped by a spot preemption (or max-run) whose run is not done is started again; its startup-script
#    relaunches run_p3.sh, which resumes from the checkpoints. Start failures are logged (STARTFAIL) for a manual
#    fallback to a new VM from the synced state.
#  - a VM whose p3.status says P3_DONE: final rsync, sha256 of every remote p3 file against the local copy,
#    and only with zero mismatches the VM is deleted with its disk and the deletion confirmed by gcloud lists.
#  - hard deadline (DEADLINE, default 2026-09-30T02:45Z, before the 8 h max-run of the first VM): every VM still
#    listed gets a best-effort final rsync and is deleted with its disk, done or not (DEADLINE_DELETED).
# Events go to stdout (one line each); exits when every listed VM is deleted.
D=/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet; P="--project GCP_PROJECT"
SSH="ssh -i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
ev() { echo "$(date -u +%H:%M) $*"; echo "$(date -u +%FT%TZ) $*" >> $D/results/p3ctl.log; }
DEADLINE=${DEADLINE:-2026-09-30T02:45:00Z}; DL=$(date -u -d "$DEADLINE" +%s)
while true; do
  left=0
  while read -r n z; do
    [ -z "$n" ] && continue
    st=$(gcloud compute instances describe $n --zone $z $P --format="value(status)" 2>/dev/null)
    [ -z "$st" ] && { grep -q "DELETED $n" $D/results/p3ctl.log || ev "GONE $n (not found)"; continue; }
    left=$((left+1))
    L=$D/results/vm/$n/p3
    if [ "$(date -u +%s)" -ge "$DL" ]; then
      ip=$(gcloud compute instances describe $n --zone $z $P --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
      [ "$st" = RUNNING ] && [ -n "$ip" ] && timeout 600 rsync -a --exclude '*.tmp' -e "$SSH" user@$ip:p3/ $L/
      gcloud compute instances delete $n --zone $z $P --quiet >/dev/null 2>&1
      still=$(gcloud compute instances list $P --filter="name=$n" --format="value(name)"; gcloud compute disks list $P --filter="name=$n" --format="value(name)")
      [ -z "$still" ] && ev "DELETED $n at the deadline $DEADLINE (DEADLINE_DELETED, status was $st; instances and disks lists empty)" || ev "DELETEFAIL $n: $still"
      continue
    fi
    if [ "$st" = TERMINATED ] || [ "$st" = STOPPED ]; then
      if grep -qs P3_DONE $L/p3.status; then st=DONE_STOPPED; else
        if gcloud compute instances start $n --zone $z $P >/dev/null 2>&1; then ev "RESTARTED $n after stop (spot preemption or max-run)"
        else ev "STARTFAIL $n $z (no capacity?)"; fi
        continue
      fi
    fi
    if grep -qs P3_DONE $L/p3.status || [ "$st" = DONE_STOPPED ]; then
      [ "$st" = DONE_STOPPED ] && { gcloud compute instances start $n --zone $z $P >/dev/null 2>&1; sleep 60; }
      ip=$(gcloud compute instances describe $n --zone $z $P --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
      timeout 900 rsync -a --exclude '*.tmp' -e "$SSH" user@$ip:p3/ $L/ || { ev "SYNCFAIL $n"; continue; }
      timeout 300 $SSH user@$ip 'cd p3 && find . -maxdepth 1 -type f ! -name "*.tmp" -exec sha256sum {} +' | sort -k2 > /tmp/mrtest/remote3_$n.sha
      nf=$(wc -l < /tmp/mrtest/remote3_$n.sha); bad=$( (cd $L && sha256sum -c --quiet /tmp/mrtest/remote3_$n.sha 2>&1) | grep -c FAILED)
      if [ "$nf" -gt 0 ] && [ "$bad" = 0 ]; then
        gcloud compute instances delete $n --zone $z $P --quiet >/dev/null 2>&1
        still=$(gcloud compute instances list $P --filter="name=$n" --format="value(name)"; gcloud compute disks list $P --filter="name=$n" --format="value(name)")
        [ -z "$still" ] && ev "DELETED $n after sha256-verified sync of $nf files (instances and disks lists empty)" || ev "DELETEFAIL $n: $still"
      else ev "VERIFYFAIL $n files=$nf mismatches=$bad"; fi
    fi
  done < $D/results/p3vms.txt
  [ $left = 0 ] && { ev "ALL_P3_VMS_DELETED"; exit 0; }
  sleep 120
done
