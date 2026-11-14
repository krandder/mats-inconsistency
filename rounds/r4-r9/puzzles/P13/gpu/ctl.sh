#!/bin/bash
# P13 controller, every 3 min, for the slots in gpu/slots.list (pattern: market-resnet p4ctl.sh, copied):
#  - live VM: rsync ~/p13/out -> results/vm/SLOT/out (sha256 of every remote file checked against the local copy),
#    push gpu/queues/SLOT.txt, gpu/slots/SLOT (parallel jobs) and gpu/inputs/SLOT;
#    runner done (RUNNER_DONE newer than the VM, no pending jobs), idle (running=0 pending=0) for >= 10 min, or age >= 5.7 h:
#    sha256-verified final sync, delete.
#  - gone VM (spot preemption or the 6 h max-run deleted it): recorded in deleted.tsv.
#  - slot with no live VM and pending jobs in its queue: launch.sh creates SLOT-rN and resumes from the synced state
#    (at most MAXVMS VMs up at once, slots in slots.list order).
#  - at DEADLINE_DEL (16:55Z 30 Sep) or with gpu/STOP: best-effort sync, delete everything, exit.
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/gpu/common.sh
touch $G/vms.tsv $G/deleted.tsv $G/ctl.log
DL=$(date -u -d "$DEADLINE_DEL" +%s)
sync_slot() {   # sync_slot SLOT VM IP -> prints "files mismatches"
  local S=$1 N=$2 IP=$3 L=$D/results/vm/$1/out
  mkdir -p $L
  timeout 900 rsync -a --exclude '*.tmp' --exclude '*.tmp.npz' -e "$SSH" $U@$IP:p13/out/ $L/ >/dev/null 2>&1 || { echo "-1 -1"; return; }
  timeout 300 $SSH $U@$IP 'cd p13/out && find . -type f ! -name "*.tmp" ! -name "*.tmp.npz" -exec sha256sum {} +' 2>/dev/null | sort -k2 > $TMPD/remote_$N.sha
  local nf=$(wc -l < $TMPD/remote_$N.sha)
  local bad=$( (cd $L && sha256sum -c --quiet $TMPD/remote_$N.sha 2>&1) | grep -c FAILED)
  cp $TMPD/remote_$N.sha $D/results/vm/$S/SHA256SUMS.remote
  echo "$nf $bad"
}
delete_vm() {   # delete_vm SLOT VM ZONE REASON
  gcloud compute instances delete $2 --zone $3 $PROJ --quiet >/dev/null 2>&1
  local still=$(gcloud compute instances list $PROJ --filter="name=$2" --format="value(name)"; gcloud compute disks list $PROJ --filter="name=$2" --format="value(name)")
  if [ -z "$still" ]; then echo -e "$2\t$(date -u +%FT%TZ)\t$4" >> $G/deleted.tsv; ev "DELETED $1 $2 ($4; instances and disks lists empty)"
  else ev "DELETEFAIL $2: $still"; fi
}
while true; do
  now=$(date -u +%s)
  stop=0; [ -f $G/STOP ] && stop=1; [ $now -ge $DL ] && stop=1
  live=0
  # VMs up now (PREREG: P13 uses up to MAXVMS VMs at once); slots earlier in slots.list launch first
  nup=0; for S in $(cat $G/slots.list 2>/dev/null); do read -r _ N _ _ <<< "$(cur_vm $S)"; [ -n "$N" ] && ! is_deleted $N && nup=$((nup+1)); done
  for S in $(cat $G/slots.list 2>/dev/null); do
    read -r _ N Z C <<< "$(cur_vm $S)"
    if [ -n "$N" ] && ! is_deleted $N; then
      st=$(vm_status $N $Z)
      if [ -z "$st" ]; then
        echo -e "$N\t$(date -u +%FT%TZ)\tgone (spot preemption or max-run; GCE deleted VM and disk)" >> $G/deleted.tsv
        ev "GONE $S $N"; continue
      fi
      live=$((live+1))
      [ "$st" != RUNNING ] && { ev "STATUS $N $st"; continue; }
      IP=$(vm_ip $N $Z)
      [ -f $G/queues/$S.txt ] && rsync -a -e "$SSH" $G/queues/$S.txt $U@$IP:p13/out/queue.txt 2>/dev/null
      [ -f $G/slots/$S ] && rsync -a -e "$SSH" $G/slots/$S $U@$IP:p13/out/SLOTS 2>/dev/null
      [ -d $G/inputs/$S ] && rsync -a -e "$SSH" $G/inputs/$S/ $U@$IP:p13/in/ 2>/dev/null
      read -r nf bad <<< "$(sync_slot $S $N $IP)"
      echo "$(date -u +%FT%TZ) $S $N files=$nf mismatch=$bad $(head -1 $D/results/vm/$S/out/HEARTBEAT 2>/dev/null)" >> $G/sync.log
      for f in $(cd $D/results/vm/$S/out 2>/dev/null && ls *.fail.* *.done 2>/dev/null); do
        grep -qx "$S/$f" $G/seen_markers.txt 2>/dev/null && continue
        echo "$S/$f" >> $G/seen_markers.txt
        case $f in *.fail.*) ev "JOBFAIL $S $f: $(tail -n 2 $D/results/vm/$S/out/logs/${f%%.fail.*}.log 2>/dev/null | tr '\n' ' ' | cut -c1-200)";; *) ev "JOBDONE $S ${f%.done}";; esac
      done
      age=$(( now - $(date -u -d "$C" +%s) ))
      hb=$(head -1 $D/results/vm/$S/out/HEARTBEAT 2>/dev/null)
      hbt=$(date -u -d "$(echo $hb | awk '{print $1}')" +%s 2>/dev/null || echo 0)
      if echo "$hb" | grep -q "running=0 pending=0" && [ "$hbt" -gt "$(date -u -d "$C" +%s)" ]; then [ -f $TMPD/idle_$N ] || date -u +%s > $TMPD/idle_$N
      else rm -f $TMPD/idle_$N; fi
      idle=0; [ -f $TMPD/idle_$N ] && idle=$(( now - $(cat $TMPD/idle_$N) ))
      rdt=$(date -u -d "$(head -1 $D/results/vm/$S/out/RUNNER_DONE 2>/dev/null)" +%s 2>/dev/null || echo 0)
      why=""
      [ $stop = 1 ] && why="stop/deadline"
      [ $idle -ge 600 ] && why="idle $((idle/60)) min, queue done"
      [ "$rdt" -gt "$(date -u -d "$C" +%s)" ] && [ -z "$(pending_jobs $S)" ] && why="runner done (queue ended, no pending jobs)"
      [ $age -ge 20520 ] && why="age $((age/60)) min (>= 5.7 h)"
      if [ -n "$why" ]; then
        read -r nf bad <<< "$(sync_slot $S $N $IP)"
        if [ "$nf" -gt 0 ] && [ "$bad" = 0 ]; then delete_vm $S $N $Z "$why; sha256-verified sync of $nf files"
        elif [ $stop = 1 ] || [ $age -ge 20880 ]; then delete_vm $S $N $Z "$why; sync NOT verified (files=$nf mismatches=$bad)"
        else ev "VERIFYFAIL $N files=$nf mismatches=$bad ($why); retry next round"; fi
      fi
      continue
    fi
    # no live VM for this slot
    [ $stop = 1 ] && continue
    p=$(pending_jobs $S | wc -l)
    [ "$p" = 0 ] && continue
    nvm=$(awk -v s=$S '$1==s' $G/vms.tsv | wc -l)
    [ $nvm -ge 16 ] && { grep -q "GAVEUP $S" $G/ctl.log || ev "GAVEUP $S after $nvm VMs"; continue; }
    [ $(( DL - now )) -lt 1800 ] && continue
    [ $nup -ge $MAXVMS ] && continue
    nup=$((nup+1))
    ev "LAUNCH $S ($p pending jobs)"
    ( $G/launch.sh $S >> $TMPD/launch_$S.log 2>&1 ) &
    live=$((live+1))
  done
  if [ $stop = 1 ]; then
    left=$(gcloud compute instances list $PROJ --filter="name~^trader-b3-p13" --format="value(name,zone.basename())")
    if [ -z "$left" ]; then ev "CTL_STOP: no trader-b3-p13 VMs left"; exit 0; fi
    echo "$left" | while read -r n z; do grep -q "^$n	" $G/deleted.tsv || delete_vm "?" $n $z "stop/deadline sweep"; done
  fi
  sleep 180
done
