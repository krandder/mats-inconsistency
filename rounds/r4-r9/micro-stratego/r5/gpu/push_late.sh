#!/bin/bash
# push_late.sh : push the staged tree /tmp/r5e2/src (late oracle files + code) to every live r5e2-s10x VM (additive
# rsync, no --delete), then set /root/soup/data/READY_ALL so the waiting eval steps start. Needed because the driver's
# launch ssh stays attached to the remote pipeline (sudo keeps the session), so drive.sh's own push never runs mid-run.
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"
for RID in ${RUNS:-r5e2-s101 r5e2-s102 r5e2-s103-b}; do
  f=$STATE/vm-$RID
  [[ -s $f ]] || { echo "no state for $RID"; continue; }
  read -r NAME ZONE < "$f"
  IP=$(vm_ip "$NAME" "$ZONE") || { echo "no ip for $NAME"; continue; }
  rsync -az --rsync-path='sudo rsync' -e "$RSH" /tmp/r5e2/src/ "$SSH_USER@$IP:/root/soup/" \
    && vm_ssh "$IP" "sudo bash -c 'ls /root/soup/data | tr \"\\n\" \" \"; touch /root/soup/data/READY_ALL'" \
    && touch /tmp/r5e2/pushed-$NAME && echo "$(date -u +%FT%TZ) pushed $RID $NAME $IP"
done
