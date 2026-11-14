# sourced by the P13 GPU scripts (copied pattern from market-resnet mkvm4/launch4/watchdog3/p4ctl, not edited there)
D=/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13
G=$D/gpu
PROJ="--project GCP_PROJECT"
RATE=0.35; CAP=40
MAXVMS=${MAXVMS:-10}                 # PREREG: P13 uses up to 10 VMs at once
DEADLINE_DEL=2026-09-30T16:55:00Z     # final sync + delete of every P13 VM (hard stop 17:17Z)
SSHO="-i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
SSH="ssh $SSHO"
U=user
TMPD=/tmp/p13gpu; mkdir -p $TMPD
ZONES="us-central1-a us-central1-b us-central1-c us-east4-c us-east4-a us-west1-b us-west1-a us-east1-c us-east1-d us-west4-a europe-west4-a europe-west4-b europe-west1-b"
ev() { echo "$(date -u +%H:%M:%S) $*"; echo "$(date -u +%FT%TZ) $*" >> $G/ctl.log; }
# current VM of a slot: last line "slot vm zone created" in vms.tsv whose vm is not in deleted.tsv
cur_vm() { awk -v s=$1 '$1==s{l=$0} END{print l}' $G/vms.tsv 2>/dev/null; }
is_deleted() { grep -q "^$1	" $G/deleted.tsv 2>/dev/null; }
vm_ip() { gcloud compute instances describe $1 --zone $2 $PROJ --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null; }
vm_status() { gcloud compute instances describe $1 --zone $2 $PROJ --format="value(status)" 2>/dev/null; }
# pending jobs of a slot, judged from the local synced copy
pending_jobs() {
  local s=$1 L=$D/results/vm/$1/out
  [ -f $G/queues/$s.txt ] || return 0
  grep -v '^#' $G/queues/$s.txt | awk 'NF && $1!="END"{sub(/@.*/,"",$1); print $1}' | while read -r n; do
    [ -f $L/$n.done ] && continue
    [ "$(ls $L/$n.fail.* 2>/dev/null | wc -l)" -ge 2 ] && continue
    echo $n; done
}
