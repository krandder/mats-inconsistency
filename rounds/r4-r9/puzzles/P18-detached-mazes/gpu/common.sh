# sourced by the P18 GPU scripts (copied from P13/gpu/common.sh, not edited there)
D=/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P18-detached-mazes
G=$D/gpu
PROJ="--project GCP_PROJECT"
RATE=0.71; CAP=40                      # PREREG: ceiling rate $0.71/h (g2-standard-4 + L4 at on-demand list price), cap $40
MAXVMS=${MAXVMS:-12}                  # PREREG: at most 12 P18 VMs at once
DEADLINE_DEL=2026-09-30T17:45:00Z     # final sync + delete of every P18 VM (all deleted before 18:30Z)
SSHO="-i $[ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"
SSH="ssh $SSHO"
U=user
TMPD=/tmp/p18gpu; mkdir -p $TMPD
PFX=trader-dds-
# regions with the fewest trader L4s at 01:20Z first (us-east4, us-west4, europe-west4), then the rest
ZONES="us-east4-c us-east4-a us-west4-a europe-west4-a europe-west4-b us-west1-b us-west1-a us-east1-c us-east1-d us-central1-a us-central1-b us-central1-c europe-west1-b"
ev() { echo "$(date -u +%H:%M:%S) $*"; echo "$(date -u +%FT%TZ) $*" >> $G/ctl.log; }
cur_vm() { awk -v s=$1 '$1==s{l=$0} END{print l}' $G/vms.tsv 2>/dev/null; }
is_deleted() { grep -q "^$1	" $G/deleted.tsv 2>/dev/null; }
vm_ip() { gcloud compute instances describe $1 --zone $2 $PROJ --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null; }
vm_status() { gcloud compute instances describe $1 --zone $2 $PROJ --format="value(status)" 2>/dev/null; }
pending_jobs() {
  local s=$1 L=$D/results/vm/$1/out
  [ -f $G/queues/$s.txt ] || return 0
  grep -v '^#' $G/queues/$s.txt | awk 'NF && $1!="END"{sub(/@.*/,"",$1); print $1}' | while read -r n; do
    [ -f $L/$n.done ] && continue
    [ "$(ls $L/$n.fail.* 2>/dev/null | wc -l)" -ge 2 ] && continue
    echo $n; done
}
ZONES_FIRST="us-west1-b us-west1-a us-west4-a us-east4-c us-central1-b"   # 04:53Z deviation (infra only): us-east4-a 10/10 and us-central1-a/c, europe-west4-b, us-east1-d mostly preempted; these zones had 0 preemptions so far
