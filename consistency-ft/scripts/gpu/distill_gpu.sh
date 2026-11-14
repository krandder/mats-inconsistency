#!/bin/bash
# Spot A100 VMs for the consistency-distillation run (after eurisko-study/logical-soup/infra/gpu-*.sh).
#
#   distill_gpu.sh up NAME "Qwen3-0.6B ..."   create a spot a2-highgpu-1g (label cdistill=1), wait for READY
#   distill_gpu.sh start NAME MODEL "ARMS" [flags]  copy code+bundle, start the job detached (nohup)
#   distill_gpu.sh status NAME                show job markers and the last log lines
#   distill_gpu.sh sync NAME                  copy /root/out back (tar + sha256 verify) into $OUT/gpu/NAME
#   distill_gpu.sh down NAME [reason]         sync, then DELETE the VM (+ boot disk), verify it is gone
#   distill_gpu.sh watchdog                   loop: delete any cdistill VM on spend cap, age limit or idle job
#   distill_gpu.sh spend                      estimated spend from the ledger
#
# Checkpointed jobs (CDISTILL_JOB=distill2, the v2 run; see distill2_gpu.sh): a GROUP of VMs named
# $LABEL-GROUP1, $LABEL-GROUP2, ... runs one queue; only one VM of a group is alive at a time.
#   distill_gpu.sh supervise GROUP "MODELS" MODEL "QUEUE" [flags]   keep the group's queue running to the end:
#       create a VM, restore the last verified sync, start the queue (items resume from their checkpoints),
#       sync every $SYNC_MIN min (sha256-verified), and when the VM disappears (spot preemption, max run
#       duration) log it and create the next VM of the group; when every item has result.json: final
#       sync, delete the VM, verify that it and its disk are gone. `touch $STATE/GROUP/STOP` ends it.
#   distill_gpu.sh gsync NAME                 one verified sync of a group VM into $STATE/GROUP/current
#   distill_gpu.sh restore NAME               push $STATE/GROUP/current to a new VM, verify sha256 there
#
# Stop guarantees: (1) every VM is created with --max-run-duration=$MAX_RUN and
# --instance-termination-action=DELETE (GCE deletes it, boot disk auto-deletes); (2) the watchdog
# loop deletes on the spend cap ($CAP_USD, all VMs of this job), age, or a job that is finished or
# idle; (3) `down` after every job, then `gcloud compute instances list` must show no cdistill VM.
# Spend = VM-hours x RATE (the on-demand list price of a2-highgpu-1g, a ceiling: the Billing API is
# disabled in this project, so spot prices are not verifiable here; spot is lower).
#
# Other jobs reuse this launcher through environment variables (defaults = the distillation run):
#   CDISTILL_LABEL   VM label and name prefix (cdistill; the prequential study uses cdistill-preq);
#                    the watchdog, spend and list only see VMs and ledger lines of this prefix
#   CDISTILL_JOB     distill | preq: which worker, remote runner and bundle files `start` copies
#   CDISTILL_OUT, CDISTILL_LEDGER, CDISTILL_SOURCE (alert source), CDISTILL_CAP_USD, CDISTILL_MAX_RUN,
#   CDISTILL_MAX_AGE_H, CDISTILL_MACHINE, CDISTILL_ZONES
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${CDISTILL_OUT:-/home/agent/data/consistency-distill-20260929}
STATE="$OUT/gpu"; mkdir -p "$STATE"
LEDGER=${CDISTILL_LEDGER:-$STATE/LEDGER.md}
PROJECT=GCP_PROJECT
LABEL=${CDISTILL_LABEL:-cdistill}
SRC=${CDISTILL_SOURCE:-consistency-distill}
JOB=${CDISTILL_JOB:-distill}
if [[ $JOB == preq ]]; then
  WORKER=preq_student.py; REMOTE=preq_remote.sh; JOB_FILES=(scripts/preq_student.py scripts/preq_check_packing.py scripts/preq_check_precision.py)
  BUNDLE_FILES=(stream.jsonl.gz blocks.json)
elif [[ $JOB == distill2 ]]; then
  WORKER=distill_student.py; REMOTE=distill2_remote.sh; JOB_FILES=(scripts/distill_student.py)
  BUNDLE_FILES=(train.jsonl validation.jsonl test.jsonl)
  # teacher -> bundle directory on General (copied to /root/job/bundle-TEACHER)
  declare -A TEACHER_BUNDLE=([32b]=/home/agent/data/consistency-distill-20260929/bundle
                             [235b]=/home/agent/data/consistency-distill-v2-20260929/bundle-235b)
else
  WORKER=distill_student.py; REMOTE=distill_remote.sh; JOB_FILES=(scripts/distill_student.py)
  BUNDLE_FILES=(train.jsonl validation.jsonl test.jsonl)
fi
MACHINE=${CDISTILL_MACHINE:-a2-highgpu-1g}
if [[ $MACHINE == g2-standard-8 ]]; then  # L4 24GB fallback
  ACCEL=nvidia-l4; RATE=0.86; DEF_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c europe-west4-a europe-west4-b europe-west4-c"
else
  MACHINE=a2-highgpu-1g; ACCEL=nvidia-tesla-a100; RATE=3.69; DEF_ZONES="us-central1-a us-central1-b us-central1-c us-central1-f europe-west4-a europe-west4-b"
fi
ZONES=(${CDISTILL_ZONES:-$DEF_ZONES})
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
# RATE: $/h ceiling per VM (on-demand list price incl. GPU), recorded on each CREATE line
CAP_USD=${CDISTILL_CAP_USD:-35}
MAX_RUN=${CDISTILL_MAX_RUN:-5h}
MAX_AGE_H=${CDISTILL_MAX_AGE_H:-5.2}
IDLE_MIN=20
SYNC_MIN=${CDISTILL_SYNC_MIN:-5}
NPAR=${CDISTILL_NPAR:-2}
SSH_USER=cdistill; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source "$SRC" "$*" >/dev/null 2>&1 || true; }
vms() {  # name zone creationTimestamp status
  "${GC[@]}" compute instances list --filter="labels.$LABEL:*" \
    --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null
}
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }

# CDISTILL_SPEND_BASIS=spot counts each VM at Google's Spot price for its machine type and region, plus $0.03/h
# for the boot disk and IP. Prices: Cloud Billing Catalog API (Compute Engine SKUs, effective 2026-09-29 07:00Z,
# read 19:25Z; vCPU x core + GiB x RAM + GPU; snapshot in $OUT/gpu/spot-prices-20260929.json). They match
# cloud.google.com/spot-vms/pricing for us-central1 and its highest prices. A region not listed counts at the
# highest price of that type in any region. Default basis: the on-demand list price on the CREATE line.
SPEND_BASIS=${CDISTILL_SPEND_BASIS:-ceiling}
SPOT_RATES="g2-standard-8:us-central1=0.512112 g2-standard-8:us-east4=0.453944 g2-standard-8:us-east1=0.512112 g2-standard-8:us-west1=0.512112 g2-standard-8:us-west4=0.576916 g2-standard-8:europe-west4=0.538252 g2-standard-8:*=0.735788"
SPOT_RATES+=" a2-highgpu-1g:us-west4=2.360575 a2-highgpu-1g:us-central1=2.20401 a2-highgpu-1g:us-east1=2.20401 a2-highgpu-1g:us-west1=2.20401 a2-highgpu-1g:us-west3=2.588465 a2-highgpu-1g:europe-west4=2.24829 a2-highgpu-1g:asia-northeast1=2.42359 a2-highgpu-1g:asia-northeast3=2.42359 a2-highgpu-1g:asia-southeast1=2.40226 a2-highgpu-1g:me-west1=2.42438 a2-highgpu-1g:*=2.588465"
spend() {  # ledger: CREATE name epoch / DELETE name epoch; running VMs count to now; only VMs of this prefix
  awk -v now="$(date +%s)" -v rate="$RATE" -v pre="vm=$LABEL-" -v basis="$SPEND_BASIS" -v spot="$SPOT_RATES" '
    BEGIN { ns=split(spot, sp, " "); for(i=1;i<=ns;i++){ split(sp[i], kv, "="); srate[kv[1]]=kv[2] } }
    index($0, pre) == 0 { next }
    / CREATE / { r=rate; ty=""; z=""
      for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); if($i~/^rate_ceiling=/)r=substr($i,14)
                          if($i~/^type=/)ty=substr($i,6); if($i~/^zone=/)z=substr($i,6) }
      if(basis=="spot"){ reg=z; sub(/-[a-z]$/, "", reg)
        if((ty ":" reg) in srate) r=srate[ty ":" reg]+0.03; else if((ty ":*") in srate) r=srate[ty ":*"]+0.03 }
      start[n]=e; rt[n]=r }
    / DELETE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7) } if(!(n in stop)) stop[n]=e }
    END { t=0; for(n in start){ s=(n in stop?stop[n]:now); t+=(s-start[n])/3600*rt[n] } printf "%.2f\n", t }' "$LEDGER" 2>/dev/null || echo 0
}

hot_zones() {  # zones where a VM of this prefix was preempted in the last CDISTILL_ZONE_COOL_MIN (30) minutes
  TZ=UTC awk -v now="$(date +%s)" -v cool="$(( ${CDISTILL_ZONE_COOL_MIN:-30} * 60 ))" -v pre="vm=$LABEL-" '
    index($0, pre) == 0 { next }
    { v=""; z=""; for(i=1;i<=NF;i++){ if($i ~ /^vm=/) v=substr($i,4); if($i ~ /^zone=/) z=substr($i,6) } }
    / CREATE / && z != "" { zone[v]=z }
    / PREEMPTED / && (v in zone) { ts=$2; gsub(/[-T:Z]/, " ", ts); if(now - mktime(ts) < cool) hot[zone[v]]=1 }
    END { for(z in hot) print z }' "$LEDGER"
}

cmd_up() {
  local name=$1 models=$2
  [[ $name =~ ^$LABEL-[a-z0-9-]+$ ]] || { echo "name must be $LABEL-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then
    echo "REFUSE: spend \$$s + half an hour > cap \$$CAP_USD" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/cdistill-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone="" hot z zs=()
  # a zone that just preempted a spot VM tends to preempt the next one within minutes (us-central1-a/-c,
  # 09-29 19:04-19:37Z): try it last for CDISTILL_ZONE_COOL_MIN minutes
  hot=" $(hot_zones | tr '\n' ' ') "
  for z in "${ZONES[@]}"; do [[ $hot == *" $z "* ]] || zs+=("$z"); done
  for z in "${ZONES[@]}"; do [[ $hot != *" $z "* ]] || zs+=("$z"); done
  [[ $hot == "  " ]] || echo "zones with a recent preemption, tried last:$hot"
  for z in "${zs[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" \
        --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=120GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="$LABEL=1" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True,cdistill-models="$models" \
        --metadata-from-file="startup-script=$REPO/scripts/gpu/distill_startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
    "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q . && cmd_delete "$name" "create failed"
    trash-put "$tmp"; exit 1
  fi
  trash-put "$tmp"
  log "CREATE vm=$name zone=$zone epoch=$(date +%s) type=$MACHINE spot max_run=$MAX_RUN rate_ceiling=$RATE models=\"$models\" spend_before=$s"
  alert "$SRC: spot $ACCEL VM $name created in $zone (self-deletes after $MAX_RUN; job cap \$$CAP_USD)"
  wait_ready "$name" "$zone"
}

wait_ready() {  # NAME ZONE: wait for READY; exit 1 = startup failed or timed out (VM deleted), 3 = VM gone (spot preemption)
  local name=$1 zone=$2 ip deadline st s
  ip=$(ip_of "$name" "$zone") || exit 3
  echo "waiting for $name ($ip) startup (up to 30 min) ..."
  deadline=$(( $(date +%s) + 1800 ))
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/cdistill/READY 2>/dev/null || { [ -f /var/lib/cdistill/FAILED ] && echo FAILED: $(cat /var/lib/cdistill/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then
      rsh "$ip" 'sudo tail -30 /var/log/cdistill-startup.log' 2>/dev/null
      log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1
    fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    # a spot VM can be preempted during startup (cdistill-v2-t3, 09-29): stop waiting at once
    if s=$("${GC[@]}" compute instances list --filter="name=$name" --format='value(status)' 2>/dev/null) \
        && [[ ! $s =~ ^(PROVISIONING|STAGING|RUNNING)$ ]]; then
      log "PREEMPTED vm=$name status=${s:-gone} (during startup)"
      if [[ -n $s ]]; then cmd_delete "$name" "stopped during startup ($s)"
      else log "DELETE vm=$name epoch=$(date +%s) reason=\"gone during startup (spot preemption)\" leftover_disks=\"$("${GC[@]}" compute disks list --filter="name=$name" --format='value(name)')\" spend_now=$(spend)"; fi
      exit 3
    fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_start() {
  local name=$1 model=$2 arms=$3; shift 3
  local zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job/scripts /root/job/bundle /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "${JOB_FILES[@]/#/$REPO/}" "$REPO/scripts/gpu/$REMOTE" "$SSH_USER@$ip:/root/job/scripts/" || exit 1
  # bundles: one per teacher named in the queue (distill2: ARM@TEACHER items), else $OUT/bundle
  local -a bdirs=() rdirs=(); local t i
  if [[ $JOB == distill2 ]]; then
    for t in $(printf '%s\n' $arms | sed 's/#.*//; s/.*@//' | sort -u); do
      [[ -n ${TEACHER_BUNDLE[$t]:-} ]] || { echo "unknown teacher '$t' in queue" >&2; exit 1; }
      bdirs+=("${TEACHER_BUNDLE[$t]}"); rdirs+=("/root/job/bundle-$t")
    done
  else
    bdirs=("$OUT/bundle"); rdirs=(/root/job/bundle)
  fi
  for i in "${!bdirs[@]}"; do
    rsh "$ip" "sudo mkdir -p ${rdirs[i]}" || exit 1
    rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "${BUNDLE_FILES[@]/#/${bdirs[i]}/}" "$SSH_USER@$ip:${rdirs[i]}/" || exit 1
    rsh "$ip" "sudo bash -c 'cd ${rdirs[i]} && sha256sum ${BUNDLE_FILES[*]}'" >"$STATE/$name-bundle-$i-remote.sha256"
    (cd "${bdirs[i]}" && sha256sum "${BUNDLE_FILES[@]}") | diff - "$STATE/$name-bundle-$i-remote.sha256" \
      || { echo "bundle sha256 mismatch: ${bdirs[i]}" >&2; exit 1; }
  done
  printf -v extra '%q ' "$@"
  rsh "$ip" "sudo bash -c 'NPAR=$NPAR nohup bash /root/job/scripts/$REMOTE $model \"$arms\" $extra >>/root/out/job-$model.out 2>&1 &'"
  log "START vm=$name model=$model arms=\"$arms\" npar=$NPAR flags=\"$*\""
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
ls /root/out; cat /root/out/JOB_DONE /root/out/JOB_FAILED 2>/dev/null
for f in /root/out/*-*.log; do echo "== $f"; grep -E '"(eval|train_setup|done|early_stop)"' "$f" | tail -3 | cut -c1-260; grep -E '"(train|block)"' "$f" | tail -1 | cut -c1-300; tail -1 "$f" | cut -c1-200; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
REMOTE
}

cmd_sync() {
  [[ $JOB == distill2 ]] && { cmd_gsync "$1"; return; }
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local dest="$STATE/$name"; mkdir -p "$dest"
  # snapshot first (files keep growing while a job runs); checkpoints stay on the VM
  rsh "$ip" 'sudo bash -c "rm -rf /tmp/snap && rsync -a --exclude SHA256SUMS --exclude \"ckpt.pt*\" /root/out/ /tmp/snap/ && cd /tmp/snap && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz . && chmod 644 /tmp/out.tgz"' || return 1
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$dest/out.tgz" || return 1
  tar -C "$dest" -xzf "$dest/out.tgz" && trash-put "$dest/out.tgz"
  (cd "$dest" && sha256sum -c --quiet SHA256SUMS) || { log "SYNC-VERIFY-FAILED vm=$name"; return 1; }
  log "SYNC vm=$name files=$(wc -l <"$dest/SHA256SUMS") sha256=ok dest=$dest"
}

cmd_delete() {  # delete without sync; verify gone
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "$SRC: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name=$name" --format='value(name,zone.basename())')
  log "DELETE vm=$name epoch=$(date +%s) reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  alert "$SRC: VM $name deleted ($reason); est. spend so far \$$(spend) ceiling"
}

cmd_down() {
  local name=$1 reason=${2:-down}
  cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"
  cmd_delete "$name" "$reason"
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $STATE/watchdog.stop ]]; do
    local list; list=$(vms)
    local s; s=$(spend)
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f '"[${WORKER:0:1}]${WORKER:1}"' >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
        if [[ $st == busy ]]; then unset "idle_since[$name]"; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="job $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done <<<"$list"
    sleep 60
  done
  rm -f "$STATE/watchdog.stop"
}

# ---- checkpointed groups (CDISTILL_JOB=distill2) ------------------------------------------------------------
# $STATE/GROUP/slot-0, slot-1: the last two verified syncs (hard links between them for unchanged files);
# $STATE/GROUP/current -> the newest verified one. A sync is only promoted to `current` after every file in it
# matches the SHA256SUMS written on the VM, so `current` is always a complete, consistent snapshot to resume from.
group_of() { local g=${1#"$LABEL"-}; echo "${g%%[0-9]*}"; }
item_dir() { local it=$1 tag=""; [[ $it == *#* ]] && { tag=-${it#*#}; it=${it%%#*}; }; echo "${it%@*}-t${it#*@}$tag"; }
glog() { printf -- "- %s %s\n" "$(now)" "$2" >>"$1/sync.log"; echo "$2"; }  # routine syncs: per-group log, not the ledger
all_done() {  # every queue item has result.json in the group's last verified sync
  local gdir=$1 model=$2 queue=$3 it
  for it in $queue; do [[ -f $gdir/current/$model/$(item_dir "$it")/result.json ]] || return 1; done
}
ckpt_summary() {  # item=newest checkpoint step (or done) for each item in a synced tree
  local d s o=""
  for d in "$1"/*/*/; do
    [[ -d $d ]] || continue
    if [[ -f $d/result.json ]]; then s=done
    else s=$(ls "$d/ckpt" 2>/dev/null | grep -oE '^step-[0-9]+\.json$' | sort -V | tail -1); s=${s%.json}; s=${s:-none}; fi
    o+="$(basename "$d")=$s,"
  done
  echo "items=${o%,}"
}

cmd_gsync() {  # snapshot /root/out on the VM, pull it into the other slot, verify sha256, then point `current` at it
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local gdir; gdir="$STATE/$(group_of "$name")"; mkdir -p "$gdir"
  # Checkpoints, adapters, predictions and result.json are written under a .tmp name and renamed, so a hard
  # link is a frozen copy of them; append-only files (logs, events.jsonl) are copied. Hashes are of the snapshot.
  rsh "$ip" 'sudo bash -s' <<'REMOTE' || { glog "$gdir" "SYNC-SNAPSHOT-FAILED vm=$name"; return 1; }
set -e
rm -rf /root/snap.new; mkdir -p /root/snap.new; cd /root/out
find . -type f ! -name '*.tmp' ! -path '*.tmp/*' ! -name 'JOB_*' ! -name SHA256SUMS -print0 |
while IFS= read -r -d '' f; do
  mkdir -p "/root/snap.new/$(dirname "$f")"
  case $f in
    *.log|*.out|*/events.jsonl) cp -p "$f" "/root/snap.new/$f" 2>/dev/null || true ;;
    *) ln "$f" "/root/snap.new/$f" 2>/dev/null || true ;;  # gone (rotated) since find: skip it
  esac
done
cd /root/snap.new
find . -type f -print0 | sort -z | xargs -0 -r sha256sum >/root/SHA256SUMS.snap
mv /root/SHA256SUMS.snap SHA256SUMS
rm -rf /root/snap; mv /root/snap.new /root/snap
REMOTE
  local cur="" slot; [[ -L $gdir/current ]] && cur=$(readlink "$gdir/current")
  if [[ $cur == slot-0 ]]; then slot=slot-1; else slot=slot-0; fi
  mkdir -p "$gdir/$slot"
  local -a ld=(); [[ -n $cur ]] && ld=(--link-dest="$gdir/$cur/")
  rsync -a --no-o --no-g --delete "${ld[@]}" --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" \
      "$SSH_USER@$ip:/root/snap/" "$gdir/$slot/" || { glog "$gdir" "SYNC-PULL-FAILED vm=$name"; return 1; }
  (cd "$gdir/$slot" && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1) \
    || { glog "$gdir" "SYNC-VERIFY-FAILED vm=$name slot=$slot (current stays $cur)"; return 1; }
  ln -sfn "$slot" "$gdir/current.new" && mv -T "$gdir/current.new" "$gdir/current"
  glog "$gdir" "SYNC vm=$name files=$(wc -l <"$gdir/$slot/SHA256SUMS") sha256=ok current=$gdir/$slot $(ckpt_summary "$gdir/current")"
}

cmd_restore() {  # push the group's last verified sync to a (new) VM's /root/out and check every sha256 there
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  local gdir; gdir="$STATE/$(group_of "$name")"
  [[ -f $gdir/current/SHA256SUMS ]] || { echo "nothing synced yet for group $(group_of "$name"): fresh start"; return 0; }
  (cd "$gdir/current/" && sha256sum -c --quiet SHA256SUMS) || { log "RESTORE-LOCAL-VERIFY-FAILED vm=$name"; return 1; }
  rsh "$ip" 'sudo mkdir -p /root/out' || return 1
  rsync -a --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$gdir/current/" "$SSH_USER@$ip:/root/out/" || return 1
  rsh "$ip" 'sudo bash -c "cd /root/out && sha256sum -c --quiet SHA256SUMS && rm -f SHA256SUMS"' \
    || { log "RESTORE-VERIFY-FAILED vm=$name"; return 1; }
  log "RESTORE vm=$name from=$gdir/$(readlink "$gdir/current") files=$(wc -l <"$gdir/current/SHA256SUMS") sha256=ok $(ckpt_summary "$gdir/current")"
}

cmd_supervise() {
  local group=$1 models=$2 model=$3 queue=$4; shift 4
  [[ $JOB == distill2 && $group =~ ^[a-z]+$ ]] || { echo "supervise needs CDISTILL_JOB=distill2 and a GROUP of letters" >&2; exit 2; }
  local gdir="$STATE/$group"; mkdir -p "$gdir"
  exec 8>"$gdir/supervise.lock"; flock -n 8 || { echo "a supervisor for group $group is already running"; exit 0; }
  local max_vms=${CDISTILL_MAX_VMS:-8} name="" lst live ours st ip zone n out rc last_sync=0 up_fails=0 ops v gone newvm
  log "SUPERVISE group=$group model=$model queue=\"$queue\" machine=$MACHINE npar=$NPAR sync_every=${SYNC_MIN}min flags=\"$*\" (sync log $gdir/sync.log)"
  while :; do
    # the group's VMs; a failed gcloud call must not look like a preemption, so it only retries
    if ! lst=$("${GC[@]}" compute instances list --filter="name~^$LABEL-$group[0-9]+\$" --format='value(name,status)' 2>>"$gdir/supervise.out"); then
      sleep 30; continue
    fi
    live=$(awk '$2 ~ /^(RUNNING|PROVISIONING|STAGING)$/ {print $1}' <<<"$lst" | sort -V | tail -1)
    ours=$(awk -v n="$name" '$1 == n {print $2}' <<<"$lst")
    if [[ -f $gdir/STOP ]]; then
      log "SUPERVISE-STOP group=$group (STOP file)"
      [[ -n $live ]] && { ( cmd_gsync "$live" ) || true; ( cmd_delete "$live" "group $group stopped" ); }
      break
    fi
    if [[ -n $name && $ours != RUNNING ]]; then  # our VM is gone or going: spot preemption, max run duration, watchdog
      ops=$("${GC[@]}" compute operations list --filter="targetLink~/instances/$name\$" \
              --format='value(operationType,insertTime)' 2>/dev/null | sort -k2 | tail -3 | tr '\n' ' ')
      if ! grep -q "DELETE vm=$name " "$LEDGER"; then
        log "PREEMPTED vm=$name status=${ours:-gone} ops=\"$ops\" resume_from=\"$(ckpt_summary "$gdir/current")\""
        log "DELETE vm=$name epoch=$(date +%s) reason=\"gone (spot preemption or max run duration)\" leftover_disks=\"$("${GC[@]}" compute disks list --filter="name=$name" --format='value(name)')\" spend_now=$(spend)"
        alert "$SRC: spot VM $name is gone ($ops). Re-creating the next VM of group $group; it resumes from the last sha256-verified sync."
      fi
      name=""; continue
    fi
    newvm=""
    if [[ -z $name && -n $live ]]; then  # a VM from an earlier supervisor run
      if grep -q "START vm=$live " "$LEDGER"; then name=$live; log "SUPERVISE-ADOPT vm=$live"; continue; fi
      if ! grep -q " CREATE vm=$live " "$LEDGER" || grep -q "DELETE vm=$live " "$LEDGER"; then
        ( cmd_delete "$live" "not a live VM of the ledger (supervisor restart)" ); continue
      fi
      # created by a supervisor that was restarted during the VM's startup: finish its startup here
      name=$live; log "SUPERVISE-ADOPT vm=$live (created, not started yet)"
      out=$( ( wait_ready "$name" "$(zone_of "$name")" ) 2>&1 ); rc=$?; printf '%s\n' "$out" >>"$gdir/supervise.out"
      if (( rc != 0 )); then name=""; continue; fi
      newvm=1
    fi
    if [[ -z $name ]]; then
      # a group VM that vanished while no supervisor watched it (e.g. preempted during a benchmark) still needs
      # its DELETE line, or spend() keeps counting it; the preemption time comes from the operations list
      for v in $(grep -oE "CREATE vm=$LABEL-$group[0-9]+ " "$LEDGER" | sed 's/CREATE vm=//; s/ $//' | sort -u); do
        grep -q "DELETE vm=$v " "$LEDGER" && continue
        awk -v n="$v" '$1 == n {f=1} END {exit !f}' <<<"$lst" && continue
        ops=$("${GC[@]}" compute operations list --filter="targetLink~/instances/$v\$" \
                --format='value(operationType,insertTime)' 2>/dev/null | sort -k2 | tail -3)
        gone=$(awk '$1 ~ /preempted|delete/ {t=$2} END {print t}' <<<"$ops")
        gone=$(date -d "${gone:-now}" +%s 2>/dev/null || date +%s); ops=$(tr '\n\t' '  ' <<<"$ops")
        log "PREEMPTED vm=$v status=gone (before any supervisor adopted it) ops=\"$ops\" resume_from=\"$(ckpt_summary "$gdir/current")\""
        log "DELETE vm=$v epoch=$gone reason=\"gone before adoption (spot preemption or max run duration)\" leftover_disks=\"$("${GC[@]}" compute disks list --filter="name=$v" --format='value(name)')\" spend_now=$(spend)"
      done
      if [[ -f $gdir/current/SHA256SUMS ]] && all_done "$gdir" "$model" "$queue"; then log "SUPERVISE-DONE group=$group (already complete)"; break; fi
      n=$(grep -oE "vm=$LABEL-$group[0-9]+ " "$LEDGER" | grep -oE '[0-9]+ $' | sort -n | tail -1); n=$(( ${n:-0} + 1 ))
      # failed creates use up a suffix but no VM: count only the VMs that were created
      if (( $(grep -cE " CREATE vm=$LABEL-$group[0-9]+ " "$LEDGER") >= max_vms )); then
        log "SUPERVISE-GIVEUP group=$group: $max_vms VMs used"; alert "$SRC: group $group gave up after $max_vms VMs; results so far in $gdir/current"; break
      fi
      name=$LABEL-$group$n
      out=$( ( cmd_up "$name" "$models" ) 2>&1 ); rc=$?; printf '%s\n' "$out" >>"$gdir/supervise.out"
      if (( rc == 3 )); then  # preempted during startup: try the next VM soon
        name=""; alert "$SRC: spot VM $LABEL-$group$n was preempted during startup; trying the next one"; sleep 30; continue
      fi
      if (( rc != 0 )); then
        name=""
        if grep -q REFUSE <<<"$out"; then log "SUPERVISE-STOP group=$group (spend cap)"; alert "$SRC: group $group stopped at the spend cap"; break; fi
        (( ++up_fails >= ${CDISTILL_MAX_UP_FAILS:-6} )) && { log "SUPERVISE-GIVEUP group=$group: $up_fails failed creates"; alert "$SRC: group $group could not get a VM $up_fails times"; break; }
        (( up_fails % 6 == 0 )) && alert "$SRC: group $group: no spot VM in any zone for $up_fails tries (5 min apart); still trying"
        sleep 300; continue
      fi
      up_fails=0; newvm=1
    fi
    if [[ $newvm ]]; then
      if ! ( cmd_restore "$name" ) >>"$gdir/supervise.out" 2>&1 || ! ( cmd_start "$name" "$model" "$queue" "$@" ) >>"$gdir/supervise.out" 2>&1; then
        log "SUPERVISE-START-FAILED vm=$name"; ( cmd_delete "$name" "restore or start failed" ); name=""; sleep 60; continue
      fi
      last_sync=$(date +%s); sleep 60; continue
    fi
    # our VM is running
    zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone" 2>/dev/null)
    st=$(rsh "$ip" 'sudo cat /root/out/JOB_DONE 2>/dev/null' 2>/dev/null || true)
    if [[ $st == rc=* ]]; then
      if ! ( cmd_gsync "$name" ) >>"$gdir/supervise.out" 2>&1; then sleep 30; continue; fi
      if all_done "$gdir" "$model" "$queue"; then
        log "SUPERVISE-DONE group=$group vm=$name $st"
        ( cmd_delete "$name" "group $group done, final sync verified" )
      else
        log "SUPERVISE-JOB-FAILED group=$group vm=$name $st $(rsh "$ip" 'sudo cat /root/out/JOB_FAILED 2>/dev/null' | tr '\n' ' ')"
        alert "$SRC: group $group job ended without all results ($st); VM deleted after the final sync, results in $gdir/current"
        ( cmd_delete "$name" "group $group job failed, final sync verified" )
      fi
      log "VERIFY-GONE group=$group instances=\"$(vms | awk -v p="^$LABEL-$group[0-9]+\$" '$1 ~ p {print $1}' | tr '\n' ' ')\" disks=\"$("${GC[@]}" compute disks list --filter="name~^$LABEL-$group[0-9]+\$" --format='value(name)' | tr '\n' ' ')\""
      break
    fi
    if (( $(date +%s) - last_sync >= SYNC_MIN * 60 )); then
      if ( cmd_gsync "$name" ) >>"$gdir/supervise.out" 2>&1; then last_sync=$(date +%s); else glog "$gdir" "SYNC-FAILED vm=$name (retry in 30 s)"; fi
    fi
    sleep 30
  done
}

case "${1:-}" in
  up) cmd_up "$2" "$3" ;;
  start) shift; cmd_start "$@" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  gsync) cmd_gsync "$2" ;;
  restore) cmd_restore "$2" ;;
  supervise) shift; cmd_supervise "$@" ;;
  watchdog) cmd_watchdog ;;
  spend) spend ;;
  list) vms ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
