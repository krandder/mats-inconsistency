#!/bin/bash
# Batch-2 follow-ups: spot CPU VMs (adapted from puzzles/P14/vm.sh; same stop guarantees, auto-resume).
#
#   vm.sh ITEM up NAME             create trader-fu-* (c2d-highcpu-32 SPOT unless FU_MACHINE), wait READY
#   vm.sh ITEM start NAME QUEUE P  push ITEM/code + queue + synced ITEM/vmout (resume), run the queue with P slots
#   vm.sh ITEM status NAME         queue progress, failures, load
#   vm.sh ITEM sync NAME           tar /root/out + SHA256SUMS -> verify -> merge into ITEM/vmout/
#   vm.sh ITEM down NAME [why]     sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   vm.sh ITEM delete NAME [why]   delete without sync
#   vm.sh all watchdog             every 5 min, all items: reconcile (book + auto-resume preempted VMs), sync busy VMs,
#                                  delete on total spend >= $FU_KILL_USD, item spend >= item cap, age, or idle > 20 min
#   vm.sh all spend | list | reconcile
# ITEM: item1 (VM names trader-fu-c1-*), item2 (trader-fu-p10-*), item3 (trader-fu-p8-*).
# Queue lines: "NAME shell command..." run in /root/job with PY=/opt/venv/bin/python, OMP_NUM_THREADS=1; a zero exit
# writes /root/out/markers/NAME.done and a restarted queue skips it. Jobs must checkpoint (<= 10 min) and resume.
# Stop guarantees: --max-run-duration=6h + --instance-termination-action=DELETE (+ boot disk auto-delete), the
# watchdog, and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x ceiling rate (on-demand list price incl. disk; spot bills less). Task cap $20 at the ceiling.
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
FU="$(cd "$(dirname "$0")" && pwd)"
ITEM=${1:-}; shift || true
BOOK="$FU/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=${FU_MACHINE:-c2d-highcpu-32}
case $MACHINE in c2d-highcpu-16) RATE=0.60 ;; c2d-highcpu-32) RATE=1.20 ;; c2d-highcpu-56) RATE=2.10 ;; *) echo "unknown machine $MACHINE" >&2; exit 2 ;; esac
ZONES=(${FU_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east4-a us-west1-b})
IMAGE_PROJECT=ubuntu-os-cloud; IMAGE_FAMILY=ubuntu-2404-lts-amd64
TOTAL_CAP=20; KILL_USD=${FU_KILL_USD:-19}
declare -A CAP=([item1]=9 [item2]=4 [item3]=5) CODE=([item1]=c1 [item2]=p10 [item3]=p8)
MAX_RUN=${FU_MAX_RUN:-6h}; MAX_AGE_H=5.9; IDLE_MIN=20
SSH_USER=fu; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
idir() { echo "$FU/$1"; }
ledger() { echo "$FU/$1/vm/LEDGER.md"; }
log() { local it=$1; shift; mkdir -p "$FU/$it/vm"; printf -- '- %s %s\n' "$(now)" "$*" >>"$(ledger "$it")"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-fu "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.trader-fu:*" --format="value(name,zone.basename(),creationTimestamp,status,labels.trader-fu)" 2>/dev/null; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }
item_spend() {
  local f; f=$(ledger "$1"); [[ -f $f ]] || { echo 0.00; return; }
  awk -v now="$(date +%s)" '
    / CREATE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); if($i~/^rate_ceiling=/)r=substr($i,14) } start[n]=e; rt[n]=r }
    / DELETE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7) } if(!(n in stop)) stop[n]=e }
    END { t=0; for(n in start){ s=(n in stop?stop[n]:now); t+=(s-start[n])/3600*rt[n] } printf "%.2f\n", t }' "$f"
}
total_spend() { local t=0 it; for it in item1 item2 item3; do t=$(awk -v a="$t" -v b="$(item_spend $it)" 'BEGIN{printf "%.2f", a+b}'); done; echo "$t"; }
book_row() {  # book_row NAME ZONE CREATED_EPOCH [DELETED_EPOCH]; rate follows the VM's own MACHINE (fixed 00:20Z: was the default -32 rate)
  local rate; case $MACHINE in c2d-highcpu-16) rate=0.60 ;; c2d-highcpu-32) rate=1.20 ;; c2d-highcpu-56) rate=2.10 ;; *) rate=$RATE ;; esac
  python3 - "$BOOK" "$@" "$rate" "$MAX_RUN" "$MACHINE" <<'PY'
import sys, time
book, name, zone, c = sys.argv[1:5]
d = sys.argv[5] if len(sys.argv) > 8 else ""
rate, maxrun, mach = sys.argv[-3], sys.argv[-2], sys.argv[-1]
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## Batch-2 follow-ups: CostNet+BB seeds, P10 init check, P8 N=4096 (trader-fu-*, cap $20)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + (f"\n\n{head}\nCPU only (c2d-highcpu spot, termination DELETE, max-run 6h). Rate ceiling = on-demand list price incl. "
        "boot disk (c2d-highcpu-32 $1.20/h, -16 $0.60/h). Watchdog: puzzles/followups-batch2/vm.sh all watchdog (5-min sync + sha256, "
        "auto-resume after preemption, deletes at est. total >= $19, item cap, age 5.9 h or idle 20 min). Logs: followups-batch2/item*/vm/LEDGER.md.\n"
        "| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n")
row0 = f"| {fmt(c)} | {name} | {zone} | {mach} spot (CPU only), max-run {maxrun} |"
row = f"{row0} {fmt(d)} | {(int(d)-int(c))/3600:.2f} | {(int(d)-int(c))/3600*float(rate):.2f} |" if d else f"{row0} running | | |"
lines = txt.rstrip("\n").split("\n")
for i, l in enumerate(lines):
    if l.startswith(row0):
        lines[i] = row; break
else:
    h = lines.index(head); j = h + 1
    while j < len(lines) and not lines[j].startswith("|"): j += 1
    while j < len(lines) and lines[j].startswith("|"): j += 1
    lines.insert(j, row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1 code=${CODE[$ITEM]}
  [[ $name =~ ^trader-fu-$code-[a-z0-9-]+$ ]] || { echo "name must be trader-fu-$code-*" >&2; exit 2; }
  local s t; s=$(item_spend "$ITEM"); t=$(total_spend)
  if awk -v s="$s" -v t="$t" -v r="$RATE" -v c="${CAP[$ITEM]}" -v T="$TOTAL_CAP" 'BEGIN{exit !(s+r*0.5>c || t+r*0.5>T)}'; then
    echo "REFUSE: item spend \$$s / total \$$t + 0.5 h would pass the cap" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/fu-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone=""
  for z in "${ZONES[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" --machine-type="$MACHINE" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=30GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="trader-fu=$ITEM" --metadata=block-project-ssh-keys=TRUE \
        --metadata-from-file="startup-script=$FU/vm_startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "$ITEM" "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
    "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q . && cmd_delete "$name" "create failed"
    rm -rf "$tmp"; exit 1
  fi
  rm -rf "$tmp"
  local ep; ep=$(date +%s)
  log "$ITEM" "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE spot max_run=$MAX_RUN rate_ceiling=$RATE item_spend_before=$s total_before=$t"
  book_row "$name" "$zone" "$ep"
  alert "trader-fu ($ITEM): spot CPU VM $name created in $zone (self-deletes after $MAX_RUN; task cap \$$TOTAL_CAP)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/fu/READY 2>/dev/null || { [ -f /var/lib/fu/FAILED ] && echo FAILED: $(cat /var/lib/fu/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/fu-startup.log' 2>/dev/null; log "$ITEM" "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "$ITEM" "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "$ITEM" "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_start() {
  local name=$1 queue=$2 par=${3:-30} zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  local D; D=$(idir "$ITEM")
  rsh "$ip" 'sudo mkdir -p /root/job /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$D/code/" "$FU/vm_run_queue.sh" "$SSH_USER@$ip:/root/job/" || exit 1
  local qn; qn="q_$(basename "$queue")"
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$queue" "$SSH_USER@$ip:/root/job/$qn" || exit 1
  # resume: everything from the last verified sync goes back (done markers, finished outputs, checkpoints)
  [[ -d $D/vmout ]] && rsync -az --rsync-path="sudo rsync" --exclude=JOB_RUNNING --exclude='RUNNER.*' --exclude=JOB_DONE \
      -e "ssh ${SSH_OPTS[*]}" "$D/vmout/" "$SSH_USER@$ip:/root/out/"
  printf '%s %s\n' "$queue" "$par" >"$D/vm/resume.$name"
  timeout 30 ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo bash -c 'cd /root/job && setsid nohup bash vm_run_queue.sh $par $qn </dev/null >>/root/out/queue.out 2>&1 &'" </dev/null
  log "$ITEM" "START vm=$name queue=$(basename "$queue") jobs=$(grep -c . "$queue") parallel=$par"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE 2>/dev/null
q=$(readlink -f /root/job/queue.txt 2>/dev/null)
echo "jobs $(ls /root/job/q_* 2>/dev/null | xargs cat | grep -c .) done $(ls /root/out/markers/*.done 2>/dev/null | wc -l) failed $(ls /root/out/markers/*.failed 2>/dev/null | wc -l) running $(pgrep -fc '[r]un_one_fu')"
ls /root/out/markers/*.failed 2>/dev/null | head -5
uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local D; D=$(idir "$ITEM"); mkdir -p "$D/vmout"
  local stage; stage=$(mktemp -d /tmp/fu-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  local bad; bad=$(cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -vc ': OK$')
  rsync -a --files-from="$stage/ok.list" "$stage/x/" "$D/vmout/"
  cp "$stage/x/SHA256SUMS" "$D/vmout/SHA256SUMS.vm.$name"
  local n; n=$(wc -l <"$stage/ok.list")
  rm -rf "$stage"
  log "$ITEM" "SYNC vm=$name verified_files=$n sha256_mismatch_skipped=$bad dest=$D/vmout"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "$ITEM" "DELETE-FAILED vm=$name reason=\"$reason\""; alert "trader-fu: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-fu" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "trader-fu: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "$ITEM" "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" total_spend_now=$(total_spend)"
  local L; L=$(ledger "$ITEM")
  local c cz ct; c=$(grep " CREATE vm=$name " "$L" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$L" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  ct=$(grep " CREATE vm=$name " "$L" | tail -1 | sed 's/.*type=\([^ ]*\).*/\1/')
  [[ -n $c ]] && MACHINE=${ct:-$MACHINE} book_row "$name" "$cz" "$c" "$ep"
  alert "trader-fu ($ITEM): VM $name deleted ($reason); est. total spend \$$(total_spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "$ITEM" "SYNC-FAILED vm=$name (deleting anyway)"; rm -f "$(idir "$ITEM")/vm/resume.$name"; cmd_delete "$name" "$reason"; }

cmd_reconcile() {  # book CREATEd VMs that vanished (spot preemption / max-run); auto-resume if resume.NAME exists
  local live; live=$("${GC[@]}" compute instances list --filter="labels.trader-fu:*" --format="value(name)") || return 1
  local it n c d L
  for it in item1 item2 item3; do
    L=$(ledger "$it"); [[ -f $L ]] || continue
    ITEM=$it
    for n in $(awk '/ CREATE /{for(i=1;i<=NF;i++) if($i~/^vm=/) print substr($i,4)}' "$L" | sort -u); do
      c=$(grep -c " CREATE vm=$n " "$L"); d=$(grep -c " DELETE vm=$n " "$L")
      (( c > d )) || continue
      grep -qx "$n" <<<"$live" && continue
      local t ep what="not found in instance list"
      t=$("${GC[@]}" compute operations list --filter="targetLink~/$n\$ AND operationType=compute.instances.preempted" --sort-by=~insertTime --limit=1 --format='value(insertTime)' 2>/dev/null)
      [[ -n $t ]] && what="spot preempted at $t"
      ep=$(date -d "${t:-now}" +%s)
      local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-fu" --format='value(name,zone.basename())')
      log "$it" "DELETE vm=$n epoch=$ep reason=\"gone without delete: $what\" leftover_disks=\"${disks:-none}\" total_spend_now=$(total_spend)"
      local ce zone ct; ce=$(grep " CREATE vm=$n " "$L" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); zone=$(grep " CREATE vm=$n " "$L" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
      ct=$(grep " CREATE vm=$n " "$L" | tail -1 | sed 's/.*type=\([^ ]*\).*/\1/')
      [[ -n $ce ]] && MACHINE=$ct book_row "$n" "$zone" "$ce" "$ep"
      alert "trader-fu: VM $n is gone ($what); booked${disks:+; leftover disks: $disks}"
      local R; R="$(idir "$it")/vm/resume.$n"
      if [[ -f $R ]]; then
        local q par base k new; read -r q par <"$R"; rm -f "$R"
        base=${n%-r[0-9]*}; k=1; while grep -q " CREATE vm=$base-r$k " "$L"; do k=$((k+1)); done; new="$base-r$k"
        log "$it" "AUTORESUME from=$n to=$new queue=$(basename "$q")"
        ( FU_MACHINE=$ct "$0" "$it" up "$new" && FU_MACHINE=$ct "$0" "$it" start "$new" "$q" "$par" ) >>"$(idir "$it")/vm/autoresume.log" 2>&1 || log "$it" "AUTORESUME-FAILED to=$new"
      fi
    done
  done
}

cmd_watchdog() {
  exec 9>"$FU/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $FU/watchdog.stop ]]; do
    cmd_reconcile >/dev/null 2>&1 || true
    local list t; list=$(vms); t=$(total_spend)
    while read -r -u 3 name zone created status it; do   # fd 3 (fixed 00:20Z): ssh in the body ate stdin, so only the first VM per cycle was checked
      [[ -n $name ]] || continue
      ITEM=$it
      local why="" age s; age=$(age_h "$created"); s=$(item_spend "$it")
      awk -v s="$t" -v c="$KILL_USD" 'BEGIN{exit !(s>=c)}' && why="total spend \$$t >= \$$KILL_USD"
      awk -v s="$s" -v c="${CAP[$it]}" 'BEGIN{exit !(s>=c)}' && why="${why:+$why; }item spend \$$s >= cap \$${CAP[$it]}"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$(ledger "$it")"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [v]m_run_queue.sh >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
        if [[ $st == busy ]]; then unset "idle_since[$name]"; cmd_sync "$name" >/dev/null 2>&1 || true; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "$it" "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == total* || $why == item* ]]; then rm -f "$(idir "$it")/vm/resume.$name"; cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done 3<<<"$list"
    sleep 300
  done
  rm -f "$FU/watchdog.stop"
}

cmd=${1:-}; shift || true
if [[ $ITEM == all ]]; then
  case $cmd in
    watchdog) cmd_watchdog ;;
    reconcile) cmd_reconcile ;;
    spend) for it in item1 item2 item3; do echo "$it $(item_spend $it) cap ${CAP[$it]}"; done; echo "total $(total_spend) cap $TOTAL_CAP" ;;
    list) vms ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
  esac
  exit
fi
[[ -n ${CODE[$ITEM]:-} ]] || { sed -n '2,20p' "$0"; exit 2; }
case $cmd in
  up) cmd_up "$1" ;;
  start) cmd_start "$1" "$2" "${3:-30}" ;;
  status) cmd_status "$1" ;;
  sync) cmd_sync "$1" ;;
  down) cmd_down "$1" "${2:-down}" ;;
  delete) cmd_delete "$1" "${2:-manual}" ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
