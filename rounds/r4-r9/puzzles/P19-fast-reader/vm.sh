#!/bin/bash
# P19 spot CPU VM (adapted from puzzles/followups-batch2/vm.sh; same stop guarantees, single item).
#
#   vm.sh up NAME              create trader-p19-* (c2d-highcpu-32 SPOT unless P19_MACHINE), wait READY
#   vm.sh start NAME QUEUE P   push code/ + queue + synced results/vmout (resume), run the queue with P slots
#   vm.sh status NAME          queue progress, failures, load
#   vm.sh sync NAME            tar /root/out + SHA256SUMS -> verify -> merge into results/vmout/
#   vm.sh down NAME [why]      sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   vm.sh delete NAME [why]    delete without sync
#   vm.sh watchdog             every 5 min: book vanished VMs, sync busy VMs, delete on spend >= cap, age, idle > 20 min
#   vm.sh spend | list
# Stop guarantees: --max-run-duration=4h + --instance-termination-action=DELETE (+ boot disk auto-delete), the
# watchdog, and `down` after the run with an instances/disks listing check.
# Spend = VM-hours x ceiling rate (on-demand list price incl. disk; spot bills less). P19 cap $5 at the ceiling.
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
FU="$(cd "$(dirname "$0")" && pwd)"
BOOK="$FU/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=${P19_MACHINE:-c2d-highcpu-32}
case $MACHINE in c2d-highcpu-16) RATE=0.60 ;; c2d-highcpu-32) RATE=1.20 ;; *) echo "unknown machine $MACHINE" >&2; exit 2 ;; esac
ZONES=(${P19_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east4-a us-west1-b})
IMAGE_PROJECT=ubuntu-os-cloud; IMAGE_FAMILY=ubuntu-2404-lts-amd64
CAP=5; KILL_USD=4.5
MAX_RUN=${P19_MAX_RUN:-4h}; MAX_AGE_H=3.9; IDLE_MIN=20
# P19_PROV=STANDARD books the same ceiling rate (the ceiling IS the on-demand list price) and cannot be preempted
PROV=${P19_PROV:-SPOT}; case $PROV in SPOT) PROVL=spot ;; STANDARD) PROVL=on-demand ;; *) echo "bad P19_PROV" >&2; exit 2 ;; esac
SSH_USER=fu; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)
LEDGER="$FU/vm/LEDGER.md"; OUTD="$FU/results/vmout"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { mkdir -p "$FU/vm"; printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-p19 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.trader-p19:*" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }
spend() {
  [[ -f $LEDGER ]] || { echo 0.00; return; }
  awk -v now="$(date +%s)" '
    / CREATE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); if($i~/^rate_ceiling=/)r=substr($i,14) } start[n]=e; rt[n]=r }
    / DELETE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7) } if(!(n in stop)) stop[n]=e }
    END { t=0; for(n in start){ s=(n in stop?stop[n]:now); t+=(s-start[n])/3600*rt[n] } printf "%.2f\n", t }' "$LEDGER"
}
book_row() {  # book_row NAME ZONE CREATED_EPOCH [DELETED_EPOCH]
  python3 - "$BOOK" "$@" "$RATE" "$MAX_RUN" "$MACHINE $PROVL" <<'PY'
import sys, time
book, name, zone, c = sys.argv[1:5]
d = sys.argv[5] if len(sys.argv) > 8 else ""
rate, maxrun, mach = sys.argv[-3], sys.argv[-2], sys.argv[-1]
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## P19 fast-reader pricing (trader-p19-*, cap $5)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + (f"\n\n{head}\nCPU only (c2d-highcpu spot, termination DELETE, max-run 4h). Rate ceiling = on-demand list price incl. "
        "boot disk (c2d-highcpu-32 $1.20/h). Watchdog: puzzles/P19-fast-reader/vm.sh watchdog (5-min sync + sha256, deletes at "
        "est. spend >= $4.50, age 3.9 h or idle 20 min). Log: P19-fast-reader/vm/LEDGER.md.\n"
        "| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n")
row0 = f"| {fmt(c)} | {name} | {zone} | {mach} (CPU only), max-run {maxrun} |"
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
  local name=$1
  [[ $name =~ ^trader-p19-[a-z0-9-]+$ ]] || { echo "name must be trader-p19-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5 h would pass the cap" >&2; exit 1; fi
  local tmp; tmp=$(mktemp -d /tmp/p19-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone=""
  for z in "${ZONES[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" --machine-type="$MACHINE" \
        --provisioning-model="$PROV" --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=30GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="trader-p19=p19" --metadata=block-project-ssh-keys=TRUE \
        --metadata-from-file="startup-script=$FU/vm/vm_startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
    "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q . && cmd_delete "$name" "create failed"
    rm -rf "$tmp"; exit 1
  fi
  rm -rf "$tmp"
  local ep; ep=$(date +%s)
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE prov=$PROVL max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  alert "trader-p19: $PROVL CPU VM $name created in $zone (self-deletes after $MAX_RUN; cap \$$CAP)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p19/READY 2>/dev/null || { [ -f /var/lib/p19/FAILED ] && echo FAILED: $(cat /var/lib/p19/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p19-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_start() {
  local name=$1 queue=$2 par=${3:-30} zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" --exclude=__pycache__ -e "ssh ${SSH_OPTS[*]}" "$FU/code/" "$FU/vm/vm_run_queue.sh" "$SSH_USER@$ip:/root/job/" || exit 1
  local qn; qn="q_$(basename "$queue")"
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$queue" "$SSH_USER@$ip:/root/job/$qn" || exit 1
  [[ -d $OUTD ]] && rsync -az --rsync-path="sudo rsync" --exclude=JOB_RUNNING --exclude='RUNNER.*' --exclude=JOB_DONE \
      -e "ssh ${SSH_OPTS[*]}" "$OUTD/" "$SSH_USER@$ip:/root/out/"
  timeout 30 ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo bash -c 'cd /root/job && setsid nohup bash vm_run_queue.sh $par $qn </dev/null >>/root/out/queue.out 2>&1 &'" </dev/null
  log "START vm=$name queue=$(basename "$queue") jobs=$(grep -c . "$queue") parallel=$par"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE 2>/dev/null
echo "jobs $(ls /root/job/q_* 2>/dev/null | xargs cat | grep -c .) done $(ls /root/out/markers/*.done 2>/dev/null | wc -l) failed $(ls /root/out/markers/*.failed 2>/dev/null | wc -l) running $(pgrep -fc '[p]19.py')"
ls /root/out/markers/*.failed 2>/dev/null | head -5
uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  mkdir -p "$OUTD"
  local stage; stage=$(mktemp -d /tmp/p19-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  local bad; bad=$(cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -vc ': OK$')
  rsync -a --files-from="$stage/ok.list" "$stage/x/" "$OUTD/"
  cp "$stage/x/SHA256SUMS" "$OUTD/SHA256SUMS.vm.$name"
  local n; n=$(wc -l <"$stage/ok.list")
  rm -rf "$stage"
  log "SYNC vm=$name verified_files=$n sha256_mismatch_skipped=$bad dest=$OUTD"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "trader-p19: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-p19" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "trader-p19: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c cz; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book_row "$name" "$cz" "$c" "$ep"
  alert "trader-p19: VM $name deleted ($reason); est. spend \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

cmd_reconcile() {  # book CREATEd VMs that vanished (spot preemption / max-run)
  local live; live=$("${GC[@]}" compute instances list --filter="labels.trader-p19:*" --format="value(name)") || return 1
  [[ -f $LEDGER ]] || return 0
  local n c d
  for n in $(awk '/ CREATE /{for(i=1;i<=NF;i++) if($i~/^vm=/) print substr($i,4)}' "$LEDGER" | sort -u); do
    c=$(grep -c " CREATE vm=$n " "$LEDGER"); d=$(grep -c " DELETE vm=$n " "$LEDGER")
    (( c > d )) || continue
    grep -qx "$n" <<<"$live" && continue
    local ep; ep=$(date +%s)
    log "DELETE vm=$n epoch=$ep reason=\"gone without delete (preempted or max-run)\" spend_now=$(spend)"
    local ce zone; ce=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); zone=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
    [[ -n $ce ]] && book_row "$n" "$zone" "$ce" "$ep"
    alert "trader-p19: VM $n is gone without a delete (preempted?); booked. Resume by hand: vm.sh up + start."
  done
}

cmd_watchdog() {
  exec 9>"$FU/vm/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $FU/vm/watchdog.stop ]]; do
    cmd_reconcile >/dev/null 2>&1 || true
    local list t; list=$(vms); t=$(spend)
    while read -r -u 3 name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$t" -v c="$KILL_USD" 'BEGIN{exit !(s>=c)}' && why="spend \$$t >= \$$KILL_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [v]m_run_queue.sh >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null </dev/null || echo unreachable)
        if [[ $st == busy ]]; then unset "idle_since[$name]"; cmd_sync "$name" >/dev/null 2>&1 </dev/null || true; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why" </dev/null; else cmd_down "$name" "watchdog: $why" </dev/null; fi
      fi
    done 3<<<"$list"
    sleep 300
  done
  rm -f "$FU/vm/watchdog.stop"
}

cmd=${1:-}; shift || true
case $cmd in
  up) cmd_up "$1" ;;
  start) cmd_start "$1" "$2" "${3:-30}" ;;
  status) cmd_status "$1" ;;
  sync) cmd_sync "$1" ;;
  down) cmd_down "$1" "${2:-down}" ;;
  delete) cmd_delete "$1" "${2:-manual}" ;;
  watchdog) cmd_watchdog ;;
  reconcile) cmd_reconcile ;;
  spend) echo "spend $(spend) cap $CAP" ;;
  list) vms ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
