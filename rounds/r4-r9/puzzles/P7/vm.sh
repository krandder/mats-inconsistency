#!/bin/bash
# P7 spot CPU VM (adapted copy of puzzles/P6/gpu.sh; same stop guarantees, no GPU).
#
#   vm.sh up NAME             create trader-cat-p7-* (c2d-highcpu-32 SPOT, label trader-cat=p7), wait READY
#   vm.sh start NAME QUEUE P  push code + queue file, run vm_run_queue.sh with P parallel jobs, detached
#   vm.sh status NAME         queue progress, failures, load
#   vm.sh sync NAME           tar /root/out + SHA256SUMS -> verify -> merge into results/
#   vm.sh down NAME [why]     sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   vm.sh watchdog            every 5 min: sync; delete on spend >= $CAP, age, or idle queue > 20 min
#   vm.sh reconcile           book VMs that vanished without a DELETE (spot preemption); the watchdog runs it too
#   vm.sh spend | list
# Stop guarantees: --max-run-duration=6h + --instance-termination-action=DELETE (+ boot disk auto-delete),
# the watchdog, and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x RATE, where RATE = $1.20/h is a ceiling for c2d-highcpu-32 on demand incl. disk
# (spot is much lower; the Billing API is not readable here).
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P7="$(cd "$(dirname "$0")" && pwd)"
RES="$P7/results"; STATE="$P7/vm"; mkdir -p "$STATE" "$RES"
LEDGER="$STATE/LEDGER.md"
BOOK="$P7/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=${P7_MACHINE:-c2d-highcpu-32}; RATE=1.20
ZONES=(${P7_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f})
IMAGE_PROJECT=ubuntu-os-cloud; IMAGE_FAMILY=ubuntu-2404-lts-amd64
CAP_USD=${P7_CAP_USD:-7}
MAX_RUN=${P7_MAX_RUN:-6h}
MAX_AGE_H=${P7_MAX_AGE_H:-5.9}
IDLE_MIN=20
SSH_USER=p7; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-cat-p7 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.trader-cat=p7" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }
spend() {
  awk -v now="$(date +%s)" -v rate="$RATE" '
    / CREATE / { r=rate; for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); if($i~/^rate_ceiling=/)r=substr($i,14) } start[n]=e; rt[n]=r }
    / DELETE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7) } if(!(n in stop)) stop[n]=e }
    END { t=0; for(n in start){ s=(n in stop?stop[n]:now); t+=(s-start[n])/3600*rt[n] } printf "%.2f\n", t }' "$LEDGER" 2>/dev/null || echo 0
}
book_row() {  # book_row NAME ZONE CREATED_EPOCH [DELETED_EPOCH]: add or complete the VM's row in GPU-LEDGER.md
  python3 - "$BOOK" "$@" "$RATE" "$MAX_RUN" <<'PY'
import os, sys, time
book, name, zone, c = sys.argv[1:5]
d = sys.argv[5] if len(sys.argv) > 7 else ""
rate, maxrun = sys.argv[-2], sys.argv[-1]
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## P7 hindsight-posterior settlement (trader-cat-p7-*, cap $8)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + f"\n\n{head}\nCPU only, no GPU. Rate ceiling: c2d-highcpu-32 SPOT + boot disk <= ${rate}/h (on-demand list price; spot is lower). Watchdog: puzzles/P7/vm.sh watchdog (5-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).\n| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n"
row0 = f"| {fmt(c)} | {name} | {zone} | {os.environ.get('P7_MACHINE', 'c2d-highcpu-32')} spot (CPU only), max-run {maxrun} |"
if d:
    h = (int(d) - int(c)) / 3600
    row = f"{row0} {fmt(d)} | {h:.2f} | {h * float(rate):.2f} |"
else:
    row = f"{row0} running | | |"
lines = txt.rstrip("\n").split("\n")
for i, l in enumerate(lines):
    if l.startswith(row0):
        lines[i] = row
        break
else:  # insert at the end of the P7 table, not the end of the file (other experiments append sections too)
    h = lines.index(head)
    j = h + 1
    while j < len(lines) and not lines[j].startswith("|"):
        j += 1
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    lines.insert(j, row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1
  [[ $name =~ ^trader-cat-p7-[a-z0-9-]+$ ]] || { echo "name must be trader-cat-p7-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5h > cap \$$CAP_USD" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p7-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone=""
  for z in "${ZONES[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" \
        --machine-type="$MACHINE" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=30GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="trader-cat=p7" \
        --metadata=block-project-ssh-keys=TRUE \
        --metadata-from-file="startup-script=$P7/vm_startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
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
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE spot max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  alert "P7 hindsight settlement: spot CPU VM $name created in $zone (self-deletes after $MAX_RUN; cap \$$CAP_USD)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p7/READY 2>/dev/null || { [ -f /var/lib/p7/FAILED ] && echo FAILED: $(cat /var/lib/p7/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p7-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_start() {
  local name=$1 queue=$2 par=${3:-30} zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$P7/maze.py" "$P7/prefix.py" "$P7/common.py" \
    "$P7/vm_run_queue.sh" "$SSH_USER@$ip:/root/job/" || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$queue" "$SSH_USER@$ip:/root/job/queue.txt" || exit 1
  # sudo can hold the ssh channel open after the detached job starts (seen 09-29); the job survives the timeout
  timeout 30 ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo bash -c 'cd /root/job && setsid nohup bash vm_run_queue.sh $par </dev/null >/root/out/queue.out 2>&1 &'" </dev/null
  log "START vm=$name queue=$(basename "$queue") jobs=$(wc -l <"$queue") parallel=$par"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE 2>/dev/null
echo "jobs $(wc -l </root/job/queue.txt) finished $(grep -l '^DONE' /root/out/*/*.log 2>/dev/null | wc -l) failed $(grep -l Traceback /root/out/*/*.log 2>/dev/null | wc -l) running $(pgrep -fc '[p]ython (maze|prefix).py')"
grep -l Traceback /root/out/*/*.log 2>/dev/null | head -3
uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local stage; stage=$(mktemp -d /tmp/p7-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  if ! (cd "$stage/x" && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
    # files still being written during tar can mismatch; keep only verified files
    (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' | xargs -r rm -f)
    log "SYNC-PARTIAL vm=$name (files changing during tar skipped)"
  fi
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  rsync -a --files-from="$stage/ok.list" "$stage/x/" "$RES/"
  cp "$stage/x/SHA256SUMS" "$RES/SHA256SUMS.vm"
  local n; n=$(wc -l <"$stage/ok.list")
  rm -rf "$stage"
  log "SYNC vm=$name verified_files=$n dest=$RES"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P7: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p7" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P7: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); local cz; cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  local ct; ct=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*type=\([^ ]*\).*/\1/')
  [[ -n $c ]] && P7_MACHINE=${ct:-$MACHINE} book_row "$name" "$cz" "$c" "$ep"
  alert "P7 hindsight settlement: VM $name deleted ($reason); est. spend so far \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

# A spot preemption deletes the VM without a DELETE line (seen 09-29 on trader-cat-p7-2), which left the
# ledger row "running" and spend() counting. Book any CREATEd VM that no longer exists, at its preemption time.
cmd_reconcile() {
  local live; live=$("${GC[@]}" compute instances list --filter="labels.trader-cat=p7" --format="value(name)") || return 1
  local n c d
  for n in $(awk '/ CREATE /{for(i=1;i<=NF;i++) if($i~/^vm=/) print substr($i,4)}' "$LEDGER" | sort -u); do
    c=$(grep -c " CREATE vm=$n " "$LEDGER"); d=$(grep -c " DELETE vm=$n " "$LEDGER")
    (( c > d )) || continue
    grep -qx "$n" <<<"$live" && continue
    local t ep zone ce
    t=$("${GC[@]}" compute operations list --filter="targetLink~/$n\$ AND operationType=compute.instances.preempted" --sort-by=~insertTime --limit=1 --format='value(insertTime)' 2>/dev/null)
    ep=$(date -d "${t:-now}" +%s)
    local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p7" --format='value(name,zone.basename())')
    local what="not found in instance list"; [[ -n $t ]] && what="spot preempted at $t"
    log "DELETE vm=$n epoch=$ep reason=\"gone without delete: $what\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
    ce=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); zone=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
    [[ -n $ce ]] && P7_MACHINE=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*type=\([^ ]*\).*/\1/') book_row "$n" "$zone" "$ce" "$ep"
    alert "P7: VM $n is gone ($what); booked in ledger${disks:+; leftover disks: $disks}"
  done
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $STATE/watchdog.stop ]]; do
    cmd_reconcile >/dev/null 2>&1 || true
    local list s; list=$(vms); s=$(spend)
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [v]m_run_queue.sh >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
        if [[ $st == busy ]]; then unset "idle_since[$name]"; cmd_sync "$name" >/dev/null 2>&1 || true; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done <<<"$list"
    sleep 300
  done
  rm -f "$STATE/watchdog.stop"
}

case "${1:-}" in
  up) cmd_up "$2" ;;
  start) cmd_start "$2" "$3" "${4:-30}" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  reconcile) cmd_reconcile ;;
  spend) spend ;;
  list) vms ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
