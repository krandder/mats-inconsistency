#!/bin/bash
# P5 spot L4 VM (copied from puzzles/P6/gpu.sh; pattern: consistency-ft distill_gpu.sh + market-resnet/watchdog.sh).
#
#   gpu.sh up NAME            create trader-cat-p5-* (g2-standard-4 + 1x L4 SPOT, label trader-cat=p5), wait READY
#   gpu.sh start NAME PLAN    push code + data + current results (resume), run driver.py PLAN detached
#   gpu.sh status NAME        queue state, last log lines, GPU use
#   gpu.sh sync NAME          tar /root/out + SHA256SUMS -> verify -> merge into results/
#   gpu.sh down NAME [why]    sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   gpu.sh watchdog           every 10 min: sync (sha256); delete on est. spend >= $7, age, or idle driver > 20 min
#   gpu.sh spend | list
# Stop guarantees: --max-run-duration=6h + --instance-termination-action=DELETE (+ boot disk auto-delete),
# the watchdog, and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x RATE, RATE = $0.40/h (spot g2-standard-4 + L4 + 50 GB disk estimate, per the P5 brief;
# the Billing API is not readable here). Job cap $8; the watchdog deletes at est. $7.
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P5="$(cd "$(dirname "$0")" && pwd)"
RES="$P5/results/vm"; STATE="$P5/gpu"; mkdir -p "$STATE" "$RES"
LEDGER="$STATE/LEDGER.md"
BOOK="$P5/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=g2-standard-4; ACCEL=nvidia-l4; RATE=0.40
ZONES=(${P5_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f})
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
CAP_USD=${P5_CAP_USD:-7}
MAX_RUN=${P5_MAX_RUN:-6h}
MAX_AGE_H=${P5_MAX_AGE_H:-5.9}
IDLE_MIN=20
SSH_USER=p5; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-cat-p5 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.trader-cat=p5" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
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
import sys, time
book, name, zone, c = sys.argv[1:5]
d = sys.argv[5] if len(sys.argv) > 7 else ""
rate, maxrun = sys.argv[-2], sys.argv[-1]
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## P5 private slots (trader-cat-p5-*, cap $8)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + f"\n\n{head}\nRate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= ${rate}/h (on-demand list price; spot is lower). Watchdog: puzzles/P5/gpu.sh watchdog (10-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).\n| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n"
row0 = f"| {fmt(c)} | {name} | {zone} | g2-standard-4 + L4 spot, max-run {maxrun} |"
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
else:
    lines.append(row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1
  [[ $name =~ ^trader-cat-p5-[a-z0-9-]+$ ]] || { echo "name must be trader-cat-p5-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5h > cap \$$CAP_USD" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p5-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone=""
  for z in "${ZONES[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" \
        --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="trader-cat=p5" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --metadata-from-file="startup-script=$P5/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
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
  local ep; ep=$(date +%s)
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  alert "P5 private slots: spot L4 VM $name created in $zone (self-deletes after $MAX_RUN; cap \$$CAP_USD)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p5/READY 2>/dev/null || { [ -f /var/lib/p5/FAILED ] && echo FAILED: $(cat /var/lib/p5/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p5-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_start() {
  local name=$1 plan=$2 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job/data /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$P5/mazes.py" "$P5/p5.py" "$P5/evaluate.py" "$P5/driver.py" "$P5/flops.py" "$P5/tests.py" \
    "$SSH_USER@$ip:/root/job/" || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$P5/data/" "$SSH_USER@$ip:/root/job/data/" || exit 1
  (cd "$P5/data" && sha256sum *.npz meta.json) >"$STATE/data.sha256"
  rsh "$ip" "sudo bash -c 'cd /root/job/data && sha256sum *.npz meta.json'" | diff -q - "$STATE/data.sha256" || { echo "data sha256 mismatch" >&2; exit 1; }
  # resume: push whatever results exist locally (checkpoints, logs, evals, state)
  mkdir -p "$RES"
  rsync -az --rsync-path="sudo rsync" --exclude 'SHA256SUMS*' -e "ssh ${SSH_OPTS[*]}" "$RES/" "$SSH_USER@$ip:/root/out/" || exit 1
  rsh "$ip" "sudo bash -c 'cd /root/job && setsid nohup /opt/venv/bin/python driver.py $plan </dev/null >>/root/out/driver-$plan.out 2>&1 &'" </dev/null
  log "START vm=$name plan=$plan"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE /root/out/JOB_FAILED 2>/dev/null
tail -8 /root/out/logs/driver.log 2>/dev/null
for f in $(ls -t /root/out/logs/*.log 2>/dev/null | grep -v driver.log | head -4); do echo "== $f"; tail -1 "$f" | cut -c1-300; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local stage; stage=$(mktemp -d /tmp/p5-sync.XXXXXX)
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
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P5: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p5" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P5: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); local cz; cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book_row "$name" "$cz" "$c" "$ep"
  alert "P5 private slots: VM $name deleted ($reason); est. spend so far \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $STATE/watchdog.stop ]]; do
    local list s; list=$(vms); s=$(spend)
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [d]river.py >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
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
    sleep 600
  done
  rm -f "$STATE/watchdog.stop"
}

case "${1:-}" in
  up) cmd_up "$2" ;;
  start) cmd_start "$2" "$3" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  spend) spend ;;
  list) vms ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
