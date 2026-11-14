#!/bin/bash
# P0 spot L4 VMs (pattern: consistency-ft scripts/gpu/distill_gpu.sh + market-resnet/watchdog.sh + puzzles/P1/gpu.sh).
#
#   p0_gpu.sh up NAME [ZONE0]     create trader-cat-p0-* (g2-standard-4 + 1x L4 SPOT, label tradercatp0=1), wait READY
#   p0_gpu.sh start NAME PLAN [CUTS]  push code/data/lane results, run vm/run_lane.py PLAN detached (resume-safe)
#   p0_gpu.sh status NAME          queue state, running jobs, last log lines, GPU use
#   p0_gpu.sh sync NAME            tar /root/out + SHA256SUMS -> sha256sum -c -> merge into results/vm/<lane>/
#   p0_gpu.sh down NAME [why]      sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   p0_gpu.sh watchdog             every 5 min: sync when a job finishes and every 15 min; delete all VMs at
#                                  spend >= $CAP; down at age > MAX_AGE_H or idle > IDLE_MIN; down a lane whose
#                                  main plan finished; recreate a preempted lane (next zone, <= 2 per lane)
#   p0_gpu.sh spend | list | lanes
# A lane is the VM name without its -rN recreate suffix (trader-cat-p0-a, -a-r1, -a-r2 share results/vm/trader-cat-p0-a).
# Stop guarantees: --max-run-duration=5h + --instance-termination-action=DELETE (+ boot disk auto-delete), the watchdog,
# and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x RATE, RATE = $0.72/h ceiling for g2-standard-4 + L4 + 100 GB disk (spot is lower; the Billing API
# is not readable here). Deleted VMs stay in the ledger, so spend never goes down.
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P0="$(cd "$(dirname "$0")" && pwd)"
MR="$P0/../../market-resnet"
RES="$P0/results/vm"; STATE="$P0/gpu"; mkdir -p "$STATE" "$RES"
LEDGER="$STATE/LEDGER.md"
BOOK="$P0/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=g2-standard-4; ACCEL=nvidia-l4; RATE=0.72
ZONES=(us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c)
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
CAP_USD=7.5          # hard: watchdog deletes every VM at this estimated spend (experiment cap $8)
MAX_RUN=5h
MAX_AGE_H=5.2
IDLE_MIN=20
MAX_RECREATE=2
SSH_USER=p0; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-cat-p0 "$*" >/dev/null 2>&1 || true; }
# space-separated; exits non-zero when gcloud fails so callers never mistake an API error for "no VMs"
vms() { local o; o=$("${GC[@]}" compute instances list --filter="name~^trader-cat-p0-" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null) || return 1; [[ -z $o ]] || tr '\t' ' ' <<<"$o"; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
lane_of() { echo "$1" | sed -E 's/-r[0-9]+$//'; }
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
head = "## P0 puzzle baselines (trader-cat-p0-*, cap $8)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + f"\n\n{head}\nRate ceiling: g2-standard-4 + 1x L4 SPOT + 100 GB boot disk <= ${rate}/h (ceiling; spot is lower). Watchdog: puzzles/P0/p0_gpu.sh watchdog (every 5 min; sync on each finished job and every 15 min + sha256 -c; deletes every P0 VM at est. spend >= $7.5; age 5.2 h; idle 20 min; <= 2 preemption recreates per lane).\n| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n"
row0 = f"| {fmt(c)} | {name} | {zone} | g2-standard-4 + L4 spot, max-run {maxrun} |"
row = f"{row0} {fmt(d)} | {(int(d) - int(c)) / 3600:.2f} | {(int(d) - int(c)) / 3600 * float(rate):.2f} |" if d else f"{row0} running | | |"
lines = txt.rstrip("\n").split("\n")
hi = lines.index(head)
for i in range(hi, len(lines)):
    if lines[i].startswith(row0):
        lines[i] = row
        break
else:
    j = hi + 1
    while j < len(lines) and not lines[j].startswith("## "):
        j += 1
    while j > hi and not lines[j - 1].strip():
        j -= 1
    lines.insert(j, row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1 z0=${2:-}
  [[ $name =~ ^trader-cat-p0-[a-z0-9-]+$ ]] || { echo "name must match ^trader-cat-p0-[a-z0-9-]+\$" >&2; return 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then log "REFUSE-CREATE vm=$name spend=$s"; echo "REFUSE: spend \$$s + 0.5h > \$$CAP_USD" >&2; return 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p0-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zs=("${ZONES[@]}")
  if [[ -n $z0 ]]; then  # rotate so the list starts at z0
    local i; for i in "${!ZONES[@]}"; do [[ ${ZONES[$i]} == "$z0" ]] && zs=("${ZONES[@]:$i}" "${ZONES[@]:0:$i}"); done
  fi
  local zone=""
  for z in "${zs[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" \
        --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=100GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="tradercatp0=1" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --metadata-from-file="startup-script=$P0/vm/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
    "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q . && cmd_delete "$name" "create failed"
    rm -rf "$tmp"; return 1
  fi
  rm -rf "$tmp"
  local ep; ep=$(date +%s)
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  alert "P0 puzzle baselines: spot L4 VM $name created in $zone (self-deletes after $MAX_RUN; watchdog cap \$$CAP_USD of \$8)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p0/READY 2>/dev/null || { [ -f /var/lib/p0/FAILED ] && echo FAILED: $(cat /var/lib/p0/FAILED); }' </dev/null 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p0-startup.log' </dev/null 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; return 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; return 1
}

cmd_push() {
  local name=$1 zone ip lane; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone"); lane=$(lane_of "$name")
  local R=(rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}")
  rsh "$ip" 'sudo mkdir -p /root/job/vm /root/job/src /root/job/frozen /root/out /root/data/pmr' </dev/null || return 1
  "${R[@]}" "$P0"/*.py "$MR/mresnet.py" "$SSH_USER@$ip:/root/job/" </dev/null || return 1
  "${R[@]}" "$MR/train.py" "$SSH_USER@$ip:/root/job/mr_train.py" </dev/null || return 1
  "${R[@]}" "$P0"/vm/*.py "$SSH_USER@$ip:/root/job/vm/" </dev/null || return 1
  "${R[@]}" "$P0/frozen/" "$SSH_USER@$ip:/root/job/frozen/" </dev/null || return 1
  "${R[@]}" "$P0/src/dt_net.pth" "$SSH_USER@$ip:/root/job/src/" </dev/null || return 1
  # Knutson's DTNetOriginal (mines-opt-ml/maze-extrapolation) for the unit-test reproduction check
  rsh "$ip" 'sudo mkdir -p /root/job/knutson/src' </dev/null
  "${R[@]}" /tmp/p0src/maze-extrapolation/src/models "$SSH_USER@$ip:/root/job/knutson/src/" </dev/null || return 1
  "${R[@]}" "$MR"/results/pilot/pilot_{A,B,Bds,C,D}.pt "$SSH_USER@$ip:/root/data/" </dev/null || return 1
  if [[ -d $P0/pmr_in ]]; then "${R[@]}" "$P0/pmr_in/" "$SSH_USER@$ip:/root/data/pmr/" </dev/null || return 1; fi
  # resume: push whatever this lane already synced back (markers, logs, .partial, checkpoints, npz)
  if [[ -d $RES/$lane ]]; then
    "${R[@]}" --exclude 'SHA256SUMS*' "$RES/$lane/" "$SSH_USER@$ip:/root/out/" </dev/null || return 1
  fi
  log "PUSH vm=$name lane=$lane"
}

cmd_start() {
  local name=$1 plan=$2 cuts=${3:-} zone ip lane; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone"); lane=$(lane_of "$name")
  cmd_push "$name" || return 1
  # systemd-run detaches cleanly (a backgrounded job under sudo bash -c kept the ssh session open)
  timeout 120 ssh -n "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo rm -f /root/out/JOB_DONE; sudo systemd-run --unit=p0-lane-\$(date +%s) --working-directory=/root/job --setenv=P0_CUTS=$cuts /bin/bash -c '/opt/venv/bin/python vm/run_lane.py $plan >>/root/out/run_lane.out 2>&1'"
  printf '%s %s %s\n' "$plan" "${cuts:--}" "$name" >"$STATE/lane.$lane"
  log "START vm=$name lane=$lane plan=$plan cuts=${cuts:-none}"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/queue.json 2>/dev/null; echo; ls /root/out/JOB_* 2>/dev/null
tail -3 /root/out/run_lane.out 2>/dev/null
for f in $(ls -t /root/out/logs/*.log 2>/dev/null | head -4); do echo "== $f"; tail -2 "$f" | cut -c1-260; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime; free -g | head -2
REMOTE
}

cmd_sync() {
  local name=$1 zone ip lane; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone"); lane=$(lane_of "$name")
  local stage; stage=$(mktemp -d /tmp/p0-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' </dev/null || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" </dev/null || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  if ! (cd "$stage/x" && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1); then
    # files still being written during tar can mismatch; keep only verified files
    (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' | xargs -r rm -f)
    log "SYNC-PARTIAL vm=$name (files changing during tar skipped)"
  fi
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  mkdir -p "$RES/$lane"; rsync -a --files-from="$stage/ok.list" "$stage/x/" "$RES/$lane/"
  cp "$stage/x/SHA256SUMS" "$RES/$lane/SHA256SUMS.vm"
  local n bad; n=$(wc -l <"$stage/ok.list")
  # re-verify the merged copy against the VM's sums
  bad=$(cd "$RES/$lane" && grep -F -f <(sed 's/^/  /' "$stage/ok.list") SHA256SUMS.vm | sha256sum -c --quiet 2>&1 | wc -l)
  rm -rf "$stage"
  log "SYNC vm=$name lane=$lane verified_files=$n recheck_bad=$bad dest=$RES/$lane"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P0: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name=$name" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P0: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c cz; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book_row "$name" "$cz" "$c" "$ep"
  alert "P0 puzzle baselines: VM $name deleted ($reason); est. spend so far \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

gone_vm() {  # a VM of an active lane vanished (spot preemption or max-run DELETE): close its ledger row
  local name=$1 ep; ep=$(date +%s)
  grep -q " DELETE vm=$name " "$LEDGER" && return 0
  log "DELETE vm=$name epoch=$ep reason=\"vanished (spot preemption or max-run)\" spend_now=$(spend)"
  local c cz; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book_row "$name" "$cz" "$c" "$ep"
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since nmark missing
  local cyc=0
  log "WATCHDOG-START pid=$$"
  while [[ ! -f $STATE/watchdog.stop ]]; do
    cyc=$((cyc + 1))
    local list s; list=$(vms) || { echo "$(now) vms failed; skipping cycle"; sleep 60; continue; }; s=$(spend)
    if awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}'; then
      while read -r name _; do [[ -n $name ]] && { log "WATCHDOG vm=$name action=delete why=\"spend cap \$$s\""; cmd_delete "$name" "watchdog: spend cap \$$s"; }; done <<<"$list"
      for f in "$STATE"/lane.*; do [[ -f $f ]] && mv "$f" "$f.capped"; done
      alert "P0: spend cap reached (\$$s), all P0 VMs deleted"
      sleep 300; continue
    fi
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age lane; age=$(age_h "$created"); lane=$(lane_of "$name")
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="age ${age}h > ${MAX_AGE_H}h"
      [[ $status == TERMINATED || $status == STOPPED ]] && why="${why:+$why; }status $status"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [r]un_lane.py >/dev/null; then echo busy; elif sudo test -f /root/out/JOB_DONE; then echo done; else echo idle; fi; sudo ls /root/out/markers 2>/dev/null | wc -l' </dev/null 2>/dev/null | tr '\n' ' ' || echo unreachable)
        local nm=${st#* }; st=${st%% *}
        if [[ $st == busy ]]; then
          unset "idle_since[$name]"
          if [[ ${nmark[$name]:-} != "$nm" ]] || (( cyc % 3 == 0 )); then cmd_sync "$name" >/dev/null 2>&1 && nmark[$name]=$nm; fi
        else
          local plan; plan=$(awk '{print $1}' "$STATE/lane.$lane" 2>/dev/null)
          if [[ $st == done && ( $plan == main-* || $plan == pmr ) ]]; then
            why="plan $plan finished"
          else
            [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
            (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
          fi
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=down why=\"$why\""
        cmd_down "$name" "watchdog: $why"
        [[ $why == "plan "* ]] && mv "$STATE/lane.$lane" "$STATE/lane.$lane.finished" 2>/dev/null
        [[ $why == "queue "* || $why == age* ]] && mv "$STATE/lane.$lane" "$STATE/lane.$lane.stopped" 2>/dev/null
      fi
    done <<<"$list"
    # preemption: an active lane whose VM vanished -> recreate in the next zone and resume
    list=$(vms) || { sleep 300; continue; }
    for f in "$STATE"/lane.*; do
      [[ $f =~ \.(finished|stopped|capped)$ || ! -f $f ]] && continue
      local lane plan cuts last; lane=${f##*/lane.}; read -r plan cuts last <"$f"
      if grep -q "^$lane\(-r[0-9]*\)\? " <<<"$list"; then unset "missing[$lane]"; continue; fi
      # recreate only after the VM is missing in two cycles 5 min apart and a by-name lookup agrees
      missing[$lane]=$(( ${missing[$lane]:-0} + 1 ))
      (( missing[$lane] < 2 )) && continue
      [[ -n $(zone_of "$last") ]] && { unset "missing[$lane]"; continue; }
      gone_vm "$last"
      if [[ -f $RES/$lane/JOB_DONE ]]; then mv "$f" "$f.finished"; continue; fi
      local nrec; nrec=$(grep -c " CREATE vm=$lane-r[0-9]* " "$LEDGER")
      if (( nrec >= MAX_RECREATE )); then log "NO-RECREATE lane=$lane (already $nrec)"; alert "P0: $last vanished; recreate limit reached, lane $lane stopped"; mv "$f" "$f.stopped"; continue; fi
      local lz nz i; lz=$(grep " CREATE vm=$last " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/'); nz=${ZONES[0]}
      for i in "${!ZONES[@]}"; do [[ ${ZONES[$i]} == "$lz" ]] && nz=${ZONES[$(( (i + 1) % ${#ZONES[@]} ))]}; done
      local new="$lane-r$((nrec + 1))"
      log "RECREATE lane=$lane old=$last new=$new zone0=$nz"
      alert "P0: $last vanished (spot preemption?); recreating as $new from $nz, resuming $plan"
      if cmd_up "$new" "$nz" && cmd_start "$new" "$plan" "$([[ $cuts == - ]] || echo "$cuts")"; then :; else
        log "RECREATE-FAILED lane=$lane new=$new"; mv "$f" "$f.stopped"
      fi
    done
    [[ -z $(vms) ]] && ! ls "$STATE"/lane.* 2>/dev/null | grep -vqE '\.(finished|stopped|capped)$' && { log "WATCHDOG-IDLE no VMs, no active lanes"; }
    sleep 300
  done
  rm -f "$STATE/watchdog.stop"
  log "WATCHDOG-STOP"
}

case "${1:-}" in
  up) cmd_up "$2" "${3:-}" ;;
  start) cmd_start "$2" "$3" "${4:-}" ;;
  push) cmd_push "$2" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  spend) spend ;;
  list) vms ;;
  lanes) for f in "$STATE"/lane.*; do [[ -f $f ]] && echo "${f##*/}: $(cat "$f")"; done ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
