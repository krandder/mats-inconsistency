#!/bin/bash
# P15 spot L4 VMs (adapted from puzzles/P12/gpu.sh).
#
#   gpu.sh up NAME            create trader-b3-p15-* (g2-standard-8 + 1x L4 SPOT, label traderb3=p15), wait READY
#   gpu.sh push NAME          push code + synced outputs (checkpoints: resume) to the VM
#   gpu.sh sh NAME 'cmd'      run a command in /root/job as root (dev/smoke); touches the heartbeat
#   gpu.sh start NAME [PLAN] [LANE]  push, then run run_vm.sh PLAN detached (resume-safe with marker files);
#                             each lane (default main) is auto-resumed on its own new VM after preemption
#   gpu.sh status NAME        markers, last log lines, GPU use
#   gpu.sh sync NAME          tar /root/out + SHA256SUMS -> verify -> merge into results/vm_out (one merged tree)
#   gpu.sh down NAME [why]    sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   gpu.sh watchdog           every 5 min (sync every 15 min); deletes at est. spend >= $17 (all p15 VMs summed),
#                             age > 5.9 h, or idle > 20 min; if gpu/WANT exists and no VM is up (spot preemption),
#                             re-creates the next VM and restarts the plan from the synced checkpoints
#   gpu.sh spend | list | check
# Stop guarantees: --max-run-duration=6h + --instance-termination-action=DELETE (+ boot disk auto-delete),
# the watchdog, and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x $0.86/h (g2-standard-8 + L4 + 50 GB pd-balanced on-demand list ceiling; spot is lower).
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P15="$(cd "$(dirname "$0")" && pwd)"
OUTD="$P15/results/vm_out"; STATE="$P15/gpu"; mkdir -p "$STATE" "$OUTD"
LEDGER="$STATE/LEDGER.md"
BOOK="$P15/../../GPU-LEDGER.md"
BUDGET="$P15/../../../BUDGET-24H-20260929.md"
PROJECT=GCP_PROJECT
MACHINE=g2-standard-8; ACCEL=nvidia-l4; RATE=0.86
# spend is counted at the published spot price (spec): g2-standard-8 US spot $0.5121/h (gcloud-compute.com, read
# 2026-09-30 00:10Z), rounded up to $0.52; RATE (on-demand list) stays as the ceiling shown in CREATE lines.
SPOT_RATE=${P15_SPOT_RATE:-0.52}
ZONES=(${P15_ZONES:-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b europe-west1-b})
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
CAP_USD=${P15_CAP_USD:-17}
MAX_RUN=${P15_MAX_RUN:-6h}
MAX_AGE_H=${P15_MAX_AGE_H:-5.9}
MAX_VMS=${P15_MAX_VMS:-3}
IDLE_MIN=20
SSH_USER=p15; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-b3-p15 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.traderb3=p15 AND name~^trader-b3-p15-" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
# lanes: several VMs run different plans; gpu/WANT[.<lane>] holds a lane's plan, gpu/VM.<name> the VM's lane
lane_of() { cat "$STATE/VM.$1" 2>/dev/null || echo main; }
want_file() { if [[ $1 == main ]]; then echo "$STATE/WANT"; else echo "$STATE/WANT.$1"; fi; }
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }
spend() {
  awk -v now="$(date +%s)" -v rate="$SPOT_RATE" '
    / CREATE / { r=rate; for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); } start[n]=e; rt[n]=r }
    / DELETE / { for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7) } if(!(n in stop)) stop[n]=e }
    END { t=0; for(n in start){ s=(n in stop?stop[n]:now); t+=(s-start[n])/3600*rt[n] } printf "%.2f\n", t }' "$LEDGER" 2>/dev/null || echo 0
}
budget_line() {  # budget_line "item text" USD: append a row to the 24h budget table with a running total
  python3 - "$BUDGET" "$1" "$2" <<'PY'
import sys, time, re
path, item, usd = sys.argv[1], sys.argv[2], float(sys.argv[3])
lines = open(path).read().rstrip("\n").split("\n")
tot = 0.0
for l in reversed(lines):
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if l.startswith("|") and len(c) >= 4:
        try:
            tot = float(re.sub(r"[^0-9.\-]", "", c[3])); break
        except ValueError:
            continue
row = f"| {time.strftime('%H:%M', time.gmtime())} | {item} | {usd:.2f} | {tot + usd:.2f} |"
# insert right after the last table row of the file
k = max(i for i, l in enumerate(lines) if l.startswith("|"))
lines.insert(k + 1, row)
open(path, "w").write("\n".join(lines) + "\n")
PY
}
book_row() {  # book_row NAME ZONE CREATED_EPOCH [DELETED_EPOCH]
  python3 - "$BOOK" "$@" "$RATE" "$MAX_RUN" <<'PY'
import sys, time
book, name, zone, c = sys.argv[1:5]
d = sys.argv[5] if len(sys.argv) > 7 else ""
rate, maxrun = sys.argv[-2], sys.argv[-1]
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## P15 tied-CA local EM (trader-b3-p15-*, cap $20)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + (f"\n\n{head}\nRate ceiling: g2-standard-8 + 1x L4 SPOT + 50 GB boot disk <= ${rate}/h (on-demand list price; spot is lower). "
                              "Watchdog: puzzles/P15/gpu.sh watchdog (5-min checks, 15-min sync + sha256; sums all p15 VM-hours; deletes at est. spend >= $17, age 5.9 h, or idle 20 min; at most 2 VMs).\n"
                              "| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n")
row0 = f"| {fmt(c)} | {name} | {zone} | g2-standard-8 + L4 spot, max-run {maxrun} |"
if d:
    h = (int(d) - int(c)) / 3600
    row = f"{row0} {fmt(d)} | {h:.2f} | {h * float(rate):.2f} |"
else:
    row = f"{row0} running | | |"
lines = txt.rstrip("\n").split("\n")
hi = lines.index(head)
for i, l in enumerate(lines):
    if l.startswith(row0):
        lines[i] = row
        break
else:
    j = hi + 1
    while j < len(lines) and not lines[j].startswith("## "):
        j += 1
    k = j
    while k > hi and not lines[k - 1].startswith("|"):
        k -= 1
    lines.insert(k, row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1
  [[ $name =~ ^trader-b3-p15-[a-z0-9-]+$ ]] || { echo "name must be trader-b3-p15-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$SPOT_RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5h > cap \$$CAP_USD" >&2; return 1; fi
  local nup; nup=$(vms | grep -c . || true)
  (( nup < MAX_VMS )) || { echo "REFUSE: $nup p15 VMs already up (max $MAX_VMS)" >&2; return 1; }
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p15-up.XXXXXX)
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
        --labels="traderb3=p15" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --metadata-from-file="startup-script=$P15/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-300)\""
    "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q . && cmd_delete "$name" "create failed"
    rm -rf "$tmp"; return 1
  fi
  rm -rf "$tmp"
  local ep; ep=$(date +%s)
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE spot_rate=$SPOT_RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  budget_line "P15 tied-CA local EM: VM $name created ($zone, L4 spot, max-run $MAX_RUN, ceiling \$$RATE/h; P15 est. so far \$$s of cap \$20)" 0
  alert "P15 tied-CA local EM: spot L4 VM $name created in $zone (self-deletes after $MAX_RUN; watchdog deletes at est. \$17, cap \$20)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st it=0
  while (( $(date +%s) < deadline )); do
    it=$((it + 1))
    if (( it % 3 == 0 )) && ! "${GC[@]}" compute instances list --filter="name=$name AND status=(PROVISIONING OR STAGING OR RUNNING)" --format='value(name)' 2>/dev/null | grep -q .; then
      log "STARTUP-PREEMPTED vm=$name"; cmd_delete "$name" "spot preempted during startup"; return 1
    fi
    st=$(rsh "$ip" 'cat /var/lib/p15/READY 2>/dev/null || { [ -f /var/lib/p15/FAILED ] && echo FAILED: $(cat /var/lib/p15/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p15-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; return 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; return 1
}

cmd_push() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job /root/out && sudo touch /root/out/HEARTBEAT' </dev/null || return 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$P15"/*.py "$P15"/*.sh "$P15"/queue*.json "$SSH_USER@$ip:/root/job/" </dev/null || return 1
  # published DT-recall checkpoints + Bansal reference code (read-only copies, fetched on General 09-29)
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" /tmp/p15ckpt /tmp/p15dt "$SSH_USER@$ip:/root/ext/" </dev/null || return 1
  if [[ -d $OUTD ]] && [[ -n $(ls -A "$OUTD" 2>/dev/null) ]]; then
    rsync -az --ignore-existing --rsync-path="sudo rsync" --exclude 'SHA256SUMS*' --exclude JOB_DONE --exclude JOB_RUNNING --exclude FAILED_STEPS --exclude HEARTBEAT --exclude run_vm.out --exclude markers/setup.done -e "ssh ${SSH_OPTS[*]}" "$OUTD/" "$SSH_USER@$ip:/root/out/" </dev/null
  fi
  log "PUSH vm=$name"
}

cmd_sh() {
  local name=$1 zone ip; shift; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" "sudo touch /root/out/HEARTBEAT; sudo bash -c $(printf '%q' "cd /root/job && export PATH=/opt/venv/bin:\$PATH PYTHONUNBUFFERED=1 && $*")"
}

cmd_start() {
  local name=$1 plan=${2:-main} lane=${3:-main} zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rm -f "$OUTD/JOB_DONE" "$STATE/DONE.$lane"; echo "$plan" >"$(want_file "$lane")"; echo "$lane" >"$STATE/VM.$name"
  cmd_push "$name" || return 1
  timeout 45 ssh -n "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo bash -c 'cd /root/job && PLAN=$plan setsid nohup bash run_vm.sh </dev/null >>/root/out/run_vm.out 2>&1 &'" || true
  sleep 5
  rsh "$ip" 'pgrep -f [r]un_vm.sh >/dev/null' </dev/null || { echo "run_vm.sh not running on $name" >&2; return 1; }
  log "START vm=$name plan=$plan lane=$lane"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE /root/out/FAILED_STEPS 2>/dev/null
ls /root/out/markers 2>/dev/null | tr '\n' ' '; echo
tail -6 /root/out/run_vm.out 2>/dev/null
for f in $(ls -t /root/out/logs/*.log 2>/dev/null | head -4); do echo "== $f"; tail -2 "$f" | cut -c1-300; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local stage; stage=$(mktemp -d /tmp/p15-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" --warning=no-file-changed . ; r=\$?; chmod 644 /tmp/out.tgz && [ \$r -le 1 ]"' </dev/null || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" </dev/null || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  if ! (cd "$stage/x" && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
    (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' | xargs -r rm -f)
    log "SYNC-PARTIAL vm=$name (files changing during tar skipped)"
  fi
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  rsync -a -u --files-from="$stage/ok.list" "$stage/x/" "$OUTD/"
  cp "$stage/x/SHA256SUMS" "$OUTD/SHA256SUMS.$name"
  local n; n=$(wc -l <"$stage/ok.list")
  rm -rf "$stage"
  log "SYNC vm=$name verified_files=$n dest=$OUTD"
}

cmd_check() {
  echo "instances:"; "${GC[@]}" compute instances list --filter="name~trader-b3-p15" --format='value(name,zone.basename(),status)'
  echo "disks:"; "${GC[@]}" compute disks list --filter="name~trader-b3-p15" --format='value(name,zone.basename())'
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  [[ $name =~ ^trader-b3-p15-[a-z0-9-]+$ ]] || { echo "refuse: $name is not trader-b3-p15-*" >&2; return 2; }
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P15: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-b3-p15-" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P15: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c cz; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  if [[ -n $c ]]; then
    book_row "$name" "$cz" "$c" "$ep"
    budget_line "P15 tied-CA local EM: VM $name deleted ($reason), $(awk -v a=$((ep - c)) 'BEGIN{printf "%.2f", a/3600}') VM-h at the \$$SPOT_RATE/h published spot price; P15 est. total \$$(spend)" "$(awk -v a=$((ep - c)) -v r=$SPOT_RATE 'BEGIN{printf "%.2f", a/3600*r}')"
  fi
  alert "P15 tied-CA local EM: VM $name deleted ($reason); est. P15 spend so far \$$(spend) (ceiling)"
}

cmd_down() {
  local name=$1 reason=${2:-down} zone ip lane; lane=$(lane_of "$name")
  cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"
  zone=$(zone_of "$name")
  if [[ -n $zone ]]; then ip=$(ip_of "$name" "$zone")
    if rsh "$ip" 'sudo test -f /root/out/JOB_DONE && ! pgrep -f "[r]un_vm.sh" >/dev/null' </dev/null 2>/dev/null; then
      touch "$STATE/DONE.$lane"; log "LANE-DONE lane=$lane vm=$name plan=\"$(cat "$(want_file "$lane")" 2>/dev/null)\""
    fi
  fi
  cmd_delete "$name" "$reason"
}

next_name() {
  local n; n=$(grep -o 'CREATE vm=trader-b3-p15-[0-9]*' "$LEDGER" 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  echo "trader-b3-p15-$(( ${n:-0} + 1 ))"
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  local cyc=0
  while [[ ! -f $STATE/watchdog.stop ]]; do
    cyc=$((cyc + 1))
    local list s; list=$(vms); s=$(spend)
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]]; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if pgrep -f "[r]un_vm.sh|/opt/venv/bin/[p]ython" >/dev/null; then echo busy; elif sudo test -f /root/out/JOB_DONE; then echo done; elif sudo find /root/out/HEARTBEAT -mmin -20 2>/dev/null | grep -q .; then echo busy; else echo idle; fi' </dev/null 2>/dev/null || echo unreachable)
        if [[ $st == done ]]; then why="plan finished"
        elif [[ $st == busy ]]; then unset "idle_since[$name]"; (( cyc % 3 == 0 )) && { cmd_sync "$name" >/dev/null 2>&1 || true; }; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="$st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done <<<"$list"
    # ledger: VMs created but no longer listed and not deleted by us = preempted (termination action DELETE)
    for n in $(grep -o 'CREATE vm=trader-b3-p15-[0-9a-z-]*' "$LEDGER" | sed 's/CREATE vm=//' | sort -u); do
      grep -q " DELETE vm=$n " "$LEDGER" && continue
      "${GC[@]}" compute instances list --filter="name=$n" --format='value(name)' 2>/dev/null | grep -q . && continue
      local ep; ep=$(date +%s)
      log "DELETE vm=$n epoch=$ep reason=\"gone (spot preemption or max-run; termination action DELETE)\" leftover_disks=\"$("${GC[@]}" compute disks list --filter="name~^trader-b3-p15-" --format='value(name)' | tr '\n' ' ')\" spend_now=$(spend)"
      local c cz; c=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$n " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
      book_row "$n" "$cz" "$c" "$ep"
      budget_line "P15 tied-CA local EM: VM $n gone (spot preemption or max-run), $(awk -v a=$((ep - c)) 'BEGIN{printf "%.2f", a/3600}') VM-h at the \$$SPOT_RATE/h published spot price; P15 est. total \$$(spend)" "$(awk -v a=$((ep - c)) -v r=$SPOT_RATE 'BEGIN{printf "%.2f", a/3600*r}')"
      alert "P15: spot VM $n is gone (preempted or max-run); est. P15 spend \$$(spend)"
    done
    # auto-resume: a plan is wanted, nothing is up, the plan is not done, and there is budget
    local live_lanes=" " vn; for vn in $(vms | awk '{print $1}'); do live_lanes+="$(lane_of "$vn") "; done
    local wf lane
    for wf in "$STATE"/WANT "$STATE"/WANT.*; do
      [[ -f $wf ]] || continue
      lane=${wf##*/WANT}; lane=${lane#.}; lane=${lane:-main}
      [[ -f $STATE/DONE.$lane || $live_lanes == *" $lane "* ]] && continue
      local s2; s2=$(spend)
      if awk -v s="$s2" -v r="$SPOT_RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5<=c)}'; then
        local nn; nn=$(next_name)
        log "RESUME lane=$lane plan=$(cat "$wf") vm=$nn"
        if cmd_up "$nn" && cmd_start "$nn" "$(cat "$wf")" "$lane"; then live_lanes+="$lane "; else log "RESUME-FAILED vm=$nn lane=$lane"; fi
      fi
    done
    sleep 300
  done
  rm -f "$STATE/watchdog.stop"
}

case "${1:-}" in
  up) cmd_up "$2" ;;
  start) cmd_start "$2" "${3:-main}" "${4:-main}" ;;
  push) cmd_push "$2" ;;
  sh) n=$2; shift 2; cmd_sh "$n" "$@" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  spend) spend ;;
  next) next_name ;;
  list) vms ;;
  check) cmd_check ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
