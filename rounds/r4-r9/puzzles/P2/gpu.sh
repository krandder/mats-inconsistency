#!/bin/bash
# P2 spot L4 VM (pattern: consistency-ft scripts/gpu/distill_gpu.sh + market-resnet/watchdog.sh, as in puzzles/P6).
#
#   gpu.sh up NAME              ledger row first, then create trader-cat-p2-* (g2-standard-4 + 1x L4 SPOT,
#                               label trader-cat=p2), wait READY
#   gpu.sh push NAME [FROM]     upload code + pre-generated data (sha256-checked on the VM); FROM = an earlier
#                               VM's synced results dir to resume from after a preemption
#   gpu.sh start NAME PLAN [K=V ...]  run queue_p2.py PLAN (smoke | main | alt) detached, with env K=V
#                               (P2_ARMS, P2_TRAP, P2_DEADLINE, P2_EVAL_SLOTS; see queue_p2.py)
#   gpu.sh status NAME          queue state, last log lines, GPU use
#   gpu.sh sync NAME            tar /root/out + SHA256SUMS -> scp -> sha256sum -c -> results/vm/NAME/
#   gpu.sh down NAME [why]      sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   gpu.sh watchdog             every 5 min: spend/age check; sync every 15 min; delete on spend >= CAP,
#                               age, or idle queue > 20 min
#   gpu.sh spend | list | gone
# Stop guarantees: --max-run-duration=5h + --instance-termination-action=DELETE (+ boot disk auto-delete),
# the watchdog, and `down` after the job with an instances/disks listing check (`gone`).
# Spend = VM-hours x RATE, RATE = $0.75/h ceiling for g2-standard-4 + L4 + 50 GB disk (spot is lower).
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P2="$(cd "$(dirname "$0")" && pwd)"
VMRES="$P2/results/vm"; STATE="$P2/gpu"; mkdir -p "$STATE" "$VMRES"
LEDGER="$STATE/LEDGER.md"
BOOK="$P2/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=g2-standard-4; ACCEL=nvidia-l4; RATE=0.75
ZONES=(${P2_ZONES:-us-central1-a us-central1-b us-central1-c})
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
CAP_USD=${P2_CAP_USD:-6}
MAX_RUN=${P2_MAX_RUN:-5h}
MAX_AGE_H=${P2_MAX_AGE_H:-4.9}
IDLE_MIN=20
SYNC_MIN=15
SSH_USER=p2; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)
CODE_FILES=(jm.py gen.py train.py eval.py tests.py queue_p2.py bench.sh)
DATA_FILES=(pool_n9.npz test.npz dev.npz ext.npz sets.meta.json)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-cat-p2 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="name~^trader-cat-p2-" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
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
book() {  # book NAME STATE [ZONE] [CREATED_EPOCH] [DELETED_EPOCH]; STATE = creating | running | failed | deleted
  python3 - "$BOOK" "$RATE" "$MAX_RUN" "$@" <<'PY'
import sys, time
book, rate, maxrun, name, state = sys.argv[1:6]
zone = sys.argv[6] if len(sys.argv) > 6 else "us-central1-?"
c = sys.argv[7] if len(sys.argv) > 7 else str(int(time.time()))
d = sys.argv[8] if len(sys.argv) > 8 else ""
fmt = lambda e: time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(e)))
head = "## P2 joint-structure markets (trader-cat-p2-*, cap $8)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + (f"\n\n{head}\nRate ceiling: g2-standard-4 + 1x L4 SPOT + 50 GB boot disk <= ${rate}/h (on-demand list "
                              "price; spot is lower). Watchdog: puzzles/P2/gpu.sh watchdog (5-min checks, 15-min sync + sha256; "
                              "deletes at est. spend >= $6, age 4.9 h, or idle 20 min).\n"
                              "| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n")
shape = f"g2-standard-4 + L4 spot, max-run {maxrun}"
if state == "deleted":
    h = (int(d) - int(c)) / 3600
    row = f"| {fmt(c)} | {name} | {zone} | {shape} | {fmt(d)} | {h:.2f} | {h * float(rate):.2f} |"
elif state == "failed":
    row = f"| {fmt(c)} | {name} | {zone} | {shape} | create failed (nothing created) | 0 | 0 |"
else:
    row = f"| {fmt(c)} | {name} | {zone} | {shape} | {state} | | |"
lines = txt.rstrip("\n").split("\n")
start = lines.index(head)
for i in range(start, len(lines)):
    if i > start and lines[i].startswith("## "):
        j = i
        while lines[j - 1] == "":
            j -= 1
        lines.insert(j, row)
        break
    if lines[i].startswith("| ") and f"| {name} |" in lines[i] and ("| creating |" in lines[i] or "| running |" in lines[i]):
        lines[i] = row
        break
else:
    lines.append(row)
open(book, "w").write("\n".join(lines) + "\n")
PY
}

cmd_up() {
  local name=$1
  [[ $name =~ ^trader-cat-p2-[a-z0-9-]+$ ]] || { echo "name must be trader-cat-p2-*" >&2; exit 2; }
  local s; s=$(spend)
  if awk -v s="$s" -v r="$RATE" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5h > cap \$$CAP_USD" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p2-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local c0; c0=$(date +%s)
  book "$name" creating "us-central1-?" "$c0"   # ledger row BEFORE instances create
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
        --labels="trader-cat=p2" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --metadata-from-file="startup-script=$P2/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
      zone=$z; break
    fi
    cat "$tmp/err" >&2
    grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available|QUOTA|unsupported|does not exist in zone' "$tmp/err" || break
  done
  if [[ -z $zone ]]; then
    log "CREATE-FAILED vm=$name err=\"$(tr '\n' ' ' <"$tmp/err" | cut -c1-200)\""
    if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
      log "CREATE vm=$name zone=unknown epoch=$c0 type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE (partial create)"
      cmd_delete "$name" "create failed"
    else
      book "$name" failed "-" "$c0"
    fi
    rm -rf "$tmp"; exit 1
  fi
  rm -rf "$tmp"
  local ep; ep=$(date +%s)
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book "$name" running "$zone" "$c0"
  alert "P2 joint-structure markets: spot L4 VM $name created in $zone (self-deletes after $MAX_RUN; cap \$$CAP_USD)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p2/READY 2>/dev/null || { [ -f /var/lib/p2/FAILED ] && echo FAILED: $(cat /var/lib/p2/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p2-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_push() {
  local name=$1 from=${2:-} zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job/data /root/job/results /root/out' || exit 1
  local tmp; tmp=$(mktemp -d /tmp/p2-push.XXXXXX)
  (cd "$P2" && sha256sum "${CODE_FILES[@]}" && cd data && sha256sum "${DATA_FILES[@]}" | sed 's#  #  data/#') >"$tmp/SUMS"
  (cd "$P2" && rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "${CODE_FILES[@]}" "$SSH_USER@$ip:/root/job/") || exit 1
  (cd "$P2/data" && rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "${DATA_FILES[@]}" "$SSH_USER@$ip:/root/job/data/") || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$tmp/SUMS" "$SSH_USER@$ip:/root/job/UPLOAD_SUMS" || exit 1
  rsh "$ip" 'sudo bash -c "cd /root/job && sha256sum -c --quiet UPLOAD_SUMS"' || { log "PUSH-VERIFY-FAILED vm=$name"; exit 1; }
  rm -rf "$tmp"
  if [[ -n $from ]]; then  # resume after a preemption: completed runs/evals and partial checkpoints
    rsync -az --rsync-path="sudo rsync" --exclude 'SHA256SUMS*' --exclude 'JOB_*' --exclude 'queue_state.json' \
      -e "ssh ${SSH_OPTS[*]}" "$from/" "$SSH_USER@$ip:/root/out/" || exit 1
  fi
  log "PUSH vm=$name code+data sha256 verified${from:+ resume_from=$from}"
}

cmd_start() {
  local name=$1 plan=$2 zone ip kv; shift 2
  for kv in "$@"; do [[ $kv =~ ^P2_[A-Z_]+=[A-Za-z0-9,._-]+$ ]] || { echo "bad env $kv" >&2; exit 2; }; done
  zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  # systemd-run returns at once (a nohup'd background job under sudo kept the ssh session open)
  local se="" o="/root/out/queue-$plan.out"; for kv in "$@"; do se+=" --setenv=$kv"; done
  rsh "$ip" "sudo systemd-run --unit=p2q-$plan-$(date +%s) --collect --working-directory=/root/job$se -p StandardOutput=append:$o -p StandardError=append:$o /opt/venv/bin/python queue_p2.py $plan" </dev/null || exit 1
  log "START vm=$name plan=$plan $*"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE /root/out/JOB_FAILED 2>/dev/null
python3 -c "import json;s=json.load(open('/root/out/queue_state.json'));print(s['plan'],'done',len(s['done']),'/',s['total'],'running',s['running'],'failed',s['failed'],'blocked',len(s['blocked']))" 2>/dev/null
tail -6 /root/out/logs/queue.log 2>/dev/null
for f in $(ls -t /root/out/logs/*.log 2>/dev/null | grep -v queue.log | head -4); do echo "== $f"; tail -1 "$f" | cut -c1-300; done
tail -8 /root/out/gpu_util.csv 2>/dev/null | tr '\n' ' '; echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local dest="$VMRES/$name"; mkdir -p "$dest"
  local stage; stage=$(mktemp -d /tmp/p2-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  if ! (cd "$stage/x" && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
    # files still being written during tar can mismatch; keep only verified files
    (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' | xargs -r rm -f)
    log "SYNC-PARTIAL vm=$name (files changing during tar skipped)"
  fi
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  rsync -a --files-from="$stage/ok.list" "$stage/x/" "$dest/"
  awk 'NR==FNR{ok[$0]=1;next} ($2 in ok)' "$stage/ok.list" "$stage/x/SHA256SUMS" >"$dest/SHA256SUMS.verified"
  local n bad; n=$(wc -l <"$stage/ok.list")
  bad=$(cd "$dest" && sha256sum -c --quiet SHA256SUMS.verified 2>&1 | wc -l)
  rm -rf "$stage"
  log "SYNC vm=$name verified_files=$n dest=$dest recheck_mismatches=$bad"
  echo "$(date +%s)" >"$STATE/last_sync_$name"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P2: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p2" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P2: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c cz; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book "$name" deleted "$cz" "$c" "$ep"
  alert "P2 joint-structure markets: VM $name deleted ($reason); est. spend so far \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

cmd_gone() {
  local i d; i=$("${GC[@]}" compute instances list --filter="name~^trader-cat-p2" --format='value(name,zone.basename(),status)')
  d=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p2" --format='value(name,zone.basename())')
  echo "instances(^trader-cat-p2): ${i:-none}"; echo "disks(^trader-cat-p2): ${d:-none}"
  [[ -z $i && -z $d ]]
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $STATE/watchdog.stop ]]; do
    local list s listed=0; if list=$(vms); then listed=1; fi; s=$(spend)
    while read -r name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [q]ueue_p2.py >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
        local last; last=$(cat "$STATE/last_sync_$name" 2>/dev/null || echo 0)
        if (( $(date +%s) - last >= SYNC_MIN * 60 )) && [[ $st != unreachable ]]; then cmd_sync "$name" >/dev/null 2>&1 || true; fi
        if [[ $st == busy ]]; then unset "idle_since[$name]"; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done <<<"$list"
    # a VM that vanished (spot preemption / max-run) while the ledger says it is up: record the delete
    (( listed )) && for n in $(grep -o ' CREATE vm=[^ ]*' "$LEDGER" 2>/dev/null | sed 's/.*vm=//' | sort -u); do
      if ! grep -q " DELETE vm=$n " "$LEDGER" && ! awk -v n="$n" '$1==n{f=1} END{exit !f}' <<<"$list"; then
        log "WATCHDOG vm=$n vanished (preempted or max-run); recording delete"
        cmd_delete "$n" "vanished: spot preemption or max-run"
      fi
    done
    sleep 300
  done
  rm -f "$STATE/watchdog.stop"
}

case "${1:-}" in
  up) cmd_up "$2" ;;
  push) cmd_push "$2" "${3:-}" ;;
  start) cmd_start "${@:2}" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  spend) spend ;;
  list) vms ;;
  gone) cmd_gone ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
