#!/bin/bash
# P11 spot L4 VM (pattern: consistency-ft scripts/gpu/distill_gpu.sh + market-resnet/watchdog.sh).
#
#   gpu.sh up NAME            create trader-cat-p11-* (g2-standard-4 + 1x L4 SPOT, label trader-cat=p11), wait READY
#   gpu.sh start NAME PLAN    push code, data + current results (resume), run queue_p11.py PLAN detached
#   gpu.sh status NAME        queue state, last log lines, GPU use
#   gpu.sh sync NAME          tar /root/out + SHA256SUMS -> verify -> merge into results/
#   gpu.sh down NAME [why]    sync, DELETE the VM (+ boot disk), verify instances AND disks are gone
#   gpu.sh watchdog           every 5 min: sync; delete on spend >= $CAP, age, or idle queue > 20 min
#   gpu.sh relay              every 2 min: sync every started VM into results/vm, then copy the union of results
#                             to each VM (never overwriting a VM's own files) so sharded queues see each other
#   gpu.sh keep SHARD         keep one VM (trader-cat-p11-s<SHARD>-HHMM) running queue shard SHARD: re-create it after a
#                             spot preemption (results resume from results/vm + ckpt.pt), `down` it when the shard's
#                             DONE-s<SHARD>.json / FAILED-s<SHARD>.json has synced; at most 5 creations per shard
#   gpu.sh spend | list
# Env for up/start: P11_MACHINE (g2-standard-4 | g2-standard-8), P11_SHARD, P11_NSHARD, P11_WORKERS, P11_MPS=1
# Stop guarantees: --max-run-duration=6h + --instance-termination-action=DELETE (+ boot disk auto-delete),
# the watchdog, and `down` after the job with an instances/disks listing check.
# Spend = VM-hours x RATE, where RATE = $0.71/h is a ceiling for g2-standard-4 + L4 on demand incl. disk
# (spot is lower; the Billing API is not readable here). The watchdog's cap check uses P11_WATCH_RATE if set
# (the spec's own watchdog estimate is $0.32/h for spot; 0.45 keeps a 40% margin over it), else RATE.
set -u -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
P11="$(cd "$(dirname "$0")" && pwd)"
RES="$P11/results/vm"; STATE="$P11/gpu"; DATA=${P11_DATA:-/tmp/p11-data}; mkdir -p "$STATE" "$RES"
LEDGER="$STATE/LEDGER.md"
BOOK="$P11/../../GPU-LEDGER.md"
PROJECT=GCP_PROJECT
MACHINE=${P11_MACHINE:-g2-standard-4}; ACCEL=nvidia-l4
case $MACHINE in g2-standard-4) RATE=0.71 ;; g2-standard-8) RATE=0.86 ;; *) echo "unknown machine $MACHINE" >&2; exit 2 ;; esac
ZONES=(${P11_ZONES:-us-central1-a us-central1-b us-central1-c})
IMAGE_PROJECT=deeplearning-platform-release; IMAGE_FAMILY=common-cu129-ubuntu-2404-nvidia-580
CAP_USD=${P11_CAP_USD:-7.5}
MAX_RUN=${P11_MAX_RUN:-6h}
MAX_AGE_H=${P11_MAX_AGE_H:-5.9}
IDLE_MIN=20
SSH_USER=p11; SSH_KEY=[ssh-key-path-removed]
GC=(gcloud --project="$PROJECT" --quiet)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf -- '- %s %s\n' "$(now)" "$*" >>"$LEDGER"; echo "$*"; }
alert() { /home/agent/.local/bin/general-status-send --source trader-cat-p11 "$*" >/dev/null 2>&1 || true; }
vms() { "${GC[@]}" compute instances list --filter="labels.trader-cat=p11" --format="value(name,zone.basename(),creationTimestamp,status)" 2>/dev/null; }
zone_of() { "${GC[@]}" compute instances list --filter="name=$1" --format='value(zone.basename())' 2>/dev/null; }
ip_of() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
rsh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }
# never copied between VMs: per-VM queue state, logs, checksums, model weights (only results + gate files travel)
RELAY_EXCL=(--exclude 'SHA256SUMS*' --exclude 'JOB_*' --exclude 'queue_state*' --exclude 'queue-*.out' --exclude 'logs/' --exclude '*.tmp')
age_h() { awk -v a="$(( $(date +%s) - $(date -d "$1" +%s) ))" 'BEGIN{printf "%.3f", a/3600}'; }
spend() {  # spend [RATE_OVERRIDE]
  awk -v now="$(date +%s)" -v rate="$RATE" -v ovr="${1:-}" '
    / CREATE / { r=rate; for(i=1;i<=NF;i++){ if($i~/^vm=/)n=substr($i,4); if($i~/^epoch=/)e=substr($i,7); if($i~/^rate_ceiling=/)r=substr($i,14) } if(ovr!="")r=ovr; start[n]=e; rt[n]=r }
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
head = "## P11 wealth cap (trader-cat-p11-*, cap $8)"
txt = open(book).read()
if head not in txt:
    txt = txt.rstrip("\n") + f"\n\n{head}\nRate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= ${rate}/h (on-demand list price; spot is lower). Watchdog: puzzles/P11/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $7.5, age, or idle 20 min).\n| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |\n|---|---|---|---|---|---|---|\n"
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

reconcile_gone() {  # VMs that vanished without a DELETE line (spot preemption + termination action DELETE): log the
  # DELETE with the preemption time (else now) so spend() stops counting them, and complete their GPU-LEDGER row.
  local n z c t e left
  for n in $(awk '/ CREATE /{for(i=1;i<=NF;i++) if($i~/^vm=/) print substr($i,4)}' "$LEDGER" | sort -u); do
    grep -q "DELETE vm=$n " "$LEDGER" && continue
    left=$("${GC[@]}" compute instances list --filter="name=$n" --format='value(name)' 2>/dev/null) || continue
    [[ -z $left ]] || continue
    z=$(awk -v n="vm=$n" '/ CREATE / && index($0, n" "){for(i=1;i<=NF;i++) if($i~/^zone=/) print substr($i,6)}' "$LEDGER" | tail -1)
    c=$(awk -v n="vm=$n" '/ CREATE / && index($0, n" "){for(i=1;i<=NF;i++) if($i~/^epoch=/) print substr($i,7)}' "$LEDGER" | tail -1)
    t=$("${GC[@]}" compute operations list --filter="targetLink~/$n\$ AND operationType~preempted" --format='value(insertTime)' 2>/dev/null | head -1)
    if [[ -n $t ]]; then e=$(date -d "$t" +%s); else e=$(date +%s); fi
    left=$("${GC[@]}" compute disks list --filter="name=$n" --format='value(name)' 2>/dev/null)
    log "DELETE vm=$n epoch=$e reason=\"${t:+spot preempted at $(date -u -d "@$e" +%H:%M:%SZ) (GCE operation compute.instances.preempted); }gone from the instance list; termination action deleted VM + disk\" leftover_disks=\"${left:-none}\" spend_now=$(spend)"
    [[ -n $z && -n $c ]] && book_row "$n" "$z" "$c" "$e"
    alert "P11: $n gone (${t:+spot preempted; }disk: ${left:-none left})"
  done
}

cmd_up() {
  local name=$1
  [[ $name =~ ^trader-cat-p11-[a-z0-9-]+$ ]] || { echo "name must be trader-cat-p11-*" >&2; exit 2; }
  # the cap counts VM-hours at the watch rate (P11_WATCH_RATE, $0.45/h from 14:10Z), like keep and watchdog
  local s; s=$(spend "${P11_WATCH_RATE:-}")
  if awk -v s="$s" -v r="${P11_WATCH_RATE:-$RATE}" -v c="$CAP_USD" 'BEGIN{exit !(s+r*0.5>c)}'; then echo "REFUSE: spend \$$s + 0.5h > cap \$$CAP_USD" >&2; exit 1; fi
  [[ -f $SSH_KEY ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
  local tmp; tmp=$(mktemp -d /tmp/p11-up.XXXXXX)
  printf '%s:%s\n' "$SSH_USER" "$(cat "$SSH_KEY.pub")" >"$tmp/ssh-keys"
  local zone=""
  for z in "${ZONES[@]}"; do
    echo "creating $name in $z ..."
    if "${GC[@]}" compute instances create "$name" --zone="$z" \
        --machine-type="$MACHINE" --accelerator="count=1,type=$ACCEL" \
        --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="$MAX_RUN" \
        --maintenance-policy=TERMINATE --no-restart-on-failure \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=60GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
        --network=default --no-service-account --no-scopes \
        --labels="trader-cat=p11" \
        --metadata=block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --metadata-from-file="startup-script=$P11/startup.sh,ssh-keys=$tmp/ssh-keys" 2>"$tmp/err"; then
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
  log "CREATE vm=$name zone=$zone epoch=$ep type=$MACHINE+L4 spot max_run=$MAX_RUN rate_ceiling=$RATE spend_before=$s"
  book_row "$name" "$zone" "$ep"
  alert "P11 wealth cap: spot L4 VM $name created in $zone (self-deletes after $MAX_RUN; cap \$$CAP_USD)"
  local ip; ip=$(ip_of "$name" "$zone")
  echo "waiting for $name ($ip) startup (up to 25 min) ..."
  local deadline=$(( $(date +%s) + 1500 )) st
  while (( $(date +%s) < deadline )); do
    st=$(rsh "$ip" 'cat /var/lib/p11/READY 2>/dev/null || { [ -f /var/lib/p11/FAILED ] && echo FAILED: $(cat /var/lib/p11/FAILED); }' 2>/dev/null || true)
    if [[ $st == FAILED* ]]; then rsh "$ip" 'sudo tail -30 /var/log/p11-startup.log' 2>/dev/null; log "STARTUP-FAILED vm=$name $st"; cmd_delete "$name" "startup failed"; exit 1; fi
    if [[ -n $st ]]; then log "READY vm=$name ip=$ip $(echo "$st" | tr '\n' ' ')"; return 0; fi
    sleep 20
  done
  log "STARTUP-TIMEOUT vm=$name"; cmd_delete "$name" "startup timeout"; exit 1
}

cmd_push() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo mkdir -p /root/job /root/out' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$P11"/*.py "$P11/VENDORED.json" "$SSH_USER@$ip:/root/job/" || exit 1
  rsh "$ip" 'sudo mkdir -p /root/job/data/sets' || exit 1
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$DATA/pool9.npz" "$SSH_USER@$ip:/root/job/data/" || exit 1
  if [[ ${P11_PUSH_800K:-0} == 1 && -f $DATA/pool9_800k.npz ]]; then rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$DATA/pool9_800k.npz" "$SSH_USER@$ip:/root/job/data/" || exit 1; fi
  rsync -az --rsync-path="sudo rsync" -e "ssh ${SSH_OPTS[*]}" "$DATA/sets/" "$SSH_USER@$ip:/root/job/data/sets/" || exit 1
  # integrity of pushed code + data
  local want got
  want=$( (cd "$P11" && sha256sum *.py VENDORED.json; cd "$DATA" && sha256sum pool9.npz sets/*.npz) | sort -k2)
  got=$(rsh "$ip" 'sudo bash -c "cd /root/job && sha256sum *.py VENDORED.json; cd /root/job/data && sha256sum pool9.npz sets/*.npz"' | sort -k2)
  [[ "$want" == "$got" ]] || { echo "sha256 mismatch after push" >&2; diff <(echo "$want") <(echo "$got") >&2; exit 1; }
  # resume: push whatever results exist locally
  mkdir -p "$RES"
  rsync -az --ignore-existing --rsync-path="sudo rsync" "${RELAY_EXCL[@]}" -e "ssh ${SSH_OPTS[*]}" "$RES/" "$SSH_USER@$ip:/root/out/" || exit 1
}

cmd_start() {
  local name=$1 plan=$2 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  cmd_push "$name" || exit 1
  local sh=${P11_SHARD:-0} ns=${P11_NSHARD:-1} w=${P11_WORKERS:-4} mps=""
  [[ ${P11_MPS:-0} == 1 ]] && mps="nvidia-cuda-mps-control -d || true; "
  timeout 90 ssh -n "${SSH_OPTS[@]}" "$SSH_USER@$ip" "sudo bash -c '${mps}cd /root/job && P11_SHARD=$sh P11_NSHARD=$ns P11_WORKERS=$w setsid nohup /opt/venv/bin/python queue_p11.py $plan </dev/null >/root/out/queue-$plan-s$sh.out 2>&1 &' </dev/null >/dev/null 2>&1" || true
  sleep 5
  rsh "$ip" 'pgrep -af "[q]ueue_p11.py"' </dev/null || { echo "queue did not start on $name" >&2; exit 1; }
  log "START vm=$name plan=$plan shard=$sh/$ns workers=$w mps=${P11_MPS:-0}"
}

cmd_status() {
  local name=$1 zone ip; zone=$(zone_of "$name"); ip=$(ip_of "$name" "$zone")
  rsh "$ip" 'sudo bash -s' <<'REMOTE'
cat /root/out/JOB_DONE /root/out/JOB_FAILED 2>/dev/null
python3 -c "import json;s=json.load(open('/root/out/queue_state.json'));print('done',len(s['done']),'/',s['total'],'running',s['running'],'failed',s['failed'],'blocked',len(s['blocked']))" 2>/dev/null
tail -5 /root/out/logs/queue.log 2>/dev/null
for f in $(ls -t /root/out/logs/*.log 2>/dev/null | grep -v queue.log | head -4); do echo "== $f"; tail -1 "$f" | cut -c1-300; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime
REMOTE
}

cmd_sync() {
  local name=$1 zone ip; zone=$(zone_of "$name"); [[ -n $zone ]] || { echo "$name not found" >&2; return 1; }
  ip=$(ip_of "$name" "$zone")
  local stage; stage=$(mktemp -d /tmp/p11-sync.XXXXXX)
  rsh "$ip" 'sudo bash -c "cd /root/out && find . -type f ! -name SHA256SUMS ! -name \"*.tmp\" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && tar -czf /tmp/out.tgz --exclude=\"*.tmp\" . && chmod 644 /tmp/out.tgz"' || { rm -rf "$stage"; return 1; }
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:/tmp/out.tgz" "$stage/out.tgz" || { rm -rf "$stage"; return 1; }
  mkdir -p "$stage/x" && tar -C "$stage/x" -xzf "$stage/out.tgz" 2>/dev/null
  if ! (cd "$stage/x" && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
    # files still being written during tar can mismatch; keep only verified files
    (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' | xargs -r rm -f)
    log "SYNC-PARTIAL vm=$name (files changing during tar skipped)"
  fi
  (cd "$stage/x" && sha256sum -c SHA256SUMS 2>/dev/null | grep ': OK$' | sed 's/: OK$//' >"$stage/ok.list")
  # --update: a VM also holds relayed copies of other shards' in-progress files (ckpt.pt, logs); never let such an
  # older copy overwrite the fresher one the owning shard delivered (it cost a resume from step 400 at 16:32Z)
  rsync -a --update --files-from="$stage/ok.list" "$stage/x/" "$RES/"
  cp "$stage/x/SHA256SUMS" "$RES/SHA256SUMS.$name"
  local n; n=$(wc -l <"$stage/ok.list")
  rm -rf "$stage"
  log "SYNC vm=$name verified_files=$n dest=$RES"
}

cmd_delete() {
  local name=$1 reason=${2:-} zone; zone=$(zone_of "$name")
  if [[ -n $zone ]]; then "${GC[@]}" compute instances delete "$name" --zone="$zone" </dev/null || true; fi
  if "${GC[@]}" compute instances list --filter="name=$name" --format='value(name)' | grep -q .; then
    log "DELETE-FAILED vm=$name reason=\"$reason\""; alert "P11: FAILED to delete VM $name, check now"; return 1
  fi
  local disks; disks=$("${GC[@]}" compute disks list --filter="name~^trader-cat-p11" --format='value(name,zone.basename())')
  [[ -n $disks ]] && alert "P11: leftover disks after deleting $name: $disks"
  local ep; ep=$(date +%s)
  log "DELETE vm=$name epoch=$ep reason=\"$reason\" leftover_disks=\"${disks:-none}\" spend_now=$(spend)"
  local c; c=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*epoch=\([0-9]*\).*/\1/'); local cz; cz=$(grep " CREATE vm=$name " "$LEDGER" | tail -1 | sed 's/.*zone=\([^ ]*\).*/\1/')
  [[ -n $c ]] && book_row "$name" "$cz" "$c" "$ep"
  alert "P11 wealth cap: VM $name deleted ($reason); est. spend so far \$$(spend) (ceiling)"
}

cmd_down() { local name=$1 reason=${2:-down}; cmd_sync "$name" || log "SYNC-FAILED vm=$name (deleting anyway)"; cmd_delete "$name" "$reason"; }

cmd_relay() {
  exec 8>"$STATE/relay.lock"; flock -n 8 || { echo "relay already running"; exit 0; }
  while [[ ! -f $STATE/relay.stop ]]; do
    local list; list=$(vms)
    while read -r -u 3 name zone created status; do
      [[ -n $name && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER" || continue
      cmd_sync "$name" </dev/null >/dev/null 2>&1 || echo "$(now) relay: sync $name failed"
    done 3<<<"$list"
    while read -r -u 3 name zone created status; do
      [[ -n $name && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER" || continue
      local ip; ip=$(ip_of "$name" "$zone")
      rsync -az --ignore-existing --rsync-path="sudo rsync" "${RELAY_EXCL[@]}" -e "ssh ${SSH_OPTS[*]}" "$RES/" "$SSH_USER@$ip:/root/out/" </dev/null >/dev/null 2>&1 \
        || echo "$(now) relay: push to $name failed"
    done 3<<<"$list"
    sleep 120
  done
  rm -f "$STATE/relay.stop"
}

cmd_keep() {
  local sh=$1; export P11_SHARD=$sh
  exec 7>"$STATE/keep-s$sh.lock"; flock -n 7 || { echo "keep s$sh already running"; exit 0; }
  local n=0 name
  while [[ ! -f $STATE/keep.stop ]]; do
    name=$(vms | awk -v p="^trader-cat-p11-s$sh-" '$1 ~ p {print $1; exit}')
    if [[ -f $RES/DONE-s$sh.json || -f $RES/FAILED-s$sh.json ]]; then
      if [[ -n $name ]]; then cmd_down "$name" "shard $sh finished ($(ls "$RES"/*-s$sh.json | xargs -n1 basename | tr '\n' ' '))"; fi
      echo "$(now) keep s$sh: finished"; break
    fi
    if [[ -z $name ]]; then
      reconcile_gone
      local sw; sw=$(spend "${P11_WATCH_RATE:-}")
      if awk -v s="$sw" -v c="$CAP_USD" 'BEGIN{exit !(s>=c-0.25)}'; then
        log "KEEP shard=$sh not re-creating: est. spend \$$sw within \$0.25 of cap \$$CAP_USD"; alert "P11: shard $sh stopped at spend cap"; break
      fi
      if (( n >= 5 )); then log "KEEP shard=$sh gave up after $n creations"; alert "P11: shard $sh gave up after $n VM creations"; break; fi
      n=$((n + 1)); name="trader-cat-p11-s$sh-$(date -u +%H%M)"
      log "KEEP shard=$sh creating $name (creation $n)"
      if ! (cmd_up "$name") || ! (cmd_start "$name" all); then
        log "KEEP shard=$sh up/start of $name failed"
        local z; z=$(zone_of "$name"); [[ -n $z ]] && cmd_down "$name" "keep: start failed"
        sleep 240
      fi
    fi
    sleep 60
  done
}

cmd_watchdog() {
  exec 9>"$STATE/watchdog.lock"; flock -n 9 || { echo "watchdog already running"; exit 0; }
  declare -A idle_since
  while [[ ! -f $STATE/watchdog.stop ]]; do
    reconcile_gone
    local list s; list=$(vms); s=$(spend "${P11_WATCH_RATE:-}")
    while read -r -u 3 name zone created status; do
      [[ -n $name ]] || continue
      local why="" age; age=$(age_h "$created")
      awk -v s="$s" -v c="$CAP_USD" 'BEGIN{exit !(s>=c)}' && why="spend cap: \$$s >= \$$CAP_USD"
      awk -v a="$age" -v m="$MAX_AGE_H" 'BEGIN{exit !(a>m)}' && why="${why:+$why; }age ${age}h > ${MAX_AGE_H}h"
      if [[ -z $why && $status == RUNNING ]] && grep -q "START vm=$name " "$LEDGER"; then
        local ip st; ip=$(ip_of "$name" "$zone")
        st=$(rsh "$ip" 'if sudo test -f /root/out/JOB_RUNNING && pgrep -f [q]ueue_p11.py >/dev/null; then echo busy; else echo idle; fi' 2>/dev/null || echo unreachable)
        if [[ $st == busy ]]; then unset "idle_since[$name]"; cmd_sync "$name" >/dev/null 2>&1 || true; else
          [[ -n ${idle_since[$name]:-} ]] || idle_since[$name]=$(date +%s)
          (( $(date +%s) - idle_since[$name] > IDLE_MIN * 60 )) && why="queue $st for > ${IDLE_MIN} min"
        fi
      fi
      if [[ -n $why ]]; then
        log "WATCHDOG vm=$name action=delete why=\"$why\""
        if [[ $why == spend* ]]; then cmd_delete "$name" "watchdog: $why"; else cmd_down "$name" "watchdog: $why"; fi
      fi
    done 3<<<"$list"
    sleep 300
  done
  rm -f "$STATE/watchdog.stop"
}

case "${1:-}" in
  up) cmd_up "$2" ;;
  push) cmd_push "$2" ;;
  start) cmd_start "$2" "$3" ;;
  status) cmd_status "$2" ;;
  sync) cmd_sync "$2" ;;
  down) cmd_down "$2" "${3:-down}" ;;
  delete) cmd_delete "$2" "${3:-manual}" ;;
  watchdog) cmd_watchdog ;;
  relay) cmd_relay ;;
  keep) cmd_keep "$2" ;;
  reconcile) reconcile_gone ;;
  spend) echo "ceiling \$$(spend) (rate $RATE); watch \$$(spend "${P11_WATCH_RATE:-0.45}") (rate ${P11_WATCH_RATE:-0.45}); cap \$$CAP_USD" ;;
  exec) n=$2; shift 2; z=$(zone_of "$n"); rsh "$(ip_of "$n" "$z")" "$@" ;;
  list) vms ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
