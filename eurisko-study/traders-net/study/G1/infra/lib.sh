# Shared settings and helpers for the traders-net GPU broker (study/G1). Sourced, not run.
# Adapted from logical-soup/infra/lib.sh (2026-09-27). Separate label, name prefix, key, log and
# budget, so the soup's watchdog and wealth-net's VM are never touched by these scripts and vice versa.
# shellcheck shell=bash
set -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
export HOME="${HOME:-/home/agent}"

INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G1_DIR="${G1_ROOT:-$(dirname "$INFRA")}"
STATE="$G1_DIR/infra/state"
QUEUE="$G1_DIR/queue"
GPU_LOG="$G1_DIR/GPU-LOG.md"
export GPU_LOG_PATH="$GPU_LOG"
PROJECT="GCP_PROJECT"
LABEL="traders-net"
PREFIX="tn-gpu-"
BUDGET_USD="${TN_BUDGET_USD:-45}"   # hard cap for all traders-net GPU (STUDY-PLAN: G1 refuses above $45)
MAX_VMS=2                           # at most 2 VMs at once
MAX_RUN_MIN_CAP=360                 # GCE auto-delete never later than 6 h
SETUP_MIN=20                        # boot + torch install margin added to each job's MAX_HOURS
SSH_USER="tn"
SSH_KEY="$STATE/tn_ed25519"
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu129-ubuntu-2404-nvidia-580"   # same image the soup verified 2026-09-27
BOOT_DISK_GB=100
MACHINE="g2-standard-8"; ACCEL="nvidia-l4"; ZONES=(us-central1-a us-central1-b us-central1-c)
GC=(gcloud --project="$PROJECT" --quiet)

mkdir -p "$STATE" "$QUEUE"/{pending,running,done,failed,refused}

# Ceiling, not a verified price: Cloud Billing API is disabled in GCP_PROJECT; public on-demand
# list price for g2-standard-8 (1x L4) in us-central1 plus the 100 GB pd-balanced disk.
rate_for() { case "$1" in g2-standard-8) echo "0.87" ;; *) echo "" ;; esac; }
RATE_NOTE="UNVERIFIED-ceiling"

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_gpu() {
  [[ -f "$GPU_LOG" ]] || printf '# GPU-LOG (traders-net, machine ledger; COST.md is derived from it)\n\n' >"$GPU_LOG"
  printf -- '- %s %s\n' "$(now_utc)" "$*" >>"$GPU_LOG"
}

# name zone creationTimestamp machineType labelled(yes/no) status
tn_vms() {
  "${GC[@]}" compute instances list \
    --filter="labels.${LABEL}:* OR name ~ ^${PREFIX}" \
    --format="value(name,zone.basename(),creationTimestamp,machineType.basename(),labels.${LABEL},status)" \
    2> >(grep -v "filter keys were not present" >&2) \
  | awk -F'\t' '{printf "%s %s %s %s %s %s\n",$1,$2,$3,$4,($5==""?"no":"yes"),$6}'
}
tn_disks() {
  "${GC[@]}" compute disks list --filter="name ~ ^${PREFIX} OR labels.${LABEL}:*" --format="value(name,zone.basename())"
}

# Spend (USD, ceiling) from GPU-LOG.md. Per VM: COST-AUDIT cost if present, else DELETE cost,
# else the CREATE/PENDING-CREATE reserve (worst case). Optional arg: only lines with ws=ARG.
spend_usd() {
  local ws=${1:-}
  [[ -f "$GPU_LOG" ]] || { echo 0; return; }
  awk -v ws="$ws" '
    function f(k,  i){ for(i=1;i<=NF;i++) if(index($i,k"=")==1) return substr($i,length(k)+2); return "" }
    / (CREATE|PENDING-CREATE) / { n=f("vm"); if(n=="") next; if(ws=="" || f("ws")==ws){ res[n]=f("reserve_usd") } }
    / DELETE / { n=f("vm"); c=f("cost_usd"); if(n!="" && c!="") del[n]=c }
    / COST-AUDIT / { n=f("vm"); c=f("cost_usd"); if(n!="" && c!="") aud[n]=c }
    END { t=0; for(n in res){ t += (n in aud ? aud[n] : (n in del ? del[n] : res[n])) } printf "%.2f\n", t }
  ' "$GPU_LOG"
}

# Per-workstream caps from infra/caps.tsv (ws<TAB>usd); unknown ws -> 0 (refused).
ws_cap() { awk -v w="$1" -F'\t' '$1==w{print $2; f=1} END{if(!f) print 0}' "$INFRA/caps.tsv"; }

age_hours() { local c; c=$(date -d "$1" +%s) || { echo 0; return; }; awk -v a="$(( $(date +%s) - c ))" 'BEGIN{printf "%.3f", a/3600}'; }

ensure_key() { [[ -f "$SSH_KEY" ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"; }
vm_ip() { "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes)
vm_ssh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"; }

post_status() {
  /home/agent/.local/bin/general-status-send --source traders-net-gpu "$*" >/dev/null 2>&1 \
    || echo "$(now_utc) general-status-send failed: $*" >>"$STATE/post-failures.log"
}

# GCE max-run-duration in minutes for a job of H hours: H*60 + SETUP_MIN, within [10, 360].
max_run_min() { awk -v h="$1" -v s="$SETUP_MIN" -v c="$MAX_RUN_MIN_CAP" 'BEGIN{m=int(h*60+s+0.999); if(m>c)m=c; if(m<10)m=10; print m}'; }
reserve_usd() { awk -v r="$(rate_for "$MACHINE")" -v m="$(max_run_min "$1")" 'BEGIN{printf "%.2f\n", r*m/60}'; }
valid_id() { [[ $1 =~ ^[a-z][a-z0-9-]{0,30}$ ]]; }
new_vm_name() { printf '%s%s-%s-%s\n' "$PREFIX" "$1" "$(date -u +%m%d%H%M)" "$RANDOM"; }
