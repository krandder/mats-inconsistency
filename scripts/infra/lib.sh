# Shared settings and helpers for the logical-soup GPU scripts. Sourced, not run.
# shellcheck shell=bash
set -o pipefail
export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin:$PATH"
export HOME="${HOME:-/home/agent}"

INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOUP_DIR="${GPU_CONTROL_ROOT:-$(dirname "$INFRA")}"
STATE="$SOUP_DIR/infra/state"
GPU_LOG="$SOUP_DIR/GPU-LOG.md"
export GPU_LOG_PATH="$GPU_LOG"
PROJECT="GCP_PROJECT"
REGION="us-central1"
LABEL="logical-soup"
BUDGET_USD=660
MAX_RUN="${GPU_MAX_RUN:-6h}" # per-run shorter GCE limit; reserve remains conservative at 6h
case "$MAX_RUN" in
  6h|55m|28m) ;;
  *) echo 'GPU_MAX_RUN must be 6h, 55m or 28m' >&2; return 2 ;;
esac
MAX_RUN_H=6
SSH_USER="soup"
SSH_KEY="$STATE/soup_ed25519"
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu129-ubuntu-2404-nvidia-580"   # CUDA 12.9, driver 580; verified 2026-09-27
BEND_REF="v2.0.29"
BEND_LOCAL="/home/agent/workspaces/eurisko-study/research/bend2-scratch/bend-2.0.29"
BUN_VERSION="1.3.14"   # same as this box
BOOT_DISK_GB=100
GC=(gcloud --project="$PROJECT" --quiet)

mkdir -p "$STATE"

# Hourly rate ceilings in USD. The Cloud Billing API is disabled in this project,
# so no rate is verified: these are the public on-demand list prices for us-central1
# (machine incl. GPU) plus the 100 GB pd-balanced boot disk (~$0.014/h). Spot is
# normally 60-70% lower, so budgeting with these over-counts spend.
rate_for() {
  case "$1" in
    g2-standard-8)  echo "0.87" ;;   # 1x L4, 8 vCPU, 32 GB
    a2-highgpu-1g)  echo "3.69" ;;   # 1x A100 40GB, 12 vCPU, 85 GB
    *)              echo "" ;;
  esac
}
RATE_NOTE="UNVERIFIED ceiling: on-demand list price, Billing API disabled; spot is lower"

now_utc() { date -u +%Y-%m-%dT%H:%MZ; }

log_gpu() {  # append one line to GPU-LOG.md
  [[ -f "$GPU_LOG" ]] || printf '# GPU-LOG\n\n' >"$GPU_LOG"
  printf -- '- %s %s\n' "$(now_utc)" "$*" >>"$GPU_LOG"
}

# All VMs that belong to the soup: labelled logical-soup, or named soup-gpu-*.
# Output: name zone creationTimestamp machineType-basename labelled(yes/no) status
soup_vms() {
  "${GC[@]}" compute instances list \
    --filter="labels.${LABEL}:* OR name ~ ^soup-gpu-" \
    --format="value(name,zone.basename(),creationTimestamp,machineType.basename(),labels.${LABEL},status)" \
    2> >(grep -v "filter keys were not present" >&2) \
  | awk -F'\t' '{printf "%s %s %s %s %s %s\n",$1,$2,$3,$4,($5==""?"no":"yes"),$6}'
}

# Estimated spend so far from GPU-LOG.md: per VM, the DELETE line's cost_usd if
# present, else the CREATE line's reserve_usd (worst case: full MAX_RUN).
spend_usd() {
  local group=${1:-}
  [[ -f "$GPU_LOG" ]] || { echo 0; return; }
  awk -v group="$group" '
    / (CREATE|PENDING-CREATE) / { n=""; g=""; for(i=1;i<=NF;i++){if($i ~ /^vm=/)n=substr($i,4); if($i ~ /^group=/)g=substr($i,7)} if(group=="" || group==g)allowed[n]=1 }
    / (CREATE|PENDING-CREATE) / { n=""; r=0; for(i=1;i<=NF;i++){ if($i ~ /^vm=/){n=substr($i,4)} if($i ~ /^reserve_usd=/){r=substr($i,13)} } if(n!=""){res[n]=r} }
    / (DELETE|COST-AUDIT) / { n=""; c=""; for(i=1;i<=NF;i++){ if($i ~ /^vm=/){n=substr($i,4)} if($i ~ /^cost_usd=/){c=substr($i,10)} } if(n!="" && c!=""){cost[n]=c} }
    END { t=0; for(n in res){ if(allowed[n]) t += (n in cost ? cost[n] : res[n]) } for(n in cost){ if(group=="" && !(n in res)) t+=cost[n] } printf "%.2f\n", t }
  ' "$GPU_LOG"
}

age_hours() {  # creationTimestamp -> hours as decimal
  local c; c=$(date -d "$1" +%s) || { echo 0; return; }
  awk -v a="$(( $(date +%s) - c ))" 'BEGIN{printf "%.2f", a/3600}'
}

ensure_key() {
  [[ -f "$SSH_KEY" ]] || ssh-keygen -q -t ed25519 -N '' -C "$SSH_USER" -f "$SSH_KEY"
}

vm_ip() {  # name zone -> external IP
  "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
}

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30)
vm_ssh() {  # ip cmd...
  local ip=$1; shift
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@"
}

post_status() {
  /home/agent/.local/bin/general-status-send --source logical-soup-gpu "$*" >/dev/null 2>&1 \
    || echo "$(now_utc) general-status-send failed: $*" >>"$STATE/post-failures.log"
}

valid_run_id() { [[ $1 =~ ^[a-z][a-z0-9-]{0,30}$ ]]; }
# Exact selection; a named rental never owns another run's VM.
run_vms() {
  local run=$1
  "${GC[@]}" compute instances list --filter="labels.${LABEL}:* AND labels.soup-run=$run" \
    --format='value(name,zone.basename(),creationTimestamp,machineType.basename(),status)'
}

# Keep named-run ownership visible to legacy helpers without changing soup_vms.
vm_run_id() {
  "${GC[@]}" compute instances describe "$1" --zone="$2" --format='value(labels.soup-run)'
}
new_vm_name() { printf 'soup-gpu-%s%s-%s\n' "${1:+$1-}" "$(date -u +%m%d%H%M%S)" "$RANDOM"; }

# Commands enter /root/soup only inside the privileged remote shell.
vm_root() {
  local ip=$1; shift
  { printf 'cd /root/soup\n'; printf '%q ' "$@"; printf '\n'; } | vm_ssh "$ip" 'sudo bash -l -s'
}
