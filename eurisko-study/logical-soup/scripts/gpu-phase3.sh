#!/usr/bin/env bash
# One seed supervisor, executed only from the frozen launch directory.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT=$PWD
source infra/lib.sh
DRY=0; [[ ${1:-} != --dry-run ]] || { DRY=1; shift; }
seed=${1:?seed}; [[ $seed =~ ^[123]$ ]] || exit 2
id=phase3-k16-f12-s$seed
export GPU_BUDGET_GROUP=pool-sddpv GPU_GROUP_BUDGET_USD=40
export GPU_SOURCE_DIR="$RUN_ROOT" GPU_RUN_TIMEOUT=19500s
if [[ $DRY == 1 ]]; then
  spent=$(spend_usd pool-sddpv)
  printf 'seed=%s run=%s GPU=L4 epochs=5000 chunk=100 timeout=19500s TTL=6h max_attempts=7 no_progress_limit=2 group_spent_or_reserved=%s group_cap=40 new_reserve=5.22\n' "$seed" "$id" "$spent"
  awk -v s="$spent" 'BEGIN{exit (s+5.22>40)}'
  exit $?
fi
[[ ${GPU_PHASE3_FROZEN:-0} == 1 && $RUN_ROOT != "$SOUP_DIR" && -f source-commit.txt ]] || { echo 'use scripts/launch-gpu-phase3.sh; live-tree execution refused' >&2; exit 2; }
exec 7>"$STATE/supervisor-$id.lock"
flock -n 7 || { echo "already supervising $id" >&2; exit 2; }
epoch_now() {
  local dump="$SOUP_DIR/runs/phase3/k16-f12-s$seed/soup.dump"
  if [[ -f $dump ]]; then od -An -tu4 -j24 -N4 "$dump" | tr -d ' '; else echo 0; fi
}
no_progress=0
for attempt in {1..7}; do
  before=$(epoch_now)
  if ((before>=5000)); then
    python3 scripts/verify-checkpoint.py "$SOUP_DIR/runs/phase3/k16-f12-s$seed" 5000
    exit 0
  fi
  echo "$(date -u +%FT%TZ) attempt=$attempt seed=$seed epoch=$before"
  rc=0
  infra/gpu-run.sh --up --run-id "$id" --driver scripts/gpu-phase3-drive.sh -- "$seed" || rc=$?
  if [[ -f $STATE/vm-$id ]]; then
    read -r name zone < "$STATE/vm-$id"
    receipts="$SOUP_DIR/runs/phase3/infra/$name"
    if ! python3 scripts/gpu-rental-audit.py "$name" "$receipts"; then
      infra/gpu-down.sh --no-sync --only "$name" --reason 'retry cleanup after failed audit' || true
      python3 scripts/gpu-rental-audit.py "$name" "$receipts" || { post_status "logical-soup: cleanup audit failed for $name; refusing more rentals"; exit 1; }
    fi
    trash-put "$STATE/vm-$id"
    [[ ! -f $STATE/intent-$name ]] || trash-put "$STATE/intent-$name"
  fi
  after=$(epoch_now)
  if ((after>=5000)); then
    python3 scripts/verify-checkpoint.py "$SOUP_DIR/runs/phase3/k16-f12-s$seed" 5000
    exit 0
  fi
  if ((after>before)); then no_progress=0; else no_progress=$((no_progress+1)); fi
  [[ $rc != 3 ]] || { echo 'group budget admission refused' >&2; exit 3; }
  [[ $rc != 75 ]] || { echo 'chunk exceeded 30 minutes; refusing rental retries' >&2; exit 75; }
  ((no_progress<2)) || { echo 'two rentals without epoch progress; stopping' >&2; exit 4; }
  case $rc in 0|124|130|255) ;; *) echo "deterministic failure rc=$rc; refusing paid retries" >&2; exit "$rc";; esac
  sleep 10
done
echo 'retry limit reached' >&2; exit 1
