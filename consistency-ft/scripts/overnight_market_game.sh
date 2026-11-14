#!/bin/bash
# Overnight frontier market-game campaign. Each configuration runs pass by pass; failed sessions
# are retried inside their pass (up to 3 attempts, with a 30-minute wait if nothing progresses,
# which is the quota-exhausted case) before the next pass starts, so no session ever trades on a
# later pass's market. Refusals that persist after 3 attempts are left failed.
# Usage: overnight_market_game.sh [WORKERS] [CONFIGS...]   (default 18 workers; configs luna-n8 astra-n16)
set -u
cd "$(dirname "$0")/.."
export PATH=/home/agent/.local/bin:$PATH
export PYTHONPATH=.:/home/agent/workspaces/bayes-engine
WORKERS=${1:-18}; shift || true
CONFIGS=${*:-"luna-n8 astra-n16"}
LOG=runs/market-game/overnight.log
mkdir -p runs/market-game
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

count() {  # out pass -> "completed failed"
  python3 - "$1" "$2" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1])/'sessions.jsonl'; want=int(sys.argv[2]); last={}
if p.exists():
    for l in p.read_text().splitlines():
        if l.strip():
            r=json.loads(l)
            if int(r.get('pass',0))==want: last[r.get('session_id')]=r.get('status')
print(sum(1 for s in last.values() if s=='completed'), sum(1 for s in last.values() if s=='failed'))
PY
}

run_config() {  # name provider effort n passes
  local name=$1 provider=$2 effort=$3 n=$4 passes=$5 out=runs/market-game/$1
  say "config $name: provider=$provider effort=$effort n=$n passes=$passes workers=$WORKERS"
  for pass in $(seq 1 "$passes"); do
    local last=-1
    for attempt in 1 2 3; do
      local resume=""; [ -f "$out/sessions.jsonl" ] && resume="--resume"
      MARKET_GAME_RETRY_FAILED=1 python3 scripts/run_market_game.py --provider "$provider" --effort "$effort" \
        --n-per-session "$n" --passes "$pass" --workers "$WORKERS" --out "$out" $resume ${DATA:+--data "$DATA"} >> "$out.log" 2>&1
      read -r completed failed <<< "$(count "$out" "$pass")"
      say "config $name pass $pass attempt $attempt: completed=$completed failed=$failed"
      [ "$failed" = "0" ] && break
      # a handful of persistent refusals is not a stall: move on when failures are under 5% of the pass
      total=$((completed + failed)); if [ "$total" -gt 0 ] && [ $((failed * 20)) -lt "$total" ] && [ "$attempt" -ge 2 ]; then say "config $name pass $pass: $failed persistent failures, continuing"; break; fi
      if [ "$completed" = "$last" ]; then say "config $name pass $pass: no progress, waiting 30 min (quota?)"; sleep 1800; fi
      last=$completed
    done
  done
  say "config $name finished"
}

for c in $CONFIGS; do
  case $c in
    astra-n8)  run_config astra-n8  astra high 8  3 ;;
    luna-n8)   run_config luna-n8   luna  high 8  3 ;;
    astra-n16) run_config astra-n16 astra high 16 3 ;;
    sol-train-n16) run_config sol-train-n16 sol high 16 ${SOL_PASSES:-1} ;;
    *) say "unknown config $c" ;;
  esac
done
say "campaign complete"
touch runs/market-game/CAMPAIGN_DONE
