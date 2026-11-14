#!/bin/bash
# Rebuild the market-game dashboard from the run directories every 5 minutes and push it to the site.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:/home/agent/workspaces/bayes-engine
while true; do
  runs=$(ls -d runs/market-game/astra-n8 runs/market-game/luna-n8 runs/market-game/astra-n16 2>/dev/null)
  if [ -n "$runs" ]; then
    python3 scripts/build_market_game_page.py $runs >> runs/market-game/deploy.log 2>&1 \
      && scp -q site/market-game.html deploy-host:fleet/apps/consistency-dash/market-game.html \
      && date -u +%FT%TZ > runs/market-game/last-deploy.txt
  fi
  [ -f runs/market-game/CAMPAIGN_DONE ] && [ -f runs/market-game/last-deploy.txt ] && break
  sleep 300
done
