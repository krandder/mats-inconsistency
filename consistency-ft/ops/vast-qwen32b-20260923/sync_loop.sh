#!/bin/bash
# Pull /workspace/q32/out from the GPU box every 3 min. Refresh the watchdog lease only if the sync worked
# AND something is alive: the remote job heartbeat (<15 min) or the controlling agent's heartbeat (<30 min).
cd /home/agent/data/qwen32b-baseline-20260923/ops
K=[ssh-key-path-removed]
mkdir -p ../remote-out
while [ ! -f stop-sync ]; do
  if rsync -az -e "ssh -i $K -p 34742 -o ConnectTimeout=20" root@[host-removed]:/workspace/q32/out/ ../remote-out/ > sync.log 2>&1; then
    now=$(date +%s); alive=0
    if [ -f ../remote-out/heartbeat.json ]; then
      hb=$(python3 -c "import json;print(int(json.load(open('../remote-out/heartbeat.json'))['at']))" 2>/dev/null || echo 0)
      [ $((now-hb)) -lt 900 ] && alive=1
    fi
    [ -f agent-heartbeat ] && [ $((now-$(stat -c %Y agent-heartbeat))) -lt 1800 ] && alive=1
    [ $alive = 1 ] && echo "{\"phase\":\"synced\",\"at\":$now}" > lease.json
    date -u +%FT%TZ > last-sync.txt
  fi
  sleep 180
done
