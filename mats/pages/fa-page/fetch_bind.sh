#!/usr/bin/env bash
# Read-only copy of the World FA binding-arm training logs. Live VMs are read over ssh; a finished run is read from its
# synced copy under /tmp/l4runs/<run>/runs/<run>/runs/.
set -u
D=/home/agent/workspaces/mats/work/fa-page/bind_logs
for R in fa-bind-1028d fa-bind-1028e; do
  SYNC=/tmp/l4runs/$R/runs/$R/runs
  if grep -q "GPU-RUN EXIT" /tmp/l4runs/$R/gpu-run.out 2>/dev/null && [ -d "$SYNC" ]; then
    { date -u -r /tmp/l4runs/$R/gpu-run.out +%Y-%m-%dT%H:%MZ
      for f in $SYNC/*.log; do echo "== $(basename $f)"; grep -E "step|Error|TRAINED|DONE" $f | cut -c1-260; done; } > $D/$R.txt
    continue
  fi
  IP=$(grep -o 'ip=[0-9.]*' /tmp/l4runs/$R/GPU-LOG.md | tail -1 | cut -d= -f2)
  K=/tmp/l4runs/$R/infra/state/soup_ed25519
  OUT=$(timeout 60 ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 soup@$IP \
    "sudo bash -c 'date -u +%Y-%m-%dT%H:%MZ; for f in /root/runs/$R/runs/*.log; do echo \"== \$(basename \$f)\"; grep -E \"step|Error|TRAINED|DONE\" \$f | cut -c1-260; done'" 2>/dev/null)
  if [ -n "$OUT" ]; then echo "$OUT" > $D/$R.txt; else echo "fetch failed for $R; keeping the previous copy" >&2; fi
done
head -1 $D/fa-bind-1028d.txt $D/fa-bind-1028e.txt
