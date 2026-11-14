#!/bin/bash
# Spot VMs preempted mid-run vanish (termination action DELETE) and gpu-down.sh logs no DELETE line for them,
# so their full reservation stays in spend_usd. For every smallnet VM with CREATE and RUN-END but no DELETE /
# COST-AUDIT line, confirm with gcloud that it no longer exists and append COST-AUDIT with cost = ceiling rate x
# (RUN-END time - CREATE time) (an upper bound: the VM died at or before RUN-END).
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
python3 - "$GPU_LOG" <<'PY' | while read -r vm zone rate hours; do
import re, sys
from datetime import datetime
cre, end, done, info = {}, {}, set(), {}
for l in open(sys.argv[1]):
    m = re.search(r'vm=(soup-gpu-smallnet\S*)', l)
    if not m: continue
    n = m.group(1)
    ts = re.match(r'- (\S+)', l).group(1).rstrip('Z')
    t = datetime.strptime(ts[:16], '%Y-%m-%dT%H:%M')
    if ' CREATE vm=' in l:
        cre[n] = t
        z = re.search(r'zone=(\S+)', l); r = re.search(r'rate_usd_h=([\d.]+)', l)
        info[n] = (z.group(1) if z else 'us-central1-a', r.group(1) if r else '0.87')
    if ' RUN-END vm=' in l: end[n] = t
    if ' DELETE vm=' in l or ' COST-AUDIT vm=' in l: done.add(n)
for n in cre:
    if n in end and n not in done:
        h = max((end[n] - cre[n]).total_seconds() / 3600 + 1 / 60, 0)
        print(n, info[n][0], info[n][1], f"{h:.3f}")
PY
  if "${GC[@]}" compute instances describe "$vm" --zone="$zone" >/dev/null 2>&1; then continue; fi
  cost=$(awk -v r="$rate" -v h="$hours" 'BEGIN{printf "%.2f", r*h}')
  log_gpu "COST-AUDIT vm=$vm rate_usd_h=$rate age_h=$hours cost_usd=$cost basis=spot-preempted-mid-run (instance absent; CREATE to RUN-END at the ceiling rate, an upper bound) by=smallnet/gpu/audit-preempted.sh"
  echo "audited $vm $cost"
done
