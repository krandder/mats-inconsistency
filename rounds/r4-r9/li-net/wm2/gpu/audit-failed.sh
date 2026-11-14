#!/bin/bash
# Zero the phantom reservation of wm2 VMs whose create failed (no instance ever existed): for every
# wm2 VM with PENDING-CREATE + CREATE-FAILED and no DELETE/COST-AUDIT line in GPU-LOG.md, confirm with
# gcloud that the instance does not exist, then append a COST-AUDIT cost_usd=0 line (spend_usd reads it).
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
python3 - "$GPU_LOG" <<'PY' | while read -r vm zones; do
import re, sys
res, done, failed = {}, set(), {}
for l in open(sys.argv[1]):
    m = re.search(r'vm=(soup-gpu-wm2\S*)', l)
    if not m: continue
    n = m.group(1)
    if ' DELETE ' in l or ' COST-AUDIT ' in l: done.add(n)
    if 'CREATE-FAILED' in l:
        z = re.search(r'zones=(.*?) err=', l); failed[n] = (z.group(1) if z else '').replace(' ', ',')
for n, z in failed.items():
    if n not in done: print(n, z)
PY
  exists=0
  for z in ${zones//,/ }; do
    "${GC[@]}" compute instances describe "$vm" --zone="$z" >/dev/null 2>&1 && exists=1
  done
  if [[ $exists == 0 ]]; then
    log_gpu "COST-AUDIT vm=$vm rate_usd_h=0.87 duration_s=0 age_h=0 cost_usd=0.03 basis=create-failed-instance-absent (stockout or spot preemption at create; at most ~2 min at the ceiling rate; instance absent in zones $zones; reserve released) by=wm2/gpu/audit-failed.sh"
    echo "audited $vm"
  else
    echo "WARNING $vm exists despite CREATE-FAILED; not audited" >&2
  fi
done
