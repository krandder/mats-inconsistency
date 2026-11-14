#!/usr/bin/env python3
"""R7-E1 audit, check 7: recompute R7-E1's GPU spend (a) from the logical-soup/GPU-LOG.md minute stamps and (b) from the GCE
operation timestamps saved in gcloud-operations.txt (read-only `gcloud compute operations list`), at the ledger's ceiling
rate $0.87/h (on-demand list price; the spot VM's real price is lower). Usage: python3 cost.py > cost-output.txt"""
import re
import time
from datetime import datetime, timezone

RATE, CEIL = 0.87, 2.50
GPULOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
ts = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
ev = {}
for line in open(GPULOG):
    m = re.search(r"vm=(soup-gpu-r7e1-[a-z0-9-]+)", line)
    if not m:
        continue
    kind = line[20:].split()[0]
    ev.setdefault(m.group(1), []).append((line[2:19], kind, re.search(r"cost_usd=([0-9.]+)", line)))
print("== (a) GPU-LOG minute stamps")
tot_a, tot_logged = 0.0, 0.0
for vm, es in ev.items():
    kinds = {k: t for t, k, _ in es}
    logged = [float(c.group(1)) for _, _, c in es if c]
    tot_logged += logged[0] if logged else 0.0
    if "CREATE" in kinds:
        start = ts(kinds["CREATE"])
        end_key = "DELETE" if "DELETE" in kinds else "COST-AUDIT"
        end = ts(kinds[end_key])
        if vm.endswith("28650"):                 # preempted: billing ends at the preemption (10:32:07Z per the GCE operation)
            end = datetime(2026, 10, 17, 10, 32, 7, tzinfo=timezone.utc)
            end_key = "preemption 10:32:07Z"
        h = (end - start).total_seconds() / 3600
        tot_a += RATE * h
        print(f"{vm}: CREATE {kinds['CREATE']} -> {end_key}: {h:.3f} h x ${RATE} = ${RATE * h:.3f}; logged ${logged}")
    else:
        print(f"{vm}: no CREATE line ({sorted(kinds)}); logged {logged} -> $0")
print(f"total (a) ${tot_a:.3f}; logged total ${tot_logged:.2f}")

print("== (b) GCE operations (successful insert end -> preemption end / delete end)")
ops = []
for line in open("gcloud-operations.txt"):
    m = re.match(r"(\S+)\s+DONE\s+(\S+)\s+(\S+)\s+(soup-gpu-r7e1-\S+)", line)
    if m:
        ops.append((m.group(1), datetime.fromisoformat(m.group(2)), datetime.fromisoformat(m.group(3)), m.group(4)))
tot_b = 0.0
for vm in sorted({o[3] for o in ops}):
    ins = sorted(o for o in ops if o[3] == vm and o[0] == "insert")
    end = [o for o in ops if o[3] == vm and o[0] in ("delete", "compute.instances.preempted")]
    if not end:
        print(f"{vm}: {len(ins)} insert attempts, no delete/preemption -> never existed (create failed in every zone) -> $0")
        continue
    start = ins[-1][2]                           # the last insert (zone c) is the one that created the instance
    stop = end[0][2]
    h = (stop - start).total_seconds() / 3600
    tot_b += RATE * h
    print(f"{vm}: insert done {start.astimezone(timezone.utc):%H:%M:%S}Z -> {end[0][0]} done {stop.astimezone(timezone.utc):%H:%M:%S}Z: "
          f"{h:.3f} h x ${RATE} = ${RATE * h:.3f}")
print(f"total (b) ${tot_b:.3f} of the ${CEIL:.2f} ceiling: {'WITHIN' if tot_b <= CEIL else 'OVER'}")
