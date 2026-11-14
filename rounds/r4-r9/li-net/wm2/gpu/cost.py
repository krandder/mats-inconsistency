"""wm2 GPU cost from logical-soup/GPU-LOG.md: per VM the last DELETE / COST-AUDIT cost_usd (rate ceiling x age),
VMs still without one counted at their reservation. Prints totals by machine type and provisioning."""
import collections
import json
import re
import sys

LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
vms = {}
for l in open(LOG):
    m = re.search(r"vm=(soup-gpu-wm2\S*)", l)
    if not m:
        continue
    n = m.group(1)
    v = vms.setdefault(n, {"type": None, "model": None, "reserve": 0.0, "cost": None, "age_h": None, "failed": False})
    if " PENDING-CREATE " in l or " CREATE vm=" in l:
        r = re.search(r"reserve_usd=([\d.]+)", l)
        if r:
            v["reserve"] = float(r.group(1))
        t = re.search(r"type=(\S+)", l)
        if t:
            v["type"] = t.group(1)
        mo = re.search(r"model=(\S+)", l)
        if mo:
            v["model"] = mo.group(1)
    if "CREATE-FAILED" in l:
        v["failed"] = True
    if " DELETE vm=" in l or " COST-AUDIT vm=" in l:
        c = re.search(r"cost_usd=([\d.]+)", l)
        a = re.search(r"age_h=([\d.]+)", l)
        if c:
            v["cost"] = float(c.group(1))
        if a:
            v["age_h"] = float(a.group(1))
tot = sum(v["cost"] if v["cost"] is not None else v["reserve"] for v in vms.values())
open_ = [n for n, v in vms.items() if v["cost"] is None]
by = collections.defaultdict(lambda: [0, 0.0, 0.0])
for n, v in vms.items():
    k = (v["type"] or "?", v["model"] or ("failed" if v["failed"] else "?"))
    by[k][0] += 1
    by[k][1] += v["cost"] or 0.0
    by[k][2] += v["age_h"] or 0.0
out = {"total_usd": round(tot, 2), "vms": len(vms), "open_reservations": open_,
       "by_type": {f"{a}/{b}": {"vms": c, "usd": round(u, 2), "vm_hours": round(h, 2)} for (a, b), (c, u, h) in by.items()}}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
